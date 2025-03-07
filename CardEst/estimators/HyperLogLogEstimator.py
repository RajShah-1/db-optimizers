from .CardinalityEstimator import CardinalityEstimator
import re
import hashlib
import numpy as np

class HyperLogLogEstimator(CardinalityEstimator):
    """Cardinality estimator based on HyperLogLog algorithm"""
    
    def __init__(self, db_connection, precision=14):
        super().__init__(db_connection)
        self.precision = precision  # Number of bits for register indexing (between 4 and 16)
        self.m = 1 << precision    # Number of registers
        self.alpha = self._get_alpha(self.m)  # Correction factor
        self.hll_sketches = {}
        self.build_sketches()
    
    def _get_alpha(self, m):
        """Get the alpha constant for bias correction"""
        if m == 16:
            return 0.673
        elif m == 32:
            return 0.697
        elif m == 64:
            return 0.709
        else:
            return 0.7213 / (1 + 1.079 / m)
    
    def _hash_function(self, value):
        """Hash function returning 64-bit hash value"""
        return int(hashlib.md5(str(value).encode()).hexdigest(), 16)
    
    def _leading_zeros(self, value, p):
        """Count leading zeros + 1 in the hash value after using p bits for register index"""
        if value == 0:
            return 64 - p + 1
        return min(64 - p + 1, (64 - p) - int(np.floor(np.log2(value))))
    
    def build_sketches(self):
        """Build HyperLogLog sketches for all columns in the database"""
        cursor = self.conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        for table in tables:
            # Get all columns for the table
            cursor.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}'")
            columns = [row[0] for row in cursor.fetchall()]
            
            # Build HLL sketches for each join column (primary/foreign keys)
            # In a real implementation, we would focus on join columns, but here we do all for simplicity
            for column in columns:
                # Initialize registers for this column
                registers = [0] * self.m
                
                # Fetch column values and update registers
                cursor.execute(f"SELECT {column} FROM {table}")
                for row in cursor.fetchall():
                    if row[0] is not None:
                        # Hash the value
                        hash_val = self._hash_function(row[0])
                        
                        # Use p bits to determine the register
                        register_idx = hash_val & (self.m - 1)
                        
                        # Calculate the number of leading zeros (+1)
                        zeros = self._leading_zeros(hash_val >> self.precision, self.precision)
                        
                        # Update the register if the new value is larger
                        registers[register_idx] = max(registers[register_idx], zeros)
                
                # Store the HLL sketch
                self.hll_sketches[(table, column)] = registers
    
    def _hll_cardinality(self, registers):
        """Estimate cardinality from a HyperLogLog register set"""
        # Compute the harmonic mean
        sum_inv = sum(2 ** (-r) for r in registers)
        estimate = self.alpha * (self.m ** 2) / sum_inv
        
        # Apply small and large range corrections
        if estimate <= 2.5 * self.m:  # Small range correction
            # Count number of zero registers
            zeros = registers.count(0)
            if zeros > 0:
                estimate = self.m * np.log(self.m / zeros)
        elif estimate > (1 / 30) * (1 << 32):  # Large range correction
            estimate = -1 * (1 << 32) * np.log(1 - estimate / (1 << 32))
        
        return estimate
    
    def _merge_sketches(self, sketch1, sketch2):
        """Merge two HyperLogLog sketches (for OR operations)"""
        return [max(r1, r2) for r1, r2 in zip(sketch1, sketch2)]
    
    def _intersect_sketches(self, sketch1, sketch2):
        """Estimate intersection of two HLL sketches (for AND operations)"""
        # Using the inclusion-exclusion principle
        card1 = self._hll_cardinality(sketch1)
        card2 = self._hll_cardinality(sketch2)
        merged = self._merge_sketches(sketch1, sketch2)
        union_card = self._hll_cardinality(merged)
        
        # Estimate intersection cardinality
        intersection_card = max(0, card1 + card2 - union_card)
        return intersection_card
    
    def estimate_cardinality(self, query):
        """Estimate query cardinality using HyperLogLog sketches"""
        # Parse the query to find tables, joins, and predicates
        tables, join_conditions, predicates = self._parse_query(query)
        
        # Start with base table cardinalities
        table_cards = {}
        for table in tables:
            cursor = self.conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            table_cards[table] = cursor.fetchone()[0]
        
        # Process simple table predicates
        table_sketches = {}
        for table in tables:
            # Start with a "full" sketch for each table
            # In a real implementation, we would have pre-computed sketches
            # Here we use a simplified approach
            table_sketches[table] = None
        
        # Process joins
        final_card = self._estimate_join_cardinality(tables, join_conditions, table_cards)
        
        return final_card
    
    def _parse_query(self, query):
        """Simple parser to extract tables, joins, and predicates from a query"""
        # This is a simplified parser - in practice, you'd use a proper SQL parser
        tables = re.findall(r'FROM\s+([a-zA-Z0-9_]+)|JOIN\s+([a-zA-Z0-9_]+)', query)
        tables = [t[0] or t[1] for t in tables]
        
        join_conditions = re.findall(r'([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)\s*=\s*([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)', query)
        
        predicates = []
        # Extract simple predicates like "table.column > value"
        pred_patterns = re.findall(r'([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)\s*([<>=]+)\s*([0-9.]+)', query)
        for p in pred_patterns:
            predicates.append((p[0], p[1], p[2], float(p[3])))
        
        return tables, join_conditions, predicates
    
    def _estimate_join_cardinality(self, tables, join_conditions, table_cards):
        """Estimate join cardinality using HyperLogLog sketches"""
        if not tables:
            return 0
        
        # For HLL, we need to consider the join graph and estimate based on sketch intersections
        # For simplicity, we use a more traditional approach here
        
        # Start with cross product
        total_card = 1
        for table in tables:
            total_card *= table_cards[table]
        
        # For each join condition, estimate selectivity using HLL sketches when available
        for t1, c1, t2, c2 in join_conditions:
            sketch1 = self.hll_sketches.get((t1, c1))
            sketch2 = self.hll_sketches.get((t2, c2))
            
            if sketch1 and sketch2:
                # Get distinct counts using HLL
                distinct1 = self._hll_cardinality(sketch1)
                distinct2 = self._hll_cardinality(sketch2)
                
                # Estimate join selectivity
                selectivity = 1 / max(distinct1, distinct2)
            else:
                # Fall back to simple estimation
                cursor = self.conn.cursor()
                cursor.execute(f"SELECT COUNT(DISTINCT {c1}) FROM {t1}")
                distinct1 = max(1, cursor.fetchone()[0])
                cursor.execute(f"SELECT COUNT(DISTINCT {c2}) FROM {t2}")
                distinct2 = max(1, cursor.fetchone()[0])
                
                selectivity = 1 / max(distinct1, distinct2)
            
            total_card *= selectivity
        
        return total_card
    
    def _estimate_predicate_selectivity(self, table, column, op, value):
        """Estimate predicate selectivity using data distributions"""
        # This is a simplified approach
        # In a real implementation, we would use more advanced methods
        
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        total_rows = cursor.fetchone()[0]
        
        # For equality predicates on columns with HLL sketches, we can estimate selectivity
        if op == '=':
            sketch = self.hll_sketches.get((table, column))
            if sketch:
                distinct_vals = self._hll_cardinality(sketch)
                return 1 / distinct_vals
        
        # Default selectivity estimates
        if op == '=':
            return 0.05  # Assume 5% selectivity for equality
        elif op in ['<', '>']:
            return 0.33  # Assume 33% selectivity for inequalities
        
        return 1.0
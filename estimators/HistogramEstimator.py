from .CardinalityEstimator import CardinalityEstimator
import re

class HistogramEstimator(CardinalityEstimator):
    """Cardinality estimator based on column histograms"""
    
    def __init__(self, db_connection, num_buckets=100):
        super().__init__(db_connection)
        self.num_buckets = num_buckets
        self.histograms = {}
        self.build_histograms()
    
    def build_histograms(self):
        """Build histograms for all columns in the database"""
        cursor = self.conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
        tables = [row[0] for row in cursor.fetchall()]
        
        for table in tables:
            # Get all columns for the table
            cursor.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}'")
            columns = [row[0] for row in cursor.fetchall()]
            
            for column in columns:
                # Build histogram for the column
                cursor.execute(f"""
                SELECT min({column}), max({column}) FROM {table} WHERE {column} IS NOT NULL
                """)
                min_val, max_val = cursor.fetchone()
                
                if min_val is not None and max_val is not None:
                    if isinstance(min_val, (int, float)):
                        # Create numeric histogram
                        bucket_width = (max_val - min_val) / self.num_buckets
                        histogram = []
                        
                        for i in range(self.num_buckets):
                            bucket_min = min_val + i * bucket_width
                            bucket_max = min_val + (i + 1) * bucket_width
                            cursor.execute(f"""
                            SELECT COUNT(*) FROM {table} 
                            WHERE {column} >= {bucket_min} AND {column} < {bucket_max}
                            """)
                            count = cursor.fetchone()[0]
                            histogram.append((bucket_min, bucket_max, count))
                        
                        self.histograms[(table, column)] = histogram
    
    def estimate_cardinality(self, query):
        """Estimate the cardinality based on histograms"""
        # Parse the query to find table joins and predicates
        tables, join_conditions, predicates = self._parse_query(query)
        
        # Estimate base cardinality for each table
        table_cards = {}
        for table in tables:
            cursor = self.conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            table_cards[table] = cursor.fetchone()[0]
        
        # Apply selectivity from predicates
        for pred in predicates:
            table, column, op, value = pred
            selectivity = self._estimate_selectivity(table, column, op, value)
            table_cards[table] *= selectivity
        
        # Apply join selectivity
        final_card = self._estimate_joins(tables, join_conditions, table_cards)
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
    
    def _estimate_selectivity(self, table, column, op, value):
        """Estimate selectivity using histograms"""
        if (table, column) not in self.histograms:
            return 1.0  # No histogram available
        
        histogram = self.histograms[(table, column)]
        total_rows = sum(bucket[2] for bucket in histogram)
        if total_rows == 0:
            return 1.0
        
        selected_rows = 0
        for bucket_min, bucket_max, count in histogram:
            if op == '=':
                if bucket_min <= value < bucket_max:
                    # Assume uniform distribution within bucket
                    selected_rows += count / (bucket_max - bucket_min)
            elif op == '<':
                if bucket_max <= value:
                    selected_rows += count
                elif bucket_min < value:
                    # Partial bucket
                    selected_rows += count * (value - bucket_min) / (bucket_max - bucket_min)
            elif op == '>':
                if bucket_min >= value:
                    selected_rows += count
                elif bucket_max > value:
                    # Partial bucket
                    selected_rows += count * (bucket_max - value) / (bucket_max - bucket_min)
        
        return selected_rows / total_rows
    
    def _estimate_joins(self, tables, join_conditions, table_cards):
        """Estimate join cardinality using independence assumption"""
        if not tables:
            return 0
        
        # Start with cross product
        total_card = 1
        for table in tables:
            total_card *= table_cards[table]
        
        # Apply join selectivity for each join condition
        for t1, c1, t2, c2 in join_conditions:
            # Simple independence-based join selectivity
            cursor = self.conn.cursor()
            cursor.execute(f"SELECT COUNT(DISTINCT {c1}) FROM {t1}")
            distinct1 = max(1, cursor.fetchone()[0])
            cursor.execute(f"SELECT COUNT(DISTINCT {c2}) FROM {t2}")
            distinct2 = max(1, cursor.fetchone()[0])
            
            # Selectivity = 1 / max(distinct values)
            selectivity = 1 / max(distinct1, distinct2)
            total_card *= selectivity
        
        return total_card

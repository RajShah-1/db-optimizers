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
        # cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        for table in tables:
            # Get all columns for the table
            # cursor.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}'")
            # columns = [row[0] for row in cursor.fetchall()]
            # print("building histogram for table", table)

            cursor.execute(f"PRAGMA table_info({table});")
            columns = [row[1] for row in cursor.fetchall()]
            
            for column in columns:
                # Build histogram for the column
                cursor.execute(f"""
                SELECT min({column}), max({column}) FROM {table} WHERE {column} IS NOT NULL
                """)
                min_val, max_val = cursor.fetchone()
                
                if min_val is not None and max_val is not None:
                    if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
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
        # print(tables, join_conditions, predicates)
        
        # Estimate base cardinality for each table
        table_cards = {}
        for table in tables:
            cursor = self.conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {tables[table]}")
            table_cards[table] = cursor.fetchone()[0]
        
        # Apply selectivity from predicates
        for pred in predicates:
            table, column, op, value = pred
            selectivity = self._estimate_selectivity(table, column, op, value)
            table_cards[table] *= selectivity
        
        # Apply join selectivity
        final_card = self._estimate_joins(tables, join_conditions, table_cards)
        return final_card

    def extract_tables(self, query):
        """Extract tables and their aliases from a SQL query."""
        alias_map = {}

        query = re.split(r'\bWHERE\b', query, flags=re.IGNORECASE)[0]
        # Match tables and aliases from the FROM clause, including comma-separated tables
        from_match = re.search(r'FROM\s+([a-zA-Z0-9_,\s]+)', query, re.IGNORECASE)
        if from_match:
            tables_part = from_match.group(1)
            tables = [t.strip() for t in tables_part.split(',')]
            # print(tables)
            
            for table in tables:
                parts = table.split()
                if len(parts) == 2:  # Table with alias
                    table_name, alias = parts
                else:  # Table without alias
                    table_name, alias = parts[0], parts[0]
                alias_map[alias] = table_name
        
        return alias_map


    def extract_joins(self, query, alias_map):
        """Extract join conditions from a SQL query."""
        join_conditions = []
        join_matches = re.findall(r'([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)\s*=\s*([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)', query)
        
        for alias1, col1, alias2, col2 in join_matches:
            table1 = alias_map.get(alias1, alias1)  # Convert alias to full table name
            table2 = alias_map.get(alias2, alias2)
            join_conditions.append((table1, col1, table2, col2))
        
        return join_conditions

    def extract_predicates(self, query, alias_map):
        """Extract WHERE predicates from a SQL query."""
        predicates = []
        where_matches = re.findall(r'([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)\s*([<>=!]+)\s*([0-9.]+|\'[^\']*\')', query)
        
        for alias, column, operator, value in where_matches:
            table = alias_map.get(alias, alias)  # Convert alias to table name
            if value.startswith("'") and value.endswith("'"):
                value = value.strip("'")  # Handle string values
            else:
                try:
                    value = float(value)  # Convert numbers
                except ValueError:
                    pass  # Keep as string if conversion fails
            predicates.append((alias, column, operator, value))
        
        return predicates

    def _parse_query(self, query):
        """Extract tables, aliases, joins, and predicates from a SQL query."""
        alias_map = self.extract_tables(query)
        join_conditions = self.extract_joins(query, alias_map)
        predicates = self.extract_predicates(query, alias_map)
        return alias_map, join_conditions, predicates


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

from .CardinalityEstimator import CardinalityEstimator
import re

class SampleBasedEstimator(CardinalityEstimator):
    """Cardinality estimator based on sampling"""
    
    def __init__(self, db_connection, sample_size=0.1):
        super().__init__(db_connection)
        self.sample_size = sample_size
        self.samples = {}
        self.build_samples()
    
    def build_samples(self):
        """Build samples for all tables in the database"""
        cursor = self.conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        for table in tables:
            # Get row count
            print("building samples for table", table)
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            total_rows = cursor.fetchone()[0]
            
            # Calculate sample size
            sample_rows = max(100, int(total_rows * self.sample_size))
            
            # Create a sample table
            sample_table = f"{table}_sample"
            cursor.execute(f"DROP TABLE IF EXISTS {sample_table}")
            cursor.execute(f"""
            CREATE TABLE {sample_table} AS
            SELECT * FROM {table}
            ORDER BY RANDOM()
            LIMIT {sample_rows}
            """)
            
            self.samples[table] = sample_table
    
    def estimate_cardinality(self, query):
        """Estimate cardinality by executing query on samples"""
        # Get the tables in the query
        tables = re.findall(r'FROM\s+([a-zA-Z0-9_]+)|JOIN\s+([a-zA-Z0-9_]+)', query)
        tables = [t[0] or t[1] for t in tables]
        
        # Replace tables with sample tables in the query
        sample_query = query
        scaling_factor = 1
        for table in tables:
            if table in self.samples:
                sample_query = sample_query.replace(f" {table} ", f" {self.samples[table]} ")
                
                # Calculate scaling factor
                cursor = self.conn.cursor()
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                total_rows = cursor.fetchone()[0]
                cursor.execute(f"SELECT COUNT(*) FROM {self.samples[table]}")
                sample_rows = cursor.fetchone()[0]
                
                scaling_factor *= (total_rows / sample_rows)
        
        # Execute the query on samples
        cursor = self.conn.cursor()
        cursor.execute(query)
        sample_count = cursor.fetchone()[0]
        
        # Scale the result
        estimated_count = sample_count * scaling_factor
        return estimated_count
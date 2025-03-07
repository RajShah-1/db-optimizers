import os
import re
import time
import json
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
from collections import defaultdict
from abc import ABC, abstractmethod

class CardinalityEstimator(ABC):
    """Abstract base class for cardinality estimators"""
    
    def __init__(self, db_connection):
        self.conn = db_connection
        self.name = self.__class__.__name__
    
    @abstractmethod
    def estimate_cardinality(self, query):
        """Estimate the cardinality of a given query"""
        pass
    
    def explain_query(self, query):
        """Get the query plan using EXPLAIN"""
        cursor = self.conn.cursor()
        explain_query = f"EXPLAIN QUERY PLAN {query}"
        cursor.execute(explain_query)
        return cursor.fetchall()

class PostgresEstimator(CardinalityEstimator):
    """Postgres default cardinality estimator"""
    
    def estimate_cardinality(self, query):
        """Get the estimated cardinality from Postgres EXPLAIN"""
        # Extract the estimated rows from EXPLAIN (format=json)
        cursor = self.conn.cursor()
        explain_query = f"EXPLAIN (FORMAT JSON) {query}"
        cursor.execute(explain_query)
        plan = cursor.fetchone()[0]
        
        # Extract the estimated rows from the plan
        estimated_rows = self._extract_estimated_rows(plan[0]['Plan'])
        return estimated_rows
    
    def _extract_estimated_rows(self, plan):
        """Extract estimated rows from a plan node recursively"""
        estimated_rows = plan.get('Plan Rows', 0)
        
        # If this is a join node, track its estimate
        if 'Plans' in plan:
            for subplan in plan['Plans']:
                sub_estimated = self._extract_estimated_rows(subplan)
                # For joins, we could also track the individual join estimates
                
        return estimated_rows

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
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
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
        count_query = f"SELECT COUNT(*) FROM ({sample_query}) AS sample_result"
        cursor.execute(count_query)
        sample_count = cursor.fetchone()[0]
        
        # Scale the result
        estimated_count = sample_count * scaling_factor
        return estimated_count

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

class QErrorBenchmark:
    """Benchmark for cardinality estimators using q-error metric"""
    
    def __init__(self, db_path, query_file):
        self.db_path = db_path
        self.query_file = query_file
        self.conn = sqlite3.connect(db_path)
        self.estimators = []
        self.queries = self._load_queries()
        self.results = {}
    
    def _load_queries(self):
        """Load JOB queries from file"""
        with open(self.query_file, 'r') as f:
            queries = json.load(f)
        return queries
    
    def add_estimator(self, estimator):
        """Add a cardinality estimator to the benchmark"""
        self.estimators.append(estimator)
    
    def run_benchmark(self):
        """Run the benchmark for all estimators"""
        for estimator in self.estimators:
            self.results[estimator.name] = self._evaluate_estimator(estimator)
    
    def _evaluate_estimator(self, estimator):
        """Evaluate a single estimator on all queries"""
        results = []
        
        for i, query_data in enumerate(self.queries):
            query_sql = query_data['query']
            
            # Get true cardinality by executing the query
            cursor = self.conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM ({query_sql}) AS actual_result")
            true_card = cursor.fetchone()[0]
            
            # Get estimated cardinality
            start_time = time.time()
            est_card = estimator.estimate_cardinality(query_sql)
            end_time = time.time()
            
            # Calculate q-error
            if true_card == 0:
                q_error = float('inf') if est_card > 0 else 1.0
            else:
                q_error = max(est_card / true_card, true_card / est_card)
            
            results.append({
                'query_id': i,
                'true_cardinality': true_card,
                'estimated_cardinality': est_card,
                'q_error': q_error,
                'execution_time': end_time - start_time,
                'num_tables': len(re.findall(r'FROM\s+([a-zA-Z0-9_]+)|JOIN\s+([a-zA-Z0-9_]+)', query_sql))
            })
        
        return results
    
    def analyze_results(self):
        """Analyze the benchmark results"""
        if not self.results:
            print("No results available. Run the benchmark first.")
            return
        
        analysis = {}
        
        for estimator_name, results in self.results.items():
            q_errors = [r['q_error'] for r in results]
            
            analysis[estimator_name] = {
                'median_q_error': np.median(q_errors),
                'mean_q_error': np.mean(q_errors),
                'max_q_error': np.max(q_errors),
                '90th_percentile': np.percentile(q_errors, 90),
                '95th_percentile': np.percentile(q_errors, 95),
                '99th_percentile': np.percentile(q_errors, 99),
                'mean_execution_time': np.mean([r['execution_time'] for r in results])
            }
        
        return analysis
    
    def analyze_by_num_tables(self):
        """Analyze results grouped by number of tables in the query"""
        if not self.results:
            print("No results available. Run the benchmark first.")
            return
        
        analysis = {}
        
        for estimator_name, results in self.results.items():
            by_tables = defaultdict(list)
            
            for r in results:
                by_tables[r['num_tables']].append(r['q_error'])
            
            analysis[estimator_name] = {
                num_tables: {
                    'median_q_error': np.median(q_errors),
                    'mean_q_error': np.mean(q_errors),
                    'count': len(q_errors)
                }
                for num_tables, q_errors in by_tables.items()
            }
        
        return analysis
    
    def plot_q_error_cdf(self, output_file=None):
        """Plot cumulative distribution function of q-errors"""
        plt.figure(figsize=(10, 6))
        
        for estimator_name, results in self.results.items():
            q_errors = sorted([r['q_error'] for r in results])
            y = np.linspace(0, 1, len(q_errors))
            plt.step(q_errors, y, label=estimator_name)
        
        plt.xscale('log')
        plt.xlabel('Q-Error')
        plt.ylabel('Cumulative Probability')
        plt.title('CDF of Q-Errors')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if output_file:
            plt.savefig(output_file)
        else:
            plt.show()
    
    def plot_q_error_by_tables(self, output_file=None):
        """Plot q-errors grouped by number of tables"""
        analysis = self.analyze_by_num_tables()
        
        # Prepare data for plotting
        estimators = list(analysis.keys())
        table_counts = sorted(set(tc for est in analysis.values() for tc in est.keys()))
        
        data = []
        for est in estimators:
            for tc in table_counts:
                if tc in analysis[est]:
                    data.append({
                        'Estimator': est,
                        'Tables': tc,
                        'Median Q-Error': analysis[est][tc]['median_q_error']
                    })
        
        df = pd.DataFrame(data)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Tables', y='Median Q-Error', hue='Estimator', data=df)
        plt.yscale('log')
        plt.title('Median Q-Error by Number of Joined Tables')
        plt.grid(True, alpha=0.3)
        
        if output_file:
            plt.savefig(output_file)
        else:
            plt.show()

def main():
    """Main function to run the benchmark"""
    # Paths to database and queries
    db_path = "./data/imdb.db"
    query_file = "./data/queries.json"
    
    # Create the benchmark
    benchmark = QErrorBenchmark(db_path, query_file)
    
    # Create database connection for estimators
    conn = sqlite3.connect(db_path)
    
    # Add estimators
    benchmark.add_estimator(PostgresEstimator(conn))
    # benchmark.add_estimator(HistogramEstimator(conn))
    benchmark.add_estimator(SampleBasedEstimator(conn))
    # benchmark.add_estimator(HyperLogLogEstimator(conn))
    
    # Run the benchmark
    benchmark.run_benchmark()
    
    # Analyze the results
    analysis = benchmark.analyze_results()
    print("Overall Analysis:")
    for estimator, metrics in analysis.items():
        print(f"\n{estimator}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
    
    # Analyze by number of tables
    analysis_by_tables = benchmark.analyze_by_num_tables()
    print("\nAnalysis by Number of Tables:")
    for estimator, table_metrics in analysis_by_tables.items():
        print(f"\n{estimator}:")
        for num_tables, metrics in sorted(table_metrics.items()):
            print(f"  {num_tables} tables (count: {metrics['count']}):")
            print(f"    Median Q-Error: {metrics['median_q_error']}")
            print(f"    Mean Q-Error: {metrics['mean_q_error']}")
    
    # Plot results
    benchmark.plot_q_error_cdf("q_error_cdf.png")
    benchmark.plot_q_error_by_tables("q_error_by_tables.png")

if __name__ == "__main__":
    main()
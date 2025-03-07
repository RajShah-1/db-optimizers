import re
import json
import time
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

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

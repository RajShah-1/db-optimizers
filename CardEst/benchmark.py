import pickle
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
        self.true_card_cache_file = './data/true_cardinalities.pkl'
        self.true_card_cache = self._load_true_cardinalities()

        self.results = {}

    def _load_true_cardinalities(self):
        """Load cached true cardinalities if available"""
        import os
        import pickle

        if os.path.exists(self.true_card_cache_file):
            with open(self.true_card_cache_file, 'rb') as f:
                print("[Cache] Loaded true cardinalities from file.")
                return pickle.load(f)
        return {}
    
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

        is_cache_updated : bool = False
        
        for i, query_data in enumerate(self.queries):
            query_sql = query_data['query']
            
            # Get true cardinality by executing the query
            # cursor = self.conn.cursor()
            # print(query_sql)
            # cursor.execute(query_sql)
            # true_card = cursor.fetchone()[0]


            query_hash = str(query_sql)
            if query_hash in self.true_card_cache:
                true_card = self.true_card_cache[query_hash]
            else:
                is_cache_updated = True
                cursor = self.conn.cursor()
                cursor.execute(query_sql)
                true_card = cursor.fetchone()[0]
                print(query_sql, true_card)
                self.true_card_cache[query_hash] = true_card

            # # HACK: Materialize all stats for the query for the ACAH estimator
            # if hasattr(estimator, 'prepare_stats_for_query'):
            #     estimator. prepare_stats_for_query(query_sql)

            if hasattr(estimator, 'materialize_all_stats_for_query'):
                estimator.materialize_all_stats_for_query(query_sql)
            
            # Get estimated cardinality
            start_time = time.time()
            est_card = estimator.estimate_cardinality(query_sql)
            end_time = time.time()

            # Apply feedback if estimator supports it
            if hasattr(estimator, "apply_feedback"):
                estimator.apply_feedback(query_sql, true_card, est_card)
            
            # Then apply feedback for future queries
            # This ensures each query is estimated using only statistics from previous queries
            
            
            # Calculate q-error
            if true_card == 0:
                q_error = float('inf') if est_card > 0 else 1.0
            else:
                q_error = max(est_card / true_card, true_card / est_card)
            
            print(f"[Benchmark] Query {i} - query: {query_sql}")
            print(f"[Benchmark] Query {i} - True: {true_card}, Est: {est_card}, Q-Error: {q_error}")
            
            results.append({
                'query_id': i,
                'true_cardinality': true_card,
                'estimated_cardinality': est_card,
                'q_error': q_error,
                'execution_time': end_time - start_time,
                'num_tables': len(re.findall(r'FROM\s+([a-zA-Z0-9_]+)|JOIN\s+([a-zA-Z0-9_]+)', query_sql))
            })

        # Save the true cardinalities to cache if updated
        if is_cache_updated:
            with open(self.true_card_cache_file, 'wb') as f:
                pickle.dump(self.true_card_cache, f)
                print("[Cache] Saved true cardinalities to file.")
        
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

    def _get_true_cardinality(self, query_sql):
        query_hash = str(query_sql)
        if query_hash in self.true_card_cache:
            return self.true_card_cache[query_hash]
        cursor = self.conn.cursor()
        cursor.execute(query_sql)
        true_card = cursor.fetchone()[0]
        self.true_card_cache[query_hash] = true_card
        return true_card

    def run_comparative_feedback_benchmark(self, estimator):
        results = {
            'before': [],
            'after': [],
            'queries': [],
        }

        # Phase 1: Estimation + feedback learning
        for query_data in self.queries:
            query_sql = query_data['query']
            true_card = self._get_true_cardinality(query_sql)

            # if hasattr(estimator, 'materialize_all_stats_for_query'):
                # estimator.materialize_all_stats_for_query(query_sql)

            est_card = estimator.estimate_cardinality(query_sql)
            q_err = max(est_card, true_card) / max(1.0, min(est_card, true_card))
            results['before'].append(q_err)
            results['queries'].append(query_sql)

            if hasattr(estimator, "apply_feedback"):
                estimator.apply_feedback(query_sql, true_card, est_card)

        # Phase 2: Re-estimate using updated model
        for query_sql in results['queries']:
            true_card = self._get_true_cardinality(query_sql)

            if hasattr(estimator, 'materialize_all_stats_for_query'):
                estimator.materialize_all_stats_for_query(query_sql)

            est_card = estimator.estimate_cardinality(query_sql)
            q_err = max(est_card, true_card) / max(1.0, min(est_card, true_card))
            results['after'].append(q_err)

        return results

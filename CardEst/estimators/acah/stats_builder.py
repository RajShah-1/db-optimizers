# stats_builder.py
"""Offline statistics generation: base histograms and correlation identification."""

import numpy as np
import math
from collections import defaultdict

from .structures import Bucket
from . import config

class StatsBuilder:
    """Builds base equi-depth histograms and identifies correlated pairs."""

    def __init__(self, db_conn):
        """Initializes the StatsBuilder with a database connection."""
        self.conn = db_conn
        self.table_column_types = defaultdict(dict)

    def build_all(self):
        """Builds and returns all necessary offline statistics."""
        print("Building initial ACAH V3 stats...")
        self._discover_schema_types()
        histograms = self._build_equidepth_histograms()
        correlated_pairs = self._identify_correlations(histograms)
        print("Initial ACAH V3 stats building complete.")
        return histograms, self.table_column_types, correlated_pairs

    def _discover_schema_types(self):
        """Identifies and stores column types (numeric/text) for all tables."""
        print("Discovering schema types...")
        cursor = self.conn.cursor()
        try:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall() if not row[0].startswith('sqlite_')]
            for table in tables:
                cursor.execute(f"PRAGMA table_info('{table}')") # Use quotes for safety
                for row in cursor.fetchall():
                    col_name, col_type_raw = row[1], row[2].upper()
                    col_type = 'numeric' if any(t in col_type_raw for t in ['INT', 'REAL', 'FLOAT', 'DOUBLE', 'NUMERIC', 'DECIMAL']) else 'text'
                    self.table_column_types[table][col_name] = col_type
        except Exception as e:
            print(f"Error discovering schema: {e}")
            raise

    def _build_equidepth_histograms(self):
        """Builds base 1D equi-depth histograms for numeric columns."""
        print("Building equi-depth histograms...")
        histograms = {}
        cursor = self.conn.cursor()
        for table, columns in self.table_column_types.items():
            for column, col_type in columns.items():
                if col_type != 'numeric': continue
                try:
                    # Use quotes for safety
                    cursor.execute(f'SELECT "{column}" FROM "{table}" WHERE "{column}" IS NOT NULL ORDER BY "{column}"')
                    sorted_values = [row[0] for row in cursor.fetchall()]
                    num_rows = len(sorted_values)
                    
                    if num_rows < config.DEFAULT_NUM_BUCKETS: continue # Skip if too few values

                    target_rows_per_bucket = math.ceil(num_rows / config.DEFAULT_NUM_BUCKETS)
                    buckets = []
                    current_index = 0

                    while current_index < num_rows:
                        start_index = current_index
                        end_index = min(start_index + target_rows_per_bucket, num_rows)
                        
                        # Handle ties at bucket boundary
                        if end_index > start_index and end_index < num_rows:
                             boundary_value = sorted_values[end_index - 1]
                             while end_index < num_rows and sorted_values[end_index] == boundary_value:
                                 end_index += 1
                        
                        bucket_values = sorted_values[start_index:end_index]
                        if not bucket_values: break 

                        min_val, max_val = bucket_values[0], bucket_values[-1]
                        count = len(bucket_values)
                        distinct_count = len(np.unique(bucket_values)) if count > 0 else 0

                        buckets.append(Bucket(min_val, max_val, count, distinct_count))
                        current_index = end_index
                        
                    if buckets:
                         # Ensure full range coverage
                         buckets[0].min_val = sorted_values[0]
                         buckets[-1].max_val = sorted_values[-1]
                         histograms[(table, column)] = buckets
                except Exception as e:
                     print(f"  Error building equi-depth hist for {table}.{column}: {e}")
        return histograms

    def _calculate_correlation(self, table, col1, col2):
        """Calculates Pearson correlation using numpy."""
        cursor = self.conn.cursor()
        try:
            # Use quotes for safety
            cursor.execute(f'SELECT "{col1}", "{col2}" FROM "{table}" WHERE "{col1}" IS NOT NULL AND "{col2}" IS NOT NULL')
            data = cursor.fetchall()
            if len(data) < 20: return 0.0 # Need minimum data points

            col1_data = np.array([row[0] for row in data], dtype=float)
            col2_data = np.array([row[1] for row in data], dtype=float)
            
            # Check for constant columns robustly
            std1 = np.std(col1_data); std2 = np.std(col2_data)
            if math.isclose(std1, 0) or math.isclose(std2, 0): return 0.0

            corr_matrix = np.corrcoef(col1_data, col2_data)
            corr = corr_matrix[0, 1]
            return abs(corr) if not np.isnan(corr) else 0.0
        except Exception as e:
            # Keep this warning for correlation issues
            print(f"  Correlation calc warning for {table}.({col1}, {col2}): {e}") 
            return 0.0

    def _identify_correlations(self, histograms):
        """Identifies correlated numeric pairs based on the configured method."""
        print("Identifying correlations...")
        correlated_pairs = {}
        for table, columns in self.table_column_types.items():
            numeric_cols = [col for col, type in columns.items() if type == 'numeric' and (table, col) in histograms]
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    score = 0.0
                    if config.CORRELATION_METHOD == 'pearson':
                         score = self._calculate_correlation(table, col1, col2)
                    # Add other methods like mutual information here if needed...
                         
                    if score >= config.CORRELATION_THRESHOLD:
                        # Store with consistent order (col1 < col2 alphabetically)
                        key = tuple(sorted((col1, col2)))
                        correlated_pairs[(table, key[0], key[1])] = score
                        # print(f"  Found correlated pair: {table}.({key[0]}, {key[1]}), score: {score:.3f}") # Reduce noise
        return correlated_pairs
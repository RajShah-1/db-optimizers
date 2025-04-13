import math
import numpy as np
from collections import defaultdict, OrderedDict

from . import config
from .structures import Bucket

class FeedbackHandlerML:
    """Fail-fast feedback: eagerly applies all useful refinements before estimation."""

    def __init__(self, db_conn, histograms_ref, column_types_ref, correlated_pairs_ref):
        self.conn = db_conn
        self.histograms = histograms_ref
        self.column_types = column_types_ref
        self.correlated_pairs = correlated_pairs_ref

        self.cond_summary_cache = OrderedDict()
        print("FeedbackHandler initialized.")

    def _find_bucket_index(self, hist, value):
        try:
            num_val = float(value)
        except:
            return None
        for i, bucket in enumerate(hist):
            if bucket.contains(num_val, i == len(hist) - 1):
                return i
        return None

    def materialize_all_relevant_summaries(self, query, estimation_details):
        """
        Eagerly materializes all relevant stats:
        - Splits all buckets on predicate columns
        - Builds all conditional summaries between joined and filtered columns
        """
        parsed = estimation_details.get('parsed', {})
        preds = parsed.get('preds', [])
        joins = parsed.get('joins', [])
        tables = parsed.get('tables', {})

        pred_cols_by_table = defaultdict(set)
        for t, c, _, _ in preds:
            pred_cols_by_table[t].add(c)

        # 1. Split buckets on all predicate columns (no size limit)
        for table, col, op, val in preds:
            hist_key = (table, col)
            hist = self.histograms.get(hist_key)
            if not hist:
                continue
            idx = self._find_bucket_index(hist, val)
            if idx is None or idx >= len(hist):
                continue
            bucket = hist[idx]
            new_buckets = self._split_single_bucket_internal(table, col, bucket, val)
            if new_buckets:
                self.histograms[hist_key] = hist[:idx] + new_buckets + hist[idx+1:]
                self._invalidate_cond_cache_for_col(table, col)

        # 2. Build conditional summaries between all interesting column pairs
        for t1, c1, t2, c2 in joins:
            pred_cols_by_table[t1].add(c1)
            pred_cols_by_table[t2].add(c2)

        for table, cols in pred_cols_by_table.items():
            cols_list = list(cols)
            for i in range(len(cols_list)):
                for j in range(i + 1, len(cols_list)):
                    a, b = cols_list[i], cols_list[j]
                    self._build_conditional_summaries_for_pair(table, a, b)
                    self._build_conditional_summaries_for_pair(table, b, a)

    def _build_conditional_summaries_for_pair(self, table, col_c, col_d):
        hist_key_c = (table, col_c)
        base_hist = self.histograms.get(hist_key_c)
        if not base_hist:
            return

        for b_idx, bucket in enumerate(base_hist):
            mini_hist = self._build_conditional_mini_hist_internal(table, col_c, bucket, col_d)
            if mini_hist:
                key = (table, col_c, b_idx, col_d)
                self.cond_summary_cache[key] = mini_hist

    def get_cond_summary(self, table, col_c, bucket_idx_c, col_d):
        cache_key = (table, col_c, bucket_idx_c, col_d)
        summary = self.cond_summary_cache.get(cache_key)
        if summary is not None:
            self.cond_summary_cache.move_to_end(cache_key)
        return summary

    def _invalidate_cond_cache_for_col(self, table, col_c):
        keys_to_remove = [key for key in self.cond_summary_cache if key[0] == table and key[1] == col_c]
        for key in keys_to_remove:
            del self.cond_summary_cache[key]

    # --- Internal utilities below ---

    def _build_conditional_mini_hist_internal(self, table, col_c, bucket_c, col_d):
        cursor = self.conn.cursor()
        query = f'SELECT "{col_d}" FROM "{table}" WHERE "{col_c}" >= ? AND "{col_c}" <= ? AND "{col_d}" IS NOT NULL ORDER BY "{col_d}"'
        try:
            cursor.execute(query, (bucket_c.min_val, bucket_c.max_val))
            values = [row[0] for row in cursor.fetchall()]
            if len(values) < config.COND_HIST_BUCKETS:
                return None

            mini_hist = []
            per_bucket = math.ceil(len(values) / config.COND_HIST_BUCKETS)
            idx = 0
            while idx < len(values):
                start = idx
                end = min(start + per_bucket, len(values))

                if end < len(values):
                    boundary = values[end - 1]
                    while end < len(values) and values[end] == boundary:
                        end += 1

                subset = values[start:end]
                if not subset:
                    break
                b_min, b_max = subset[0], subset[-1]
                count = len(subset)
                ndv = len(np.unique(subset))
                mini_hist.append(Bucket(b_min, b_max, count, ndv))
                idx = end

            if mini_hist:
                mini_hist[0].min_val = values[0]
                mini_hist[-1].max_val = values[-1]
                return mini_hist
        except Exception as e:
            print(f"    Error building conditional histogram: {e}")
        return None

    def _split_single_bucket_internal(self, table, col, bucket, val):
        cursor = self.conn.cursor()
        try:
            cursor.execute(f'SELECT "{col}" FROM "{table}" WHERE "{col}" >= ? AND "{col}" <= ? ORDER BY "{col}"',
                           (bucket.min_val, bucket.max_val))
            values = [row[0] for row in cursor.fetchall()]
            if len(values) < 2:
                return None

            mid_idx = len(values) // 2
            split_val = values[mid_idx]

            cursor.execute(f'SELECT COUNT(*), COUNT(DISTINCT "{col}") FROM "{table}" WHERE "{col}" >= ? AND "{col}" <= ?',
                           (bucket.min_val, split_val))
            count1, ndv1 = cursor.fetchone()

            cursor.execute(f'SELECT MIN("{col}") FROM "{table}" WHERE "{col}" > ? AND "{col}" <= ?',
                           (split_val, bucket.max_val))
            new_min = cursor.fetchone()[0]
            if new_min is None:
                return None

            cursor.execute(f'SELECT COUNT(*), COUNT(DISTINCT "{col}") FROM "{table}" WHERE "{col}" >= ? AND "{col}" <= ?',
                           (new_min, bucket.max_val))
            count2, ndv2 = cursor.fetchone()

            b1 = Bucket(bucket.min_val, split_val, count1, ndv1)
            b2 = Bucket(new_min, bucket.max_val, count2, ndv2)
            return [b1, b2]
        except Exception as e:
            print(f"    Error splitting bucket: {e}")
            return None

import math
from collections import OrderedDict

import numpy as np

from estimators.acah import config
from estimators.acah.structures import Bucket

class CondSummaryCatalog:
    def __init__(self, conn, cache_size=10000):
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.conn = conn

    def get(self, table, col_c, bucket_idx_c, col_d):
        key = (table, col_c, bucket_idx_c, col_d)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def set(self, table, col_c, bucket_idx_c, col_d, mini_hist):
        key = (table, col_c, bucket_idx_c, col_d)
        self.cache[key] = mini_hist
        self.cache.move_to_end(key)
        self._evict_if_needed()

    def invalidate_by_column(self, table, col_c):
        keys = [k for k in self.cache if k[0] == table and k[1] == col_c]
        for k in keys:
            del self.cache[k]

    def build_all_summaries_for(self, table, col_c, col_d, hist_catalog):
        hist = hist_catalog.get(table, col_c)
        if not hist:
            return
        for bucket_idx, bucket in enumerate(hist):
            summary = self._build_cond_summary(table, col_c, bucket, col_d)
            if summary:
                self.set(table, col_c, bucket_idx, col_d, summary)

    def _evict_if_needed(self):
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

    def _build_cond_summary(self, table, col_c, bucket, col_d):
        """Builds a conditional mini-histogram on col_d for a given bucket of col_c."""
        cursor = self.conn.cursor()
        try:
            cursor.execute(f'''
                SELECT "{col_d}" FROM "{table}" 
                WHERE "{col_c}" >= ? AND "{col_c}" <= ? AND "{col_d}" IS NOT NULL 
                ORDER BY "{col_d}"
            ''', (bucket.min_val, bucket.max_val))
            values = [r[0] for r in cursor.fetchall()]
            if len(values) < config.COND_HIST_BUCKETS:
                return None

            mini_hist = []
            rows_per_bucket = math.ceil(len(values) / config.COND_HIST_BUCKETS)
            idx = 0
            while idx < len(values):
                end = min(idx + rows_per_bucket, len(values))
                boundary = values[end - 1]
                while end < len(values) and values[end] == boundary:
                    end += 1
                bucket_values = values[idx:end]
                mh_min, mh_max = bucket_values[0], bucket_values[-1]
                mini_hist.append(Bucket(mh_min, mh_max, len(bucket_values), len(np.unique(bucket_values))))
                idx = end

            if mini_hist:
                mini_hist[0].min_val = values[0]
                mini_hist[-1].max_val = values[-1]
                return mini_hist
            else:
                return None
        except Exception as e:
            print(f"Error building conditional summary {table}.{col_c}->{col_d}: {e}")
            return None

    def invalidate(self, table, column):
        """
        Invalidates all conditional summaries that are conditioned ON col_c.
        This is usually called after the histogram for col_c has changed (e.g., split).
        """
        keys_to_remove = [key for key in self.cache if key[0] == table and key[1] == column]
        for key in keys_to_remove:
            del self.cache[key]

    def materialize_summary(self, table, col_a, col_b, hist_catalog):
        """
        Public API to build and store conditional summary col_a → col_b.
        To be used when applying feedback.
        """
        hist = hist_catalog.get(table, col_a)
        if not hist:
            return
        for bucket_idx, bucket in enumerate(hist):
            mini_hist = self._build_cond_summary(table, col_a, bucket, col_b)
            if mini_hist:
                self.set(table, col_a, bucket_idx, col_b, mini_hist)


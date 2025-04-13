from estimators.acah import config
from estimators.acah.structures import Bucket

class HistogramCatalog:
    def __init__(self, histograms, conn):
        self.conn = conn
        self.histograms = histograms

    def get(self, table, column):
        return self.histograms.get((table, column))

    def set(self, table, column, buckets):
        self.histograms[(table, column)] = buckets

    def get_all(self):
        return self.histograms

    def split_bucket(self, table, column, bucket_idx, new_buckets):
        hist = self.histograms.get((table, column), [])
        if not hist or bucket_idx >= len(hist): return False
        self.histograms[(table, column)] = hist[:bucket_idx] + new_buckets + hist[bucket_idx + 1:]
        return True

    def split_bucket_on_value(self, table, column, value):
        hist = self.get(table, column)
        if not hist:
            return
        idx = self._find_bucket_index(hist, value)
        if idx is None:
            return
        bucket = hist[idx]
        new_buckets = self._split_bucket_logic(table, column, bucket, value)
        if new_buckets:
            self.replace(table, column, idx, new_buckets)

    def _find_bucket_index(self, hist, value):
        """Finds the index of the bucket containing the value."""
        try:
            num_value = float(value)
        except:
            return None

        for i, b in enumerate(hist):
            if b.contains(num_value, is_last_bucket=(i == len(hist) - 1)):
                return i
        return None

    def replace(self, table, column, index, new_buckets):
        """
        Replaces the bucket at `index` in the histogram for (table, column) with `new_buckets`.
        """
        key = (table, column)
        if key in self.histograms:
            original = self.histograms[key]
            if 0 <= index < len(original):
                self.histograms[key] = original[:index] + new_buckets + original[index + 1:]

    def _split_bucket_logic(self, table, column, bucket, value):
        """Split a bucket into two around the value (median if needed)."""
        cursor = self.conn.cursor()
        try:
            cursor.execute(f'''
                SELECT "{column}" FROM "{table}" 
                WHERE "{column}" >= ? AND "{column}" <= ? 
                ORDER BY "{column}"
            ''', (bucket.min_val, bucket.max_val))
            values = [r[0] for r in cursor.fetchall()]
            if len(values) < config.MIN_BUCKET_SIZE * 2:
                return None

            split_idx = len(values) // 2
            split_val = values[split_idx]

            # First half
            cursor.execute(f'''
                SELECT COUNT(*), COUNT(DISTINCT "{column}") FROM "{table}" 
                WHERE "{column}" >= ? AND "{column}" <= ?
            ''', (bucket.min_val, split_val))
            cnt1, ndv1 = cursor.fetchone()

            # Second half
            cursor.execute(f'''
                SELECT MIN("{column}") FROM "{table}" 
                WHERE "{column}" > ? AND "{column}" <= ?
            ''', (split_val, bucket.max_val))
            b2_min = cursor.fetchone()[0]
            if b2_min is None:
                return None

            cursor.execute(f'''
                SELECT COUNT(*), COUNT(DISTINCT "{column}") FROM "{table}" 
                WHERE "{column}" >= ? AND "{column}" <= ?
            ''', (b2_min, bucket.max_val))
            cnt2, ndv2 = cursor.fetchone()

            if cnt1 < config.MIN_BUCKET_SIZE or cnt2 < config.MIN_BUCKET_SIZE:
                return None

            return [
                Bucket(bucket.min_val, split_val, cnt1, ndv1),
                Bucket(b2_min, bucket.max_val, cnt2, ndv2)
            ]
        except Exception as e:
            print(f"Error splitting bucket {table}.{column}: {e}")
            return None

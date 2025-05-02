import os
import pickle
from collections import defaultdict

from estimators.acah.config import SCHEMA_INSPECTOR_CACHE

class SchemaInspector:
    """
    Extracts schema-level stats like:
    - Primary keys
    - Column NDVs
    - "Keyness" score = NDV / total count
    """

    def __init__(self, conn):
        self.conn = conn
        self.ndv = {}               # (table, col) -> ndv
        self.total_rows = {}        # table -> total count
        self.keyness = {}           # (table, col) -> score
        self.pk_cols = defaultdict(set)  # table -> set(col)

        if os.path.exists(SCHEMA_INSPECTOR_CACHE):
            self._load_cache()
        else:
            self._inspect_schema()
            self._save_cache()

    def _inspect_schema(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cursor.fetchall() if not r[0].startswith("sqlite_")]

        for table in tables:
            try:
                cursor.execute(f'SELECT COUNT(*) FROM "{table}"')
                total = cursor.fetchone()[0]
                self.total_rows[table] = total
            except:
                continue

            try:
                cursor.execute(f'PRAGMA table_info("{table}")')
                cols_info = cursor.fetchall()
                for col_info in cols_info:
                    col = col_info[1]
                    is_pk = col_info[5] == 1
                    if is_pk:
                        self.pk_cols[table].add(col)

                    try:
                        cursor.execute(f'SELECT COUNT(DISTINCT "{col}") FROM "{table}"')
                        ndv = cursor.fetchone()[0]
                        self.ndv[(table, col)] = ndv
                        self.keyness[(table, col)] = ndv / max(1, total)
                    except:
                        continue
            except:
                continue

    def _save_cache(self):
        data = {
            'ndv': self.ndv,
            'total_rows': self.total_rows,
            'keyness': self.keyness,
            'pk_cols': dict(self.pk_cols)
        }
        os.makedirs(os.path.dirname(SCHEMA_INSPECTOR_CACHE), exist_ok=True)
        with open(SCHEMA_INSPECTOR_CACHE, "wb") as f:
            pickle.dump(data, f)

    def _load_cache(self):
        with open(SCHEMA_INSPECTOR_CACHE, "rb") as f:
            data = pickle.load(f)
            self.ndv = data.get("ndv", {})
            self.total_rows = data.get("total_rows", {})
            self.keyness = data.get("keyness", {})
            self.pk_cols = defaultdict(set, data.get("pk_cols", {}))

    def get_ndv(self, table, col):
        return self.ndv.get((table, col), 1)

    def get_keyness(self, table, col):
        return self.keyness.get((table, col), 1.0)

    def is_primary_key(self, table, col):
        return col in self.pk_cols.get(table, set())

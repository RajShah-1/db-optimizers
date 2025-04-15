import os
import pickle

from estimators.acah.catalog.cond_summary_catalog import CondSummaryCatalog
from estimators.acah.catalog.hist_catalog import HistogramCatalog
from estimators.acah import config

class StatisticsCatalog:
    _instance = None

    def __init__(self):
        if StatisticsCatalog._instance is not None:
            raise Exception("Use StatisticsCatalog.get() instead of instantiating directly")
        self.histogram_catalog : HistogramCatalog = None
        self.cond_summary_catalog : CondSummaryCatalog = None
        self.column_types = {}
        self.correlated_pairs = {}

    @staticmethod
    def get():
        if StatisticsCatalog._instance is None:
            StatisticsCatalog._instance = StatisticsCatalog()
        return StatisticsCatalog._instance

    def initialize_from_builder(self, stats_builder):
        histograms, col_types, corr_pairs = stats_builder.build_all()
        self.histogram_catalog = HistogramCatalog(histograms, stats_builder.conn)
        self.cond_summary_catalog = CondSummaryCatalog(stats_builder.conn, config.COND_SUMMARY_CACHE_SIZE)
        self.column_types = col_types
        self.correlated_pairs = corr_pairs

    def initialize_from_file(self, conn, path=None):
        path = path or config.STATS_FILE
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.histogram_catalog = HistogramCatalog(data["histograms"], conn)
        self.cond_summary_catalog = CondSummaryCatalog(conn, config.COND_SUMMARY_CACHE_SIZE)
        self.column_types = data["column_types"]
        self.correlated_pairs = data["correlated_pairs"]

    def save_to_file(self, path=None):
        path = path or config.STATS_FILE
        dir_path = os.path.dirname(path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(path, "wb") as f:
            data = {
                "histograms": self.histogram_catalog.get_all(),
                "column_types": self.column_types,
                "correlated_pairs": self.correlated_pairs
            }
            pickle.dump(data, f)

    def get_histogram_catalog(self):
        return self.histogram_catalog

    def get_cond_summary_catalog(self):
        return self.cond_summary_catalog

    def get_column_types(self):
        return self.column_types

    def get_correlated_pairs(self):
        return self.correlated_pairs
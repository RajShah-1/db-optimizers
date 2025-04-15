# main_estimator.py
"""Main facade class for ACAH Estimator V3."""

from ..CardinalityEstimator import CardinalityEstimator
from .stats_builder import StatsBuilder
from .estimator.query_estimator import QueryEstimator
from .feedback_handler import FeedbackHandlerML
from estimators.acah.catalog.stats_catalog import StatisticsCatalog
from . import config
import os


class ACAHv3Estimator(CardinalityEstimator):
    """Facade class managing state, estimation, and feedback for ACAH V3."""

    def __init__(self, conn):
        super().__init__(conn)
        self.stats_catalog = StatisticsCatalog.get()
        self._initialize_catalog(conn)

        self.query_estimator = QueryEstimator(self.conn)
        self.feedback_handler = FeedbackHandlerML(
            self.conn,
            self.stats_catalog
        )

    def _initialize_catalog(self, conn):
        if os.path.exists(config.STATS_FILE):
            try:
                self.stats_catalog.initialize_from_file(conn)
                print(f"Loaded stats from {config.STATS_FILE}")
                return
            except Exception as e:
                print(f"Error loading stats: {e}")
        builder = StatsBuilder(self.conn)
        self.stats_catalog.initialize_from_builder(builder)
        self.stats_catalog.save_to_file()
        print("Stats initialized from scratch.")

    def estimate_cardinality(self, query):
        estimate, details = self.query_estimator.estimate(query, self.stats_catalog.cond_summary_catalog.get)
        return estimate

    def materialize_all_stats_for_query(self, query):
        _, details = self.query_estimator.estimate(query, self.stats_catalog.cond_summary_catalog.get)
        self.feedback_handler.materialize_all_relevant_summaries(query, details)

    def apply_pending_refinements(self, save_stats_after=True):
        return 0  # ML refinements not used in fail-fast mode

    def save_stats(self):
        self.stats_catalog.save_to_file()

from abc import ABC, abstractmethod


class CardinalityEstimator(ABC): 
    """Abstract base class for cardinality estimators"""

    def __init__(self, db_connection):
        if db_connection is None:
            raise ValueError("Database connection cannot be None")
        self.conn = db_connection
        # Extract the concrete class name automatically
        self.name = self.__class__.__name__
        print(f"Initialized Estimator: {self.name}")

    @abstractmethod
    def estimate_cardinality(self, query):
        """Estimates the cardinality for the given SQL query."""
        pass

    def explain_query(self, query):
        """Uses the database's EXPLAIN QUERY PLAN feature."""
        if not self.conn:
            print("Error: No database connection available for EXPLAIN.")
            return None
        cursor = self.conn.cursor()
        explain_query = f"EXPLAIN QUERY PLAN {query}"
        try:
            cursor.execute(explain_query)
            return cursor.fetchall()
        except Exception as e:
            print(f"Error executing EXPLAIN QUERY PLAN: {e}")
            return None

    def learn_from_error(self, query, actual_card):
        """Allows adaptive estimators to learn from observed errors."""
        pass

    def apply_pending_refinements(self, save_stats_after=True):
        """Triggers the application of any queued learning/refinement steps."""
        pass

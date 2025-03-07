from abc import ABC, abstractmethod

class CardinalityEstimator(ABC):
    """Abstract base class for cardinality estimators"""
    
    def __init__(self, db_connection):
        self.conn = db_connection
        self.name = self.__class__.__name__
    
    @abstractmethod
    def estimate_cardinality(self, query):
        pass
    
    def explain_query(self, query):
        cursor = self.conn.cursor()
        explain_query = f"EXPLAIN QUERY PLAN {query}"
        cursor.execute(explain_query)
        return cursor.fetchall()

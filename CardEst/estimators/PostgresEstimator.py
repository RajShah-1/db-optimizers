from .CardinalityEstimator import CardinalityEstimator

class PostgresEstimator(CardinalityEstimator):
    """Postgres default cardinality estimator"""
    
    def estimate_cardinality(self, query):
        """Get the estimated cardinality from Postgres EXPLAIN"""
        # Extract the estimated rows from EXPLAIN (format=json)
        cursor = self.conn.cursor()
        explain_query = f"EXPLAIN (FORMAT JSON) {query}"
        cursor.execute(explain_query)
        plan = cursor.fetchone()[0]
        
        # Extract the estimated rows from the plan
        estimated_rows = self._extract_estimated_rows(plan[0]['Plan'])
        return estimated_rows
    
    def _extract_estimated_rows(self, plan):
        """Extract estimated rows from a plan node recursively"""
        estimated_rows = plan.get('Plan Rows', 0)
        
        # If this is a join node, track its estimate
        if 'Plans' in plan:
            for subplan in plan['Plans']:
                sub_estimated = self._extract_estimated_rows(subplan)
                # For joins, we could also track the individual join estimates
                
        return estimated_rows

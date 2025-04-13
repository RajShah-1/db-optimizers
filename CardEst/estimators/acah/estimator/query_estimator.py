"""ACAH Estimator V3 using a modular pipeline for cardinality estimation."""

from .pipeline import QueryEstimatorPipeline, QueryEstimatorContext
from .. import query_parser

class QueryEstimator:
    def __init__(self, db_conn, histograms, column_types, correlated_pairs):
        self.pipeline = QueryEstimatorPipeline()
        self.context = QueryEstimatorContext(
            conn=db_conn,
            histograms=histograms,
            column_types=column_types,
            correlated_pairs=correlated_pairs
        )

    def estimate(self, query, get_cond_summary_func):
        self.context.reset(query, get_cond_summary_func)
        try:
            tables_map, join_conditions, predicates = query_parser.parse_query(query, self.context.column_types)
            self.context.set_parsed_query(tables_map, join_conditions, predicates)
            self.pipeline.run(self.context)
            return max(1.0, self.context.final_cardinality), self.context.details.copy()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.context.details['error'] = str(e)
            return 1.0, self.context.details.copy()

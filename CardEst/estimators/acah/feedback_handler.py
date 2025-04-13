# feedback_handler.py
"""Handles refinement tasks in ACAH V3 (fail-fast mode, no ML)."""

from collections import defaultdict

class FeedbackHandlerML:
    """Eager refinement manager for ACAH V3 (fail-fast, non-ML mode)."""

    def __init__(self, conn, stats_catalog):
        self.conn = conn
        self.stats_catalog = stats_catalog
        self.hist_catalog = stats_catalog.histogram_catalog
        self.cond_catalog = stats_catalog.cond_summary_catalog
        self.column_types = stats_catalog.column_types
        self.correlated_pairs = stats_catalog.correlated_pairs

    def materialize_all_relevant_summaries(self, query, estimation_details):
        """Eagerly materialize all refinements: splits and conditional summaries."""
        parsed = estimation_details.get("parsed", {})
        preds = parsed.get("preds", [])
        joins = parsed.get("joins", [])

        pred_cols_by_table = defaultdict(set)

        # 1. Split on all predicate columns
        for t, c, _, val in preds:
            pred_cols_by_table[t].add(c)
            self.hist_catalog.split_bucket_on_value(table=t, column=c, value=val)
            self.cond_catalog.invalidate(table=t, column=c)

        # 2. Collect join keys
        for t1, c1, t2, c2 in joins:
            pred_cols_by_table[t1].add(c1)
            pred_cols_by_table[t2].add(c2)

        # 3. Build summaries for all interesting (col_a, col_b) pairs
        for table, cols in pred_cols_by_table.items():
            cols_list = list(cols)
            for col_a in cols_list:
                for col_b in cols_list:
                    if col_a == col_b:
                        continue
                    self.cond_catalog.build_all_summaries_for(table, col_a, col_b, self.hist_catalog)

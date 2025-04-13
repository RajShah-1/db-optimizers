# pipeline.py
"""Modular pipeline components for ACAH V3 Estimator (updated with conditional summaries)."""

from collections import defaultdict
import math

class QueryEstimatorContext:
    def __init__(self, conn, histograms, column_types, correlated_pairs):
        self.conn = conn
        self.histograms = histograms
        self.column_types = column_types
        self.correlated_pairs = correlated_pairs
        self.reset()

    def reset(self, query=None, get_cond_summary_func=None):
        self.query = query
        self.get_cond_summary_func = get_cond_summary_func
        self.details = {}
        self.tables_map = {}
        self.predicates = []
        self.joins = []
        self.table_cardinalities = {}
        self.effective_table_cards = {}
        self.final_cardinality = 1.0

    def set_parsed_query(self, tables_map, joins, predicates):
        self.tables_map = tables_map
        self.joins = joins
        self.predicates = predicates
        self.details['parsed'] = {
            'tables': tables_map,
            'joins': joins,
            'preds': predicates
        }


class QueryEstimatorPipeline:
    def __init__(self):
        self.nodes = [
            InitialTableCardinalities(),
            PredicateSelectivity(),
            JoinOverlapEstimator()
        ]

    def run(self, context: QueryEstimatorContext):
        for node in self.nodes:
            node.run(context)


class InitialTableCardinalities:
    def run(self, context: QueryEstimatorContext):
        cursor = context.conn.cursor()
        cards = {}
        for alias, table in context.tables_map.items():
            try:
                cursor.execute(f'SELECT COUNT(*) FROM "{table}"')
                cards[table] = cursor.fetchone()[0]
            except:
                cards[table] = 1
        context.table_cardinalities = cards
        context.details['initial_cards'] = cards


class PredicateSelectivity:
    def run(self, context: QueryEstimatorContext):
        preds_by_table = defaultdict(list)
        for idx, pred in enumerate(context.predicates):
            preds_by_table[pred[0]].append({'original_index': idx, 'pred': pred})

        eff_cards = {}
        table_sels = {}
        pred_sels = {}

        for table, card in context.table_cardinalities.items():
            if card <= 0:
                eff_cards[table] = 0
                table_sels[table] = 0.0
                continue

            sel, pred_sel_detail = self._estimate_table_sel(context, table, preds_by_table[table])
            eff_cards[table] = max(1, card * sel)
            table_sels[table] = sel
            pred_sels.update(pred_sel_detail)

        context.effective_table_cards = eff_cards
        context.details['effective_cards'] = eff_cards
        context.details['table_selectivities'] = table_sels
        context.details['predicate_selectivities'] = pred_sels

    def _estimate_table_sel(self, context, table, preds):
        """
        Estimates table selectivity based on predicates, considering correlated columns.

        Args:
            context: Context object containing correlated pairs and histograms.
            table: Table identifier.
            preds: List of predicate dictionaries, each with 'pred' and 'original_index'.

        Returns:
            Tuple of (estimated selectivity, dictionary of predicate selectivities).
        """
        sel = 1.0
        pred_sels = {}
        correlated_pairs_processed = set()  # Track processed correlated pairs

        # First, process individual predicate selectivities
        for pred_data in preds:
            pred = pred_data['pred']
            idx = pred_data['original_index']

            # Check if this predicate is part of a correlated pair
            correlated = False
            for col1, col2, score in context.correlated_pairs.get(table, []):  # Use get to avoid KeyError if table not in correlated_pairs
                if pred[1] in (col1, col2):
                    correlated = True
                    break

            if not correlated:
                pred_sel = self._estimate_ind_sel(context, table, pred)
                pred_sels[idx] = ('ind', pred_sel)
                sel *= pred_sel

        # Then, process correlated predicate selectivities
        for pred_data1 in preds:
            pred1 = pred_data1['pred']
            idx1 = pred_data1['original_index']
            for pred_data2 in preds:
                pred2 = pred_data2['pred']
                idx2 = pred_data2['original_index']

                key = tuple(sorted((pred1[1], pred2[1])))
                full_key = (table, key[0], key[1])

                if full_key in context.correlated_pairs and key not in correlated_pairs_processed:
                    score = context.correlated_pairs[full_key]
                    correlated_pairs_processed.add(key)  # Mark pair as processed

                    # Calculate conditional selectivity
                    if pred1[1] == key[0]:
                        other_col = pred2[1]
                        main_pred = pred1
                    else:
                        other_col = pred1[1]
                        main_pred = pred2

                    pred_sel = self._estimate_with_cond_summary(context, table, main_pred, other_col)

                    if pred_sel is not None:
                        pred_sels[idx1] = ('cond_summary', pred_sel)
                        sel *= pred_sel

        return sel, pred_sels

    def _estimate_with_cond_summary(self, context, table, pred, other_col):
        """Try to estimate selectivity using conditional summaries."""
        get_cond_summary = context.get_cond_summary_func
        if not get_cond_summary:
            return None

        col_c = pred[1]
        op = pred[2]
        val = pred[3]

        # Try each other column to see if there is a cached summary
        if other_col == col_c:
            return None
        hist_c = context.histograms.get((table, other_col))
        if not hist_c:
            return None
        bucket_idx = self._find_bucket_index(hist_c, val)
        if bucket_idx is None:
            return None

        cond_hist = get_cond_summary(table, other_col, bucket_idx, col_c)
        if not cond_hist:
            return None

        # Estimate using conditional histogram
        return self._estimate_from_conditional_hist(cond_hist, op, val)

    def _estimate_ind_sel(self, context, table, pred):
        _, col, op, val = pred
        hist = context.histograms.get(table, col)
        if not hist:
            return 0.1

        try:
            val = float(val)
        except:
            return 0.05

        total = sum(b.count for b in hist)
        match = 0
        for b in hist:
            b.min_val = b.min_val if b.min_val != '' else 0
            b.max_val = b.max_val if b.max_val != '' else 0
            if b.count == 0:
                continue
            if op == '=' and b.min_val <= val <= b.max_val:
                match += b.count / max(1, b.distinct_count)
            elif op == '<':
                if val > b.min_val:
                    frac = min(1.0, (val - b.min_val) / (b.get_range() or 1))
                    match += b.count * frac
            elif op == '>':
                if val < b.max_val:
                    frac = min(1.0, (b.max_val - val) / (b.get_range() or 1))
                    match += b.count * frac

        return max(1.0 / (total + 1), match / total)


class JoinOverlapEstimator:
    def run(self, context: QueryEstimatorContext):
        joins = context.joins
        cards = context.effective_table_cards
        hists = context.histograms
        get_cond_summary = context.get_cond_summary_func
        final_card = 1.0
        for card in cards.values():
            final_card *= max(1.0, card)

        if not joins:
            context.details['join_details'] = {'type': 'cross_product', 'final_reduction_factor': 1.0}
            context.final_cardinality = final_card
            return

        reduction = 1.0
        steps = []
        for idx, (t1, c1, t2, c2) in enumerate(joins):
            h1 = hists.get(t1, c1)
            h2 = hists.get(t2, c2)
            if not h1 or not h2:
                sel = 1.0 / 100
                steps.append({'join_idx': idx, 'method': 'fallback', 'selectivity': sel})
                reduction *= sel
                continue

            sel = self._overlap_selectivity(h1, h2)
            steps.append({'join_idx': idx, 'method': 'hist_overlap', 'selectivity': sel})
            reduction *= sel

        context.details['join_details'] = {
            'type': 'hist_overlap',
            'steps': steps,
            'final_reduction_factor': reduction
        }
        context.final_cardinality = final_card * reduction

    def _overlap_selectivity(self, h1, h2):
        total1 = sum(b.count for b in h1)
        total2 = sum(b.count for b in h2)
        if total1 == 0 or total2 == 0:
            return 1.0 / 100

        match = 0
        for b1 in h1:
            for b2 in h2:
                overlap = max(0, min(b1.max_val, b2.max_val) - max(b1.min_val, b2.min_val))
                if overlap > 0:
                    match += (b1.count / max(1, b1.distinct_count)) * (overlap / (b2.get_range() or 1)) * b2.count

        return max(1.0 / (total1 * total2 + 1), match / (total1 * total2))

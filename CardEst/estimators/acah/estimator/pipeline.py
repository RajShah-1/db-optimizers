# query_pipeline.py
"""Modular pipeline components for ACAH V3 Estimator."""

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
        self.details['parsed'] = {'tables': tables_map, 'joins': joins, 'preds': predicates}


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
        sel = 1.0
        pred_sels = {}
        for entry in preds:
            idx = entry['original_index']
            pred = entry['pred']
            pred_sel = self._estimate_ind_sel(context, table, pred)
            pred_sels[idx] = ('ind', pred_sel)
            sel *= pred_sel
        return sel, pred_sels

    def _estimate_ind_sel(self, context, table, pred):
        _, col, op, val = pred
        hist = context.histograms.get((table, col))
        if not hist:
            return 0.1

        try:
            val = float(val)
        except:
            return 0.05

        total = sum(b.count for b in hist)
        match = 0
        for b in hist:
            b.min_val = b.min_val if b.min_val is not '' else 0
            b.max_val = b.max_val if b.max_val is not '' else 0
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
            h1, h2 = hists.get((t1, c1)), hists.get((t2, c2))
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

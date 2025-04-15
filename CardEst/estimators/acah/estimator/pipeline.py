# pipeline.py
"""Modular pipeline components for ACAH V3 Estimator (with actual comparison logging)."""
import json
from collections import defaultdict
import math
from pstats import Stats

from estimators.acah.catalog.stats_catalog import StatisticsCatalog


class QueryEstimatorContext:
    def __init__(self, conn):
        self.conn = conn
        self.histograms = StatisticsCatalog.get().get_histogram_catalog()
        self.column_types = StatisticsCatalog.get().get_column_types()
        self.correlated_pairs = StatisticsCatalog.get().get_correlated_pairs()
        self.reset()

    def reset(self, query=None, compare_with_actual=True):
        self.query = query
        self.compare_with_actual = compare_with_actual
        self.details = {}
        self.debug_info = {}
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

        print(json.dumps(context.debug_info, indent=2))


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
        debug_preds = []

        for table, card in context.table_cardinalities.items():
            if card <= 0:
                eff_cards[table] = 0
                table_sels[table] = 0.0
                continue

            sel, pred_sel_detail, debug_entries = self._estimate_table_sel(context, table, preds_by_table[table], card)
            eff_cards[table] = max(1, card * sel)
            table_sels[table] = sel
            pred_sels.update(pred_sel_detail)
            debug_preds.extend(debug_entries)

        context.effective_table_cards = eff_cards
        context.details['effective_cards'] = eff_cards
        context.details['table_selectivities'] = table_sels
        context.details['predicate_selectivities'] = pred_sels
        context.debug_info['predicate_debug'] = debug_preds

<<<<<<< HEAD
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
=======
    def _estimate_table_sel(self, context, table, preds, table_card):
        '''
        Applies selectivity of all the table level predicates.

        TODO: Use conditional summaries here.
        '''
        sel = 1.0
        pred_sels = {}
        debug = []
        for entry in preds:
            idx = entry['original_index']
            pred = entry['pred']
            pred_sel = self._estimate_ind_sel(context, table, pred)
            pred_sels[idx] = ('ind', pred_sel)
            sel *= pred_sel

            debug_entry = {
                'table': table,
                'col': pred[1],
                'operator': pred[2],
                'value': pred[3],
                'estimated_selectivity': pred_sel,
                'estimated_cardinality': pred_sel * table_card,
            }

            if context.compare_with_actual:
                try:
                    cursor = context.conn.cursor()
                    query = f'SELECT COUNT(*) FROM "{table}" WHERE "{pred[1]}" {pred[2]} ?'
                    cursor.execute(query, (pred[3],))
                    actual = cursor.fetchone()[0]
                    debug_entry['actual_cardinality'] = actual
                    debug_entry['actual/est'] = actual / (pred_sel * table_card)
                    debug_entry['est/actual'] = (pred_sel * table_card) / actual
                except Exception as e:
                    debug_entry['actual_cardinality'] = None
                    debug_entry['error'] = str(e)

            debug.append(debug_entry)

        return sel, pred_sels, debug
>>>>>>> a1316d3 (General cleanup; before merging cond-summary)

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
            return 0.1 # TODO: Hack! Fix this for strings

        try:
            val = float(val)
        except:
            return 0.05

        total = sum(b.count for b in hist)
        match = 0
        for b in hist:
            b.min_val = b.min_val if b.min_val != '' else 0
            b.max_val = b.max_val if b.max_val != '' else 1e9
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
        debug_joins = []

        for idx, (t1, c1, t2, c2) in enumerate(joins):
            h1 = hists.get(t1, c1)
            h2 = hists.get(t2, c2)
            est_sel = 1.0 / 100
            method = 'fallback'
            if h1 and h2:
                est_sel = self._overlap_selectivity(h1, h2)
                method = 'hist_overlap'

            steps.append({'join_idx': idx, 'method': method, 'selectivity': est_sel})
            reduction *= est_sel

            debug_entry = {
                'join_index': idx,
                'table1': t1, 'column1': c1,
                'table2': t2, 'column2': c2,
                'estimated_selectivity': est_sel,
                'estimated_join_cardinality': est_sel * cards[t1] * cards[t2]
            }

            if context.compare_with_actual:
                try:
                    cursor = context.conn.cursor()
                    q = f'''
                        SELECT COUNT(*) FROM "{t1}" JOIN "{t2}"
                        ON "{t1}"."{c1}" = "{t2}"."{c2}"
                    '''
                    cursor.execute(q)
                    actual = cursor.fetchone()[0]
                    debug_entry['actual_cardinality'] = actual

                    debug_entry['actual/est'] = actual / (est_sel * cards[t1] * cards[t2])
                    debug_entry['est/actual'] = est_sel * cards[t1] * cards[t2] / actual
                except Exception as e:
                    debug_entry['actual_cardinality'] = None
                    debug_entry['error'] = str(e)

            debug_joins.append(debug_entry)

        context.details['join_details'] = {
            'type': 'hist_overlap',
            'steps': steps,
            'final_reduction_factor': reduction
        }
        context.final_cardinality = final_card * reduction
        context.debug_info['join_debug'] = debug_joins

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

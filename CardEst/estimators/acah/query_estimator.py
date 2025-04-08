# query_estimator.py
"""Online estimation logic for ACAH Estimator V3. Calculates cardinality."""

from collections import defaultdict
import numpy as np 
import math

from .structures import Bucket 
from . import query_parser 
from . import config 

class QueryEstimator:
    """Parses queries and estimates cardinality using provided stats and summaries."""

    def __init__(self, db_conn, histograms, column_types, correlated_pairs):
        """Initializes with stats needed for estimation."""
        self.conn = db_conn 
        self.histograms = histograms
        self.column_types = column_types
        self.correlated_pairs = correlated_pairs
        self.current_estimation_details = {} 
        self.table_row_counts_cache = {}
        self.ndv_cache = {}

        # self._warmup_caches()

    def _warmup_caches(self):
        """Precompute and cache row counts and NDVs for all tables/columns."""
        print("[ACAH] Populating row count and NDV caches...")

        cursor = self.conn.cursor()

        for table, columns in self.column_types.items():
            # Table row count
            try:
                cursor.execute(f'SELECT COUNT(*) FROM "{table}"')
                count = cursor.fetchone()[0]
                self.table_row_counts_cache[table] = count
            except Exception as e:
                print(f"  Warning: Failed to get row count for {table}: {e}")
                self.table_row_counts_cache[table] = 1

            # Column NDVs
            for column in columns:
                try:
                    cursor.execute(f'SELECT COUNT(DISTINCT "{column}") FROM "{table}" WHERE "{column}" IS NOT NULL')
                    ndv = max(1, cursor.fetchone()[0])
                    self.ndv_cache[(table, column)] = ndv
                except Exception as e:
                    print(f"  Warning: Failed to get NDV for {table}.{column}: {e}")
                    self.ndv_cache[(table, column)] = 100  # Fallback



    def _add_detail(self, key, value):
        """Adds intermediate results to the details dictionary for feedback."""
        self.current_estimation_details[key] = value

    def estimate(self, query, get_cond_summary_func):
        """
        Estimates cardinality for the given query.

        Args:
            query (str): The SQL query string.
            get_cond_summary_func (callable): Function provided by FeedbackHandler 
                                             to retrieve conditional summaries (checks cache).

        Returns:
            tuple: (estimated_cardinality, estimation_details_dict)
        """
        self.current_estimation_details = {'query': query} 
        try:
            tables_map, join_conditions, predicates = query_parser.parse_query(query, self.column_types) 
            self._add_detail('parsed', {'tables': tables_map, 'joins': join_conditions, 'preds': predicates})
            if not tables_map: return 1.0, self.current_estimation_details

            # 1. Initial Cardinalities
            table_cards = self.get_table_cards_from_cache(tables_map)

            self._add_detail('initial_cards', table_cards.copy())

            # 2. Single-Table Selectivities (using conditional summaries where possible)
            preds_by_table = defaultdict(list)
            for idx, pred in enumerate(predicates): preds_by_table[pred[0]].append({'original_index': idx, 'pred': pred})

            effective_table_cards = {}
            table_selectivities = {}
            predicate_selectivities = {} 
            for table, original_card in table_cards.items():
                 if original_card > 0:
                     sel, pred_sels_details = self._estimate_single_table_selectivity(
                         table, preds_by_table[table], original_card, get_cond_summary_func)
                     table_selectivities[table] = sel
                     effective_table_cards[table] = max(1, original_card * sel)
                     predicate_selectivities.update(pred_sels_details) 
                 else:
                     table_selectivities[table] = 0.0; effective_table_cards[table] = 0.0 
            self._add_detail('table_selectivities', table_selectivities)
            self._add_detail('predicate_selectivities', predicate_selectivities) 
            self._add_detail('effective_cards', effective_table_cards)

            # 3. Join Estimation (using histogram overlap)
            final_card, join_details = self._estimate_joins_hist_overlap(
                tables_map, join_conditions, effective_table_cards)
            self._add_detail('join_details', join_details) 
            
            self._add_detail('final_card', final_card)
            return max(1.0, final_card), self.current_estimation_details.copy() 
            
        except Exception as e:
            print(f"Error during V3 cardinality estimation: {e}")
            import traceback; traceback.print_exc()
            self._add_detail('error', str(e)); return 1.0, self.current_estimation_details.copy()

    def get_table_cards_from_cache(self, tables_map):
        table_cards = {}
        cursor = self.conn.cursor()
        for alias, table in tables_map.items():   
            if table in self.table_row_counts_cache:
                table_cards[table] = self.table_row_counts_cache[table]
            else:
                try:
                    cursor.execute(f'SELECT COUNT(*) FROM "{table}"') 
                    res = cursor.fetchone(); count = res[0] if res else 0
                    self.table_row_counts_cache[table] = count
                    table_cards[table] = count
                except Exception: 
                    table_cards[table] = 1
        return table_cards

    # --- Internal Estimation Helpers ---

    def _estimate_selectivity_from_hist(self, histogram, operator, value):
        """Estimates selectivity for a predicate given a histogram (list of Buckets)."""
        if not histogram: return 1.0 # No info, assume no filtering
        try:
            numeric_value = float(value)
        except ValueError:
            return 1.0  # Default estimate for unprocessable values

        total_rows_in_hist = sum(b.count for b in histogram)
        if total_rows_in_hist == 0: return 1.0 # Empty table/column

        selected_rows_est = 0
        try: numeric_value = float(value)
        except (ValueError, TypeError): return 0.01 # Cannot compare type, assume very low sel

        num_buckets = len(histogram)
        for i, bucket in enumerate(histogram):
            if bucket.count == 0: continue
            count, distinct = bucket.count, max(1, bucket.distinct_count)
            is_last = (i == num_buckets - 1)

            # Check different operators
            if operator == '=':
                if bucket.contains(numeric_value, is_last):
                    # Uniformity assumption within distinct values in bucket
                    sel_in_bucket = (1.0 / distinct) if not math.isclose(bucket.get_range(), 0) else 1.0
                    selected_rows_est += count * sel_in_bucket
                    break # Exact match found
            elif operator == '<':
                b_min, b_max = bucket.min_val, (bucket.max_val if bucket.max_val != '' else 0)   
                safe_b_range = bucket.get_range() if bucket.get_range() > 1e-9 else 1.0
                if b_max <= numeric_value and not is_last: selected_rows_est += count # Whole bucket included
                elif b_min < numeric_value: # Partial bucket
                    fraction = (numeric_value - b_min) / safe_b_range
                    selected_rows_est += count * max(0.0, min(1.0, fraction))
            elif operator == '>':
                b_min, b_max = bucket.min_val, bucket.max_val
                safe_b_range = bucket.get_range() if bucket.get_range() > 1e-9 else 1.0
                if b_min >= numeric_value: selected_rows_est += count # Whole bucket included
                elif b_max > numeric_value: # Partial bucket
                     fraction = (b_max - numeric_value) / safe_b_range
                     selected_rows_est += count * max(0.0, min(1.0, fraction))
            # TODO: Implement <=, >=, != operators similarly

        selectivity = selected_rows_est / total_rows_in_hist if total_rows_in_hist > 0 else 0.0
        # Avoid returning exactly 0 if possible, represents minimum selectivity guess
        return selectivity if selectivity > 1e-9 else (1.0 / (total_rows_in_hist + 1))

    def _estimate_selectivity_for_value_in_bucket(self, bucket, operator, value, is_last_bucket):
        """Estimates the fraction of a bucket selected by a predicate."""
        if bucket.count == 0: return 0.0
        b_min, b_max = bucket.min_val, bucket.max_val
        distinct = max(1, bucket.distinct_count)
        safe_b_range = bucket.get_range() if bucket.get_range() > 1e-9 else 1.0
        
        sel = 0.0
        if operator == '=':
            if bucket.contains(value, is_last_bucket):
                 # Uniformity assumption within distinct values
                 sel = (1.0 / distinct) if safe_b_range > 1e-9 else 1.0
        elif operator == '<':
            if value <= b_min: sel = 0.0
            elif value >= b_max: sel = 1.0 # Includes endpoint if last bucket logic handled in contains
            else: sel = (value - b_min) / safe_b_range
        elif operator == '>':
            if value >= b_max: sel = 0.0
            elif value <= b_min: sel = 1.0
            else: sel = (b_max - value) / safe_b_range
        # Add other ops...
        
        return max(0.0, min(1.0, sel))

    def _find_bucket_index(self, table, column, value):
        """Finds the index of the bucket containing the value in the base histogram."""
        hist = self.histograms.get((table, column))
        if not hist: return None
        try: num_value = float(value)
        except: return None # Value not suitable for numeric histogram
        
        num_buckets = len(hist)
        for i, bucket in enumerate(hist):
            # Use bucket's contains method
            if bucket.contains(num_value, i == num_buckets - 1): 
                return i
        return None # Value might be out of histogram range

    def _estimate_pred_given_pred(self, table, pred_c, pred_d, get_cond_summary_func, idx_c, idx_d):
        """Estimates Sel(Pred_D | Pred_C) using conditional mini-hists via getter."""
        _, col_c, op_c, val_c = pred_c
        _, col_d, op_d, val_d = pred_d
        hist_key_c = (table, col_c)
        if hist_key_c not in self.histograms: return None 

        base_hist_c = self.histograms[hist_key_c]
        total_rows_match_c, total_rows_match_c_and_d = 0.0, 0.0
        used_cond_summary_info = [] # Track summary usage for details

        try: num_val_c, num_val_d = float(val_c), float(val_d)
        except (ValueError, TypeError): return None # Predicate values not numeric

        num_buckets = len(base_hist_c)
        for i, bucket_c in enumerate(base_hist_c):
            if bucket_c.count == 0: continue
            is_last_c = (i == num_buckets - 1)
            
            # Estimate fraction/count of rows matching Pred_C within this bucket
            sel_c_in_bucket = self._estimate_selectivity_for_value_in_bucket(bucket_c, op_c, num_val_c, is_last_c)
            rows_match_c_this_bucket = bucket_c.count * sel_c_in_bucket
            if rows_match_c_this_bucket <= 1e-6: continue # Skip if negligible match
            
            total_rows_match_c += rows_match_c_this_bucket

            # Attempt to get conditional summary (mini-histogram for col_d)
            cond_mini_hist_d = get_cond_summary_func(table, col_c, i, col_d) 
            
            if cond_mini_hist_d:
                # Estimate Sel(Pred_D) using the retrieved mini-histogram
                sel_d_in_mini_hist = self._estimate_selectivity_from_hist(cond_mini_hist_d, op_d, num_val_d)
                total_rows_match_c_and_d += rows_match_c_this_bucket * sel_d_in_mini_hist
                used_cond_summary_info.append({'bucket_idx': i, 'method': 'cond_summary'})
            else: 
                # Fallback: Assume independence within this slice -> use overall Sel(Pred_D)
                # Pass original index of Pred_D for detail logging consistency
                sel_d_overall = self._estimate_independent_selectivity(table, pred_d, idx_d) 
                total_rows_match_c_and_d += rows_match_c_this_bucket * sel_d_overall
                used_cond_summary_info.append({'bucket_idx': i, 'method': 'fallback_independent'})

        if total_rows_match_c <= 1e-6: return 0.0 # Pred_C selects effectively nothing

        conditional_selectivity = total_rows_match_c_and_d / total_rows_match_c
        
        # Add details about this step
        self._add_detail(f'pred_pair_{idx_c}_{idx_d}_cond_sel', 
                         {'table': table, 'col_c': col_c, 'col_d': col_d, 
                          'cond_sel': conditional_selectivity, 'access_info': used_cond_summary_info})
                          
        return max(0.0, min(1.0, conditional_selectivity))

    def _estimate_independent_selectivity(self, table, predicate, predicate_idx):
        """Estimates selectivity using only the base histogram, logs details."""
        _, column, op, value = predicate

        hist_key = (table, column)
        sel = 0.1 # Default fallback
        bucket_idx_accessed = None
        method = 'fallback_guess'

        col_type = self.column_types.get(table, {}).get(column)
        is_numeric = col_type == 'numeric'

        if is_numeric and hist_key in self.histograms:
            method = 'base_histogram'
            hist = self.histograms[hist_key]
            sel = self._estimate_selectivity_from_hist(hist, op, value)
            bucket_idx_accessed = self._find_bucket_index(table, column, value)
        elif not is_numeric:
             method = 'fallback_non_numeric'
             if op == '=': sel = 0.05
             elif op in ['<', '>', '<=', '>=']: sel = 0.3
             else: sel = 0.1
        else: # Numeric but no histogram
             method = 'fallback_no_histogram'
             if op == '=': sel = 0.05
             elif op in ['<', '>', '<=', '>=']: sel = 0.3
             else: sel = 0.1
        
        # Log details for this independent estimation
        self._add_detail(f'pred_{predicate_idx}_ind_sel', 
                         {'table': table, 'col': column, 'op': op, 'val': value, 
                          'selectivity': sel, 'method': method, 'bucket_idx': bucket_idx_accessed})
        return sel

    def _estimate_single_table_selectivity(self, table, indexed_predicates, original_card, get_cond_summary_func):
        """Combines predicate selectivities, preferring conditional estimates."""
        if not indexed_predicates: return 1.0, {}

        pred_estimates = {} # {original_idx: (type, value)}
        processed_indices = set() # Set of original_indices

        num_preds = len(indexed_predicates)
        for i in range(num_preds):
             current_pred_info = indexed_predicates[i]; current_idx = current_pred_info['original_index']
             if current_idx in processed_indices: continue
             pred_c = current_pred_info['pred']; _, col_c, _, _ = pred_c
             
             found_correlated_partner = False
             for j in range(i + 1, num_preds):
                 partner_pred_info = indexed_predicates[j]; partner_idx = partner_pred_info['original_index']
                 if partner_idx in processed_indices: continue
                 pred_d = partner_pred_info['pred']; _, col_d, _, _ = pred_d
                 
                 pair_key = tuple(sorted((col_c, col_d)))
                 is_correlated = (table, pair_key[0], pair_key[1]) in self.correlated_pairs
                 
                 if is_correlated:
                      cond_sel_d_given_c = self._estimate_pred_given_pred(table, pred_c, pred_d, get_cond_summary_func, current_idx, partner_idx)
                      if cond_sel_d_given_c is not None:
                           # Estimate Sel(Pred_C) independently ONLY IF needed (already done inside cond?)
                           # Re-use independent logic but just get the value
                           sel_c = self._estimate_independent_selectivity(table, pred_c, current_idx) 
                           pair_sel = sel_c * cond_sel_d_given_c
                           
                           # Store detailed results, marking how selectivity was derived
                           pred_estimates[current_idx] = ('cond_base', sel_c) 
                           pred_estimates[partner_idx] = ('cond_dep', cond_sel_d_given_c) 
                           pred_estimates[f'pair_{current_idx}_{partner_idx}'] = ('pair', pair_sel) 
                           
                           processed_indices.add(current_idx); processed_indices.add(partner_idx)
                           found_correlated_partner = True; break 
                      # Try D -> C?

             if not found_correlated_partner and current_idx not in processed_indices:
                  sel_i = self._estimate_independent_selectivity(table, pred_c, current_idx) 
                  pred_estimates[current_idx] = ('ind', sel_i)
                  processed_indices.add(current_idx)

        # Combine selectivities
        final_selectivity = 1.0; pairs_multiplied = set()
        for i in range(num_preds):
            original_idx = indexed_predicates[i]['original_index']
            if original_idx not in pred_estimates: continue 
            est_type, est_value = pred_estimates[original_idx]
            
            if est_type == 'ind': final_selectivity *= est_value
            elif est_type == 'cond_base': # Find and multiply by the combined pair selectivity once
                pair_found = False
                for j in range(i + 1, num_preds):
                     partner_idx = indexed_predicates[j]['original_index']
                     # Check both (i,j) and (j,i) pair keys
                     pair_key_str_ij = f'pair_{original_idx}_{partner_idx}'
                     pair_key_str_ji = f'pair_{partner_idx}_{original_idx}'
                     pair_tuple = tuple(sorted((original_idx, partner_idx)))

                     used_key = None
                     if pair_key_str_ij in pred_estimates: used_key = pair_key_str_ij
                     elif pair_key_str_ji in pred_estimates: used_key = pair_key_str_ji
                     
                     if used_key and pair_tuple not in pairs_multiplied:
                          final_selectivity *= pred_estimates[used_key][1] 
                          pairs_multiplied.add(pair_tuple); pair_found = True; break
        
        return max(0.0, min(1.0, final_selectivity)), pred_estimates

    def _estimate_joins_hist_overlap(self, tables_map, join_conditions, effective_table_cards):
        """Estimates join cardinality using histogram overlap, returns final card and details."""
        join_details = {'steps': [], 'type': 'hist_overlap' if join_conditions else 'no_joins'}
        
        if not tables_map or not effective_table_cards: return 0.0, join_details
        if not join_conditions: 
             final_card = 1.0
             for card in effective_table_cards.values(): final_card *= max(1.0, card)
             join_details['type'] = 'cross_product'
             return max(1.0, final_card), join_details

        # Assume independence between joins for simplicity (major limitation)
        # Start with cross product of effective cardinalities
        current_card = 1.0
        for card in effective_table_cards.values(): current_card *= max(1.0, card)
        reduction_factor = 1.0
        join_details['initial_cross_product'] = current_card

        for idx, (t1, c1, t2, c2) in enumerate(join_conditions):
            step_detail = {'join_idx': idx, 'join': f'{t1}.{c1}={t2}.{c2}'}
            hist1, hist2 = self.histograms.get((t1, c1)), self.histograms.get((t2, c2))
            card1, card2 = effective_table_cards.get(t1, 1.0), effective_table_cards.get(t2, 1.0)
            join_selectivity = 0.1 # Default guess

            if not hist1 or not hist2 or card1 <= 1e-6 or card2 <= 1e-6: # Fallback if no stats/empty inputs
                step_detail['method'] = 'selinger_ndv_fallback'
                ndv1 = sum(b.distinct_count for b in hist1) if hist1 else self._get_fallback_ndv(t1, c1)
                ndv2 = sum(b.distinct_count for b in hist2) if hist2 else self._get_fallback_ndv(t2, c2)
                join_selectivity = 1.0 / max(1.0, float(ndv1), float(ndv2)) # Ensure float division
                step_detail['ndv1'] = ndv1; step_detail['ndv2'] = ndv2
            else: # Histogram overlap calculation
                step_detail['method'] = 'hist_overlap'
                estimated_join_size = 0
                original_card1 = sum(b.count for b in hist1); original_card2 = sum(b.count for b in hist2)
                scale1 = (card1 / original_card1) if original_card1 > 0 else 0
                scale2 = (card2 / original_card2) if original_card2 > 0 else 0

                if scale1 > 1e-6 and scale2 > 1e-6: # Only proceed if effective cards are non-trivial
                    for i, bucket1 in enumerate(hist1):
                        is_last1 = (i == len(hist1) - 1)
                        eff_b1_count = bucket1.count * scale1
                        # Scale distinct count proportionally, but ensure it's at least 1 if count > 0
                        eff_b1_distinct = max(1.0, bucket1.distinct_count * scale1) if eff_b1_count > 1e-6 else 0.0 
                        if eff_b1_count <= 1e-6 or eff_b1_distinct <= 1e-6: continue
                        
                        matching_rows_h2_eff = 0.0
                        for j, bucket2 in enumerate(hist2):
                            is_last2 = (j == len(hist2) - 1)
                            eff_b2_count = bucket2.count * scale2
                            if eff_b2_count <= 1e-6: continue
                            
                            # Check overlap between bucket1 and bucket2 ranges
                            b1_min, b1_max = bucket1.min_val, bucket1.max_val
                            b2_min, b2_max = bucket2.min_val, bucket2.max_val
                            overlap_min = max(b1_min, b2_min); overlap_max = min(b1_max, b2_max)
                            
                            # Ensure overlap calculation handles endpoints correctly
                            if overlap_max >= overlap_min: 
                                safe_b2_range = bucket2.get_range() if bucket2.get_range() > 1e-9 else 1.0
                                overlap_width = overlap_max - overlap_min
                                # Add small epsilon for comparison if endpoints match
                                if math.isclose(overlap_width, 0) and math.isclose(overlap_min, overlap_max):
                                     # Single point overlap - check if point is within both bucket ranges strictly (using contains)
                                     if bucket1.contains(overlap_min, is_last1) and bucket2.contains(overlap_min, is_last2):
                                          # How to estimate fraction for single point? Use distinct count?
                                          overlap_fraction = (1.0 / max(1, bucket2.distinct_count)) if not math.isclose(safe_b2_range, 0) else 1.0
                                     else: overlap_fraction = 0.0
                                else: # Range overlap
                                     overlap_fraction = overlap_width / safe_b2_range

                                overlap_fraction = max(0.0, min(1.0, overlap_fraction))
                                matching_rows_h2_eff += eff_b2_count * overlap_fraction
                        
                        # Combine: Assume uniform distribution within distinct values
                        estimated_join_size += (eff_b1_count / eff_b1_distinct) * matching_rows_h2_eff

                    cross_product_eff = card1 * card2
                    join_selectivity = estimated_join_size / cross_product_eff if cross_product_eff > 1e-6 else 0.0
                    step_detail['estimated_join_size'] = estimated_join_size
                else: join_selectivity = 0.0 # If effective card is zero

            join_selectivity = max(0.0, min(1.0, join_selectivity))
            # Avoid zero selectivity trap
            if estimated_join_size > 1e-6 and join_selectivity < 1e-9: join_selectivity = 1.0 / (cross_product_eff + 1)

            reduction_factor *= join_selectivity
            step_detail['selectivity'] = join_selectivity
            join_details['steps'].append(step_detail)

        final_card_est = max(1.0, current_card * reduction_factor)
        join_details['final_reduction_factor'] = reduction_factor
        return final_card_est, join_details

    def _get_fallback_ndv(self, table, column):
        cache_key = (table, column)
        if cache_key in self.ndv_cache:
            return self.ndv_cache[cache_key]

        cursor = self.conn.cursor()
        try:
            cursor.execute(f'SELECT COUNT(DISTINCT "{column}") FROM "{table}" WHERE "{column}" IS NOT NULL') 
            res = cursor.fetchone(); ndv = max(1, res[0] if res else 1)
            self.ndv_cache[cache_key] = ndv
            return ndv
        except Exception:
            return 100


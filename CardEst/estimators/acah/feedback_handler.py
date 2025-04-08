# feedback_handler_ml.py
"""Manages feedback loop for ACAH V3, using ML model guidance."""

import time
from collections import OrderedDict, defaultdict
import math
import numpy as np 

from .structures import Bucket
from . import config
# Import concrete ML functions
from . import ml_models # Use the concrete implementations

class FeedbackHandlerML:
    """Handles learning using ML guidance and applying refinements."""

    def __init__(self, db_conn, histograms_ref, column_types_ref, correlated_pairs_ref):
        """Initializes with refs to stats and ML model path config."""
        self.conn = db_conn
        self.histograms = histograms_ref 
        self.column_types = column_types_ref
        self.correlated_pairs = correlated_pairs_ref
        
        self.cond_summary_cache = OrderedDict() 
        self.cond_summary_pair_usage = defaultdict(int) 
        self.refinement_queue = []
        
        print("FeedbackHandlerML initialized.")

    def _find_bucket_index(self, hist, value):
        try:
            num_val = float(value)
        except:
            return None
        for i, bucket in enumerate(hist):
            if bucket.contains(num_val, i == len(hist) - 1):
                return i
        return None
    
    def materialize_all_relevant_summaries(self, query, estimation_details):
        """Builds all conditional summaries and applies base histogram splits for the given query."""
        parsed = estimation_details.get('parsed', {})
        preds = parsed.get('preds', [])
        tables = parsed.get('tables', {})

        self._split_all_predicate_buckets(preds)
        self._build_all_correlated_conditional_summaries(preds)

    def _split_all_predicate_buckets(self, predicates):
        """Splits buckets for all predicate columns involved in the query."""
        for table, col, op, val in predicates:
            hist_key = (table, col)
            hist = self.histograms.get(hist_key)
            if not hist:
                continue

            idx = self._find_bucket_index(hist, val)
            if idx is None or idx >= len(hist):
                continue

            bucket = hist[idx]
            if bucket.count < config.MIN_BUCKET_SIZE * 2 or bucket.distinct_count <= 1:
                continue

            new_buckets = self._split_single_bucket_internal(table, col, bucket, val)
            if new_buckets:
                self.histograms[hist_key] = hist[:idx] + new_buckets + hist[idx+1:]
                self._invalidate_cond_cache_for_col(table, col)
    
    def _build_all_correlated_conditional_summaries(self, predicates):
        """For all correlated (col_c, col_d) pairs in predicates, build mini-histograms for each bucket of col_c."""
        from collections import defaultdict
        pred_cols_by_table = defaultdict(set)

        for t, c, _, _ in predicates:
            pred_cols_by_table[t].add(c)

        for table, cols in pred_cols_by_table.items():
            cols_list = list(cols)
            for i in range(len(cols_list)):
                for j in range(i + 1, len(cols_list)):
                    col_c, col_d = cols_list[i], cols_list[j]
                    pair_key = tuple(sorted((col_c, col_d)))
                    if (table, pair_key[0], pair_key[1]) not in self.correlated_pairs:
                        continue

                    self._build_conditional_summaries_for_pair(table, col_c, col_d)

    def _build_conditional_summaries_for_pair(self, table, col_c, col_d):
        """Builds conditional mini-histograms for col_d, conditioned on all buckets of col_c."""
        hist_key_c = (table, col_c)
        base_hist = self.histograms.get(hist_key_c)
        if not base_hist:
            return

        for b_idx, bucket in enumerate(base_hist):
            if bucket.count < config.MIN_BUCKET_SIZE:
                continue

            mini_hist = self._build_conditional_mini_hist_internal(table, col_c, bucket, col_d)
            if mini_hist:
                key = (table, col_c, b_idx, col_d)
                self.cond_summary_cache[key] = mini_hist


    # --- Cache Management --- 
    def get_cond_summary(self, table, col_c, bucket_idx_c, col_d):
        """Retrieves conditional summary from LRU cache."""
        cache_key = (table, col_c, bucket_idx_c, col_d)
        summary = self.cond_summary_cache.get(cache_key)
        if summary is not None:
            self.cond_summary_cache.move_to_end(cache_key) # Mark as recently used
            pair_key = tuple(sorted((col_c, col_d)))
            self.cond_summary_pair_usage[(table, pair_key[0], pair_key[1])] += 1 
        return summary

    def _check_cache_eviction(self):
        """Evicts LRU conditional summaries if cache exceeds budget."""
        while len(self.cond_summary_cache) > config.COND_SUMMARY_CACHE_SIZE:
             self.cond_summary_cache.popitem(last=False)

    def _invalidate_cond_cache_for_col(self, table, col_c):
         """Removes cached summaries conditioned ON col_c (e.g., after hist split)."""
         keys_to_remove = [key for key in self.cond_summary_cache if key[0] == table and key[1] == col_c]
         if keys_to_remove:
              for key in keys_to_remove: del self.cond_summary_cache[key]

    # --- Learning Logic ---
    def learn_from_error(self, query, estimated_card, actual_card, estimation_details):
        """Uses ML models to attribute error and suggest refinements."""
        if actual_card is None or actual_card <= 0 or estimated_card <= 0: return
        q_error = max(estimated_card / actual_card, actual_card / estimated_card)
        if q_error <= config.Q_ERROR_THRESHOLD: return 

        print(f"\nHigh Q-Error detected! Est: {estimated_card:.1f}, Actual: {actual_card:.1f}, Q: {q_error:.2f}")
        
        try:
            estimation_details['q_error'] = q_error; estimation_details['actual_card'] = actual_card
            feature_vector, features_dict = ml_models.extract_features(
                query, estimation_details, self.histograms, self.correlated_pairs)
            
            predicted_faults = ml_models.predict_faulty_component(feature_vector)
            if not predicted_faults: return 

            # Pass histograms ref to action prediction if needed by rules
            suggested_action = ml_models.predict_refinement_action(
                features_dict, predicted_faults, estimation_details, self.histograms 
            )
            
            # Log data point regardless of whether action is queued (for model improvement)
            ml_models.log_training_data(features_dict, feature_vector, q_error, actual_card, suggested_action, None)

            if suggested_action: self.queue_refinement(suggested_action) 

        except Exception as e: print(f"  Error during ML-guided learning: {e}"); import traceback; traceback.print_exc()

    # --- Refinement Application ---
    def queue_refinement(self, task):
        """Adds ML-suggested task if valid and within limits."""
        if isinstance(task, tuple) and len(task) > 1:
            if task not in self.refinement_queue and len(self.refinement_queue) < config.ADAPTATION_QUEUE_LIMIT:
                self.refinement_queue.append(task)
                # print(f"  Action added to queue: {task}") # Reduce noise
        # else: print(f"  Warning: Invalid task format suggested: {task}") # Reduce noise

    def apply_refinements(self):
        """Applies queued refinements by executing traditional actions."""
        if not self.refinement_queue: return 0
        print(f"\nApplying {len(self.refinement_queue)} ML-suggested V3 refinement tasks...")
        processed_tasks = 0; queue_copy = list(self.refinement_queue); self.refinement_queue.clear() 
        
        for task in queue_copy: 
            task_type = task[0]; applied = False
            try:
                 if task_type == 'rebuild_cond_hist' or task_type == 'create_cond_hist':
                     self._check_cache_eviction() 
                     _, table, col_c, bucket_idx, col_d = task
                     cache_key = (table, col_c, bucket_idx, col_d); hist_key_c = (table, col_c)
                     if hist_key_c in self.histograms and bucket_idx < len(self.histograms[hist_key_c]):
                          bucket_c = self.histograms[hist_key_c][bucket_idx]
                          if bucket_c.count >= config.MIN_BUCKET_SIZE: 
                               mini_hist = self._build_conditional_mini_hist_internal(table, col_c, bucket_c, col_d) 
                               if mini_hist:
                                    self.cond_summary_cache[cache_key] = mini_hist
                                    self.cond_summary_cache.move_to_end(cache_key); applied = True
                               elif cache_key in self.cond_summary_cache: del self.cond_summary_cache[cache_key]
                 elif task_type == 'split_bucket':
                     _, table, col, bucket_idx, target_value = task
                     hist_key = (table, col)
                     if hist_key in self.histograms and bucket_idx < len(self.histograms[hist_key]):
                          hist = self.histograms[hist_key]; bucket_to_split = hist[bucket_idx]
                          # Check size and distinct count before attempting split
                          if bucket_to_split.count >= config.MIN_BUCKET_SIZE * 2 and bucket_to_split.distinct_count > 1:
                              new_buckets = self._split_single_bucket_internal(table, col, bucket_to_split, target_value) 
                              if new_buckets:
                                   self.histograms[hist_key] = hist[:bucket_idx] + new_buckets + hist[bucket_idx+1:]
                                   self._invalidate_cond_cache_for_col(table, col); applied = True
                 # Add other refinement task executions ('merge_buckets' etc.)
            except Exception as e: print(f"    Error applying task {task}: {e}")
            if applied: processed_tasks += 1
        print(f"Finished applying {processed_tasks} V3 tasks.")
        return processed_tasks

    # --- Internal Helpers for executing actions ---
    # (Need _build_conditional_mini_hist_internal, _split_single_bucket_internal)
    # (Copy these from V3 placeholder feedback_handler_ml.py)
    def _build_conditional_mini_hist_internal(self, table, col_c, bucket_c, col_d):
        # ... (Logic from V3 placeholder feedback_handler_ml.py) ...
        cursor = self.conn.cursor(); b_min, b_max = bucket_c.min_val, bucket_c.max_val
        query = f'SELECT "{col_d}" FROM "{table}" WHERE "{col_c}" >= ? AND "{col_c}" <= ? AND "{col_d}" IS NOT NULL ORDER BY "{col_d}"'
        try:
            cursor.execute(query, (b_min, b_max))
            sorted_values_d = [row[0] for row in cursor.fetchall()]; num_rows_d = len(sorted_values_d)
            if num_rows_d < config.COND_HIST_BUCKETS: return None 
            mini_hist = []; target_rows_per_bucket = math.ceil(num_rows_d / config.COND_HIST_BUCKETS)
            current_index = 0
            while current_index < num_rows_d:
                start_index = current_index; end_index = min(start_index + target_rows_per_bucket, num_rows_d)
                if end_index < num_rows_d: 
                    boundary_value = sorted_values_d[end_index - 1]
                    while end_index < num_rows_d and sorted_values_d[end_index] == boundary_value: end_index += 1
                bucket_values = sorted_values_d[start_index:end_index]
                if not bucket_values: break
                mh_min, mh_max = bucket_values[0], bucket_values[-1]
                mh_count = len(bucket_values); mh_distinct = len(np.unique(bucket_values)) if mh_count > 0 else 0
                mini_hist.append(Bucket(mh_min, mh_max, mh_count, mh_distinct)) 
                current_index = end_index
            if mini_hist: mini_hist[0].min_val = sorted_values_d[0]; mini_hist[-1].max_val = sorted_values_d[-1]; return mini_hist
            else: return None
        except Exception as e: print(f"    Error building mini-hist internal: {e}"); return None

    def _split_single_bucket_internal(self, table, column, bucket_to_split, target_value=None):
        # ... (Logic from V3 placeholder feedback_handler_ml.py) ...
        cursor = self.conn.cursor(); b_min, b_max = bucket_to_split.min_val, bucket_to_split.max_val
        try:
            query = f'SELECT "{column}" FROM "{table}" WHERE "{column}" >= ? AND "{column}" <= ? ORDER BY "{column}"'
            cursor.execute(query, (b_min, b_max)); values = [row[0] for row in cursor.fetchall()]
            if len(values) < config.MIN_BUCKET_SIZE * 2: return None 
            median_idx = len(values) // 2; split_value = values[median_idx]
            # Crude adjustment if all values same up to median
            if median_idx > 0 and all(v == split_value for v in values[:median_idx]):
                 if all(v == split_value for v in values): return None # Cannot split
                 # Try finding first value different from split_value
                 first_diff_idx = -1
                 for k in range(median_idx, len(values)):
                      if values[k] != split_value: first_diff_idx = k; break
                 if first_diff_idx != -1: median_idx = first_diff_idx; split_value = values[median_idx-1] # Split before the change
                 
            cursor.execute(f'SELECT COUNT(*), COUNT(DISTINCT "{column}") FROM "{table}" WHERE "{column}" >= ? AND "{column}" <= ?', (b_min, split_value))
            count1, distinct1 = cursor.fetchone()
            cursor.execute(f'SELECT MIN("{column}") FROM "{table}" WHERE "{column}" > ? AND "{column}" <= ?', (split_value, b_max))
            actual_b2_min_res = cursor.fetchone(); actual_b2_min = actual_b2_min_res[0] if actual_b2_min_res and actual_b2_min_res[0] is not None else None
            if actual_b2_min is None: return None 
            cursor.execute(f'SELECT COUNT(*), COUNT(DISTINCT "{column}") FROM "{table}" WHERE "{column}" >= ? AND "{column}" <= ?', (actual_b2_min, b_max))
            count2, distinct2 = cursor.fetchone()
            if (count1 or 0) >= config.MIN_BUCKET_SIZE and (count2 or 0) >= config.MIN_BUCKET_SIZE:
                 b1 = Bucket(b_min, split_value, count1 or 0, distinct1 or 0)
                 b2 = Bucket(actual_b2_min, b_max, count2 or 0, distinct2 or 0)
                 return [b1, b2]
            else: return None 
        except Exception as e: print(f"    Error during bucket split internal: {e}"); return None

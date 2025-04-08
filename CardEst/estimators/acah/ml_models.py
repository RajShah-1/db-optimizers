# ml_models.py
"""Concrete ML models (using scikit-learn) for ACAH V3 Feedback Handler."""

import time
import random
import json
import os
import joblib 
import numpy as np
from sklearn.ensemble import RandomForestClassifier 
from sklearn.exceptions import NotFittedError
from collections import Counter, defaultdict

from . import config # Use V3 config

# --- Feature Extraction ---
# Define a fixed order of feature names for the numerical vector
FEATURE_ORDER = [
    'num_tables', 'num_joins', 'num_predicates', 'pred_eq_count', 'pred_range_count', 
    'pred_like_count', 'pred_isnull_count', 'log_estimated_card', 'log_q_error', 
    'under_over_estimate', 'avg_correlation', 'num_correlated_pairs', 'avg_hist_buckets',
    'min_predicate_selectivity', 'min_sel_bucket_count_norm', 'min_sel_bucket_ndv_norm', 
    'min_sel_bucket_range_norm', 'is_correlated_involved' 
    # Add more features: join key hist properties, predicate value positions etc.
]
NUM_FEATURES = len(FEATURE_ORDER) # Needed for dummy training

def _find_bucket_index_internal(hist, value):
    """Internal simplified version to find bucket index (no table/col args)."""
    if not hist: return None
    try: num_value = float(value)
    except: return None 
    num_buckets = len(hist)
    for i, bucket in enumerate(hist):
        if bucket.contains(num_value, i == num_buckets - 1): return i
    return None

def extract_features(query, estimation_details, histograms, correlated_pairs):
    """V1.1: Extracts features into a numerical vector and a dictionary."""
    features = {}
    parsed = estimation_details.get('parsed', {})
    preds = parsed.get('preds', []) # List of (table, col, op, val)
    joins = parsed.get('joins', []) # List of (t1, c1, t2, c2)
    tables = parsed.get('tables', {}) # Dict {alias: table}
    
    # Query Structure
    features['num_tables'] = len(tables)
    features['num_joins'] = len(joins)
    features['num_predicates'] = len(preds)
    pred_ops = [p[2] for p in preds] 
    op_counts = Counter(pred_ops)
    features['pred_eq_count'] = op_counts.get('=', 0) + op_counts.get('IS NULL', 0) # Treat IS NULL like equality?
    features['pred_range_count'] = sum(op_counts.get(op, 0) for op in ['<', '>', '<=', '>='])
    features['pred_like_count'] = op_counts.get('LIKE', 0)
    features['pred_isnull_count'] = op_counts.get('IS NULL', 0) + op_counts.get('IS NOT NULL', 0)

    # Estimation Results (Log scale)
    est_card = estimation_details.get('final_card', 1.0)
    features['log_estimated_card'] = np.log10(max(1.0, est_card))
    q_error = estimation_details.get('q_error', 1.0) # Assumes q_error was added to details
    features['log_q_error'] = np.log10(max(1.0, q_error))
    actual_card = estimation_details.get('actual_card', est_card)
    features['under_over_estimate'] = np.sign(est_card - actual_card) if q_error > 1.01 else 0

    # Stats State
    features['avg_correlation'] = np.mean(list(correlated_pairs.values())) if correlated_pairs else 0
    features['num_correlated_pairs'] = len(correlated_pairs)
    avg_buckets = np.mean([len(h) for h in histograms.values() if h]) if histograms else config.DEFAULT_NUM_BUCKETS
    features['avg_hist_buckets'] = avg_buckets

    # Features about correlated columns involved in predicates
    correlated_cols_in_query = set()
    pred_cols_by_table = defaultdict(set)
    for t, c, _, _ in preds: pred_cols_by_table[t].add(c)
    for table, cols in pred_cols_by_table.items():
        cols_list = list(cols)
        for i in range(len(cols_list)):
            for j in range(i + 1, len(cols_list)):
                 key = tuple(sorted((cols_list[i], cols_list[j])))
                 if (table, key[0], key[1]) in correlated_pairs:
                      correlated_cols_in_query.add(f"{table}.{key[0]}")
                      correlated_cols_in_query.add(f"{table}.{key[1]}")
    features['is_correlated_involved'] = 1 if correlated_cols_in_query else 0

    # Features related to the predicate with *minimum individual selectivity*
    pred_sels = estimation_details.get('predicate_selectivities', {})
    min_sel = 1.0; min_sel_pred_idx = -1; min_sel_pred_info = None
    for idx, (typ, sel) in pred_sels.items():
         # Consider only independent or base conditional selectivities
         if isinstance(idx, int) and typ in ['ind', 'cond_base'] and sel < min_sel: 
              min_sel = sel; min_sel_pred_idx = idx
    
    features['min_predicate_selectivity'] = min_sel if min_sel < 1.0 else 1.0 # Cap at 1.0

    # Initialize bucket features
    features['min_sel_bucket_count_norm'] = 0.0
    features['min_sel_bucket_ndv_norm'] = 0.0
    features['min_sel_bucket_range_norm'] = 0.0

    if min_sel_pred_idx != -1 and min_sel_pred_idx < len(preds):
         table, col, op, val = preds[min_sel_pred_idx]
         hist_key = (table, col)
         if hist_key in histograms:
              hist = histograms[hist_key]
              total_hist_rows = sum(b.count for b in hist)
              total_hist_range = (hist[-1].max_val - hist[0].min_val) if hist and hist[-1].max_val is not None and hist[0].min_val is not None else 1.0
              if math.isclose(total_hist_range, 0): total_hist_range = 1.0 # Avoid division by zero

              bucket_idx = _find_bucket_index_internal(hist, val) 
              if bucket_idx is not None and bucket_idx < len(hist):
                  bucket = hist[bucket_idx]
                  # Normalize bucket features
                  features['min_sel_bucket_count_norm'] = bucket.count / total_hist_rows if total_hist_rows > 0 else 0
                  features['min_sel_bucket_ndv_norm'] = bucket.distinct_count / bucket.count if bucket.count > 0 else 0
                  features['min_sel_bucket_range_norm'] = bucket.get_range() / total_hist_range
              
    # Ensure all features in FEATURE_ORDER are present, default to 0
    feature_vector = np.array([features.get(name, 0) for name in FEATURE_ORDER], dtype=float)
    # Replace NaN/Inf with 0 (important for ML models)
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

    return feature_vector, features # Return both vector and dict

# --- ML Model Loading / Prediction ---
_loaded_models = {}
def _load_model(model_path):
    """Loads a joblib model, caching it."""
    if model_path in _loaded_models: return _loaded_models[model_path]
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path); print(f"Loaded ML model from {model_path}")
            _loaded_models[model_path] = model; return model
        except Exception as e: print(f"Error loading ML model {model_path}: {e}")
    else: print(f"Warning: ML model file not found: {model_path}. Using default.")
    return None

# --- Fault Attribution Model ---
def predict_faulty_component(feature_vector):
    """Predicts probabilities for fault types using a loaded RandomForest model."""
    model_path = config.FAULT_ATTRIBUTION_MODEL_FILE
    model = _load_model(model_path)
    probabilities = None
    if model:
        try:
            if feature_vector.ndim == 1: feature_vector = feature_vector.reshape(1, -1)
            if feature_vector.shape[1] != model.n_features_in_:
                 print(f"ERROR: Feature mismatch! Model expects {model.n_features_in_}, got {feature_vector.shape[1]}. Using default.")
            else:
                 probabilities = model.predict_proba(feature_vector)[0] 
        except NotFittedError: print(f"Warning: Fault model {model_path} not fitted. Using default.")
        except Exception as e: print(f"Error during fault prediction: {e}. Using default.")
             
    if probabilities is None: # Default dummy logic
        probabilities = np.array([0.6, 0.1, 0.1, 0.2]) # [BASE, MISS, INACC, JOIN]
        np.random.shuffle(probabilities) 

    # Ensure probabilities match the order and number of FAULT_COMPONENT_LABELS
    if len(probabilities) != len(config.FAULT_COMPONENT_LABELS):
         print(f"ERROR: Model probability output size mismatch ({len(probabilities)}) vs labels ({len(config.FAULT_COMPONENT_LABELS)}). Using default.")
         probabilities = np.array([1.0 / len(config.FAULT_COMPONENT_LABELS)] * len(config.FAULT_COMPONENT_LABELS))

    ranked_faults = sorted(zip(config.FAULT_COMPONENT_LABELS, probabilities), key=lambda x: x[1], reverse=True)
    return ranked_faults

# --- Refinement Action Prediction (Rule-based using predicted fault) ---
def predict_refinement_action(features_dict, ranked_faults, estimation_details, histograms):
    """Suggests a refinement action based on the top predicted fault type (Rule-based)."""
    if not ranked_faults: return None
    top_fault_type = ranked_faults[0][0]
    # print(f"  Top predicted fault type: {top_fault_type}") # Reduce noise

    action = None
    parsed = estimation_details.get('parsed', {})
    preds = parsed.get('preds', [])
    
    # Find the predicate most likely associated with the error (e.g., min selectivity one)
    target_pred_info = None
    min_sel_pred_idx = -1
    pred_sels = estimation_details.get('predicate_selectivities', {})
    min_sel = 1.0
    for idx, (typ, sel) in pred_sels.items():
         if isinstance(idx, int) and typ in ['ind', 'cond_base'] and sel < min_sel: 
              min_sel = sel; min_sel_pred_idx = idx
    if min_sel_pred_idx != -1 and min_sel_pred_idx < len(preds):
         target_pred_info = {'idx': min_sel_pred_idx, 'pred': preds[min_sel_pred_idx]}
    elif preds: # Fallback to first predicate if min sel not found
         target_pred_info = {'idx': 0, 'pred': preds[0]}

    if top_fault_type == 'BASE_HIST' and target_pred_info:
        table, col, op, val = target_pred_info['pred']
        hist = histograms.get((table, col))
        bucket_idx = _find_bucket_index_internal(hist, val) 
        if bucket_idx is not None:
             action = ('split_bucket', table, col, bucket_idx, val) 

    elif top_fault_type == 'COND_SUMMARY_MISSING' or top_fault_type == 'COND_SUMMARY_INACCURATE':
         # Find a correlated pair involved
         correlated_pair_target = None
         pred_cols_by_table = defaultdict(set)
         for t, c, _, _ in preds: pred_cols_by_table[t].add(c)
         for table, cols in pred_cols_by_table.items():
             cols_list = list(cols); break_outer = False
             for i in range(len(cols_list)):
                 for j in range(i + 1, len(cols_list)):
                      key = tuple(sorted((cols_list[i], cols_list[j])))
                      # Use features_dict which should contain correlated_pairs from context
                      if (table, key[0], key[1]) in features_dict.get('correlated_pairs', {}): 
                           correlated_pair_target = (table, key[0], key[1]); break_outer = True; break
                 if break_outer: break
             if break_outer: break
             
         if correlated_pair_target:
              table, col1, col2 = correlated_pair_target
              # Find corresponding predicates and values
              val1, val2 = None, None; pred1, pred2 = None, None
              for p_idx, p in enumerate(preds):
                   if p[0] == table and p[1] == col1: val1 = p[3]; pred1 = {'idx': p_idx, 'pred': p};
                   if p[0] == table and p[1] == col2: val2 = p[3]; pred2 = {'idx': p_idx, 'pred': p};
              
              # Queue creation/rebuild based on the predicate identified as min selectivity (if possible)
              target_col_c, target_col_d, target_val_c, target_idx_c = (col1, col2, val1, pred1['idx']) if pred1 and pred1['idx'] == min_sel_pred_idx else (col2, col1, val2, pred2['idx'] if pred2 else -1)
              
              if target_val_c is not None and target_idx_c != -1:
                   hist_c = histograms.get((table, target_col_c))
                   bucket_idx_c = _find_bucket_index_internal(hist_c, target_val_c)
                   if bucket_idx_c is not None:
                        action_type = 'create_cond_hist' if top_fault_type == 'COND_SUMMARY_MISSING' else 'rebuild_cond_hist'
                        action = (action_type, table, target_col_c, bucket_idx_c, target_col_d)
                   
    elif top_fault_type == 'JOIN':
         joins = parsed.get('joins', [])
         if joins:
              t1, c1, t2, c2 = joins[0] # Target first join
              # Suggest splitting base histogram of one join key column (e.g., the one with min selectivity predicate, if involved)
              target_table, target_col = (t1, c1) # Default
              if target_pred_info and target_pred_info['pred'][0] == t1 and target_pred_info['pred'][1] == c1: pass # Keep t1, c1
              elif target_pred_info and target_pred_info['pred'][0] == t2 and target_pred_info['pred'][1] == c2: target_table, target_col = (t2, c2) # Switch target
              
              hist = histograms.get((target_table, target_col))
              # Split middle bucket as a heuristic if no target value known
              bucket_idx_to_split = len(hist) // 2 if hist else 0
              dummy_val = hist[bucket_idx_to_split].min_val if hist and bucket_idx_to_split < len(hist) else 0
              action = ('split_bucket', target_table, target_col, bucket_idx_to_split, dummy_val) 

    # if action: print(f"    ML Rules Suggest: {action}") # Reduce noise
    return action

# --- Data Logging ---
def log_training_data(features_dict, feature_vector, q_error, actual_card, suggested_action, outcome=None):
    """Logs data point for offline ML training."""
    # (Same logic as V3 placeholder version)
    log_entry = {'timestamp': time.time(), 'feature_version': features_dict.get('feature_version', 'unknown'),
                 'features_dict': features_dict, 'feature_vector': feature_vector.tolist(), 
                 'q_error': q_error, 'actual_card': actual_card, 
                 'suggested_action': suggested_action, 'outcome': outcome}
    try:
        log_dir = os.path.dirname(config.LOG_FILE_PATH);
        if log_dir and not os.path.exists(log_dir): os.makedirs(log_dir)
        with open(config.LOG_FILE_PATH, "a") as f: f.write(json.dumps(log_entry) + "\n")
    except Exception as e: print(f"Error logging training data: {e}")

# --- Dummy Model Training ---
def train_and_save_dummy_models(force_retrain=False):
    """Creates simple, dummy scikit-learn models if they don't exist."""
    # Fault Attribution Model
    model_path = config.FAULT_ATTRIBUTION_MODEL_FILE
    if not os.path.exists(model_path) or force_retrain:
        print(f"Training dummy Fault Attribution model ({model_path})...")
        # Generate dummy data matching NUM_FEATURES
        X_dummy = np.random.rand(len(config.FAULT_COMPONENT_LABELS) * 5, NUM_FEATURES) 
        y_dummy = np.array([i % len(config.FAULT_COMPONENT_LABELS) for i in range(len(config.FAULT_COMPONENT_LABELS) * 5)]) 
        model = RandomForestClassifier(n_estimators=10, random_state=42, class_weight='balanced')
        try:
            model.fit(X_dummy, y_dummy)
            model_dir = os.path.dirname(model_path);
            if model_dir and not os.path.exists(model_dir): os.makedirs(model_dir)
            joblib.dump(model, model_path); print("Dummy Fault Attribution model saved.")
        except Exception as e: print(f"Error training/saving dummy fault model: {e}")

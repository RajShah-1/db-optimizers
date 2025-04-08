# config.py
"""Configuration constants for ACAH Estimator V3."""
import os

# Histogram Settings
DEFAULT_NUM_BUCKETS = 50        # Initial target number of buckets per histogram
MIN_BUCKET_SIZE = 10          # Minimum rows for a bucket to justify its existence

# Correlation Settings
CORRELATION_METHOD = 'pearson'  # Method for numerical correlation ('pearson')
CORRELATION_THRESHOLD = 0.5     # Minimum absolute score to consider columns correlated
TOP_K_INITIAL_CORR = 5        # Precompute initial summaries for top K correlated pairs (now optional, handled by feedback)

# Conditional Summary Settings
COND_SUMMARY_TYPE = 'mini_hist' # Type of conditional summary ('mini_hist')
COND_HIST_BUCKETS = 10          # Number of buckets in conditional mini-histograms

# Feedback & Adaptation Settings
Q_ERROR_THRESHOLD = 5.0         # Q-Error threshold (max(est/act, act/est)) to trigger learning
ADAPTATION_BUDGET = 100         # Max distinct (table, col_c, col_d) PAIRS allowed to have summaries in cache
ADAPTATION_QUEUE_LIMIT = 50     # Max pending refinement tasks in the queue

# LRU Cache Settings (Max individual summary entries, e.g., per bucket)
# Estimate based on budget and typical bucket count
COND_SUMMARY_CACHE_SIZE = ADAPTATION_BUDGET * DEFAULT_NUM_BUCKETS * 2

# Persistence and Logging
STATS_FILE_DIR = "./data/"
STATS_FILENAME = "acah_stats_v3.pkl"
STATS_FILE = os.path.join(STATS_FILE_DIR, STATS_FILENAME)
CATALOG_CACHE_FILENAME = "acah_v3_catalog_cache.pkl"
CATALOG_CACHE_FILE = os.path.join(STATS_FILE_DIR, CATALOG_CACHE_FILENAME)

ML_MODEL_DIR = "./ml_models/"
FEATURE_VERSION = "1.1"         # For feature compatibility with models
FAULT_ATTRIBUTION_MODEL_FILENAME = f"fault_attribution_v{FEATURE_VERSION}.joblib"
FAULT_ATTRIBUTION_MODEL_FILE = os.path.join(ML_MODEL_DIR, FAULT_ATTRIBUTION_MODEL_FILENAME)
LOG_FILENAME = "acah_v3_training_log.jsonl"
LOG_FILE_PATH = os.path.join(ML_MODEL_DIR, LOG_FILENAME)

# Define fault component types/labels used for ML model training/prediction
FAULT_COMPONENT_LABELS = ['BASE_HIST', 'COND_SUMMARY_MISSING', 'COND_SUMMARY_INACCURATE', 'JOIN']
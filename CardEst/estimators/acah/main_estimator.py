# main_estimator.py
"""Main facade class for ACAH Estimator V3."""

import os
import pickle 
from ..CardinalityEstimator import CardinalityEstimator

from . import config
from .stats_builder import StatsBuilder
from .estimator.query_estimator import QueryEstimator
from .feedback_handler import FeedbackHandlerML
from . import ml_models

class ACAHv3Estimator(CardinalityEstimator): 
    """Facade class managing state, estimation, and feedback for ACAH V3."""

    def __init__(self, conn, train_dummy_models=False): # Added flag
        self.conn = None 
        try:
            super().__init__(conn) 
        except Exception as e:
            print(f"Error connecting to database: {e}")
            if conn: conn.close(); raise 

        # Core state & Components (same init as V3 placeholder version)
        self.histograms = {}; self.column_types = {}; self.correlated_pairs = {}
        self.stats_builder = StatsBuilder(self.conn)
        self.query_estimator = None 
        self.feedback_handler = None # Use FeedbackHandlerML

        # Option to train dummy models on init if files don't exist
        if train_dummy_models:
             print("Attempting to train dummy ML models...")
             ml_models.train_and_save_dummy_models(force_retrain=False) # Creates if not exist

        self._load_or_initialize_stats() # Loads stats, initializes components
        self._last_estimation_details_map = {} 

    
    def materialize_all_stats_for_query(self, query):
        _, estimation_details = self.query_estimator.estimate(query, self.feedback_handler.get_cond_summary)
        self.feedback_handler.materialize_all_relevant_summaries(query, estimation_details)


    # _load_or_initialize_stats (same as V3 placeholder version, uses FeedbackHandlerML)
    def _load_or_initialize_stats(self):
        loaded = False
        if os.path.exists(config.STATS_FILE):
            try:
                with open(config.STATS_FILE, "rb") as f:
                    saved_data = pickle.load(f)
                    if all(k in saved_data for k in ['histograms', 'column_types', 'correlated_pairs']):
                        self.histograms = saved_data['histograms']
                        self.column_types = saved_data['column_types']
                        self.correlated_pairs = saved_data['correlated_pairs']
                        print(f"Loaded ACAH V3 stats from {config.STATS_FILE}.")
                        loaded = True
                    else: print("V3 Stats file missing keys. Rebuilding.")
            except Exception as e: print(f"Error loading V3 stats file: {e}. Rebuilding.")
        if not loaded:
            try:
                histograms, column_types, correlated_pairs = self.stats_builder.build_all()
                self.histograms = histograms; self.column_types = column_types; self.correlated_pairs = correlated_pairs
                self.save_stats() 
            except Exception as e: print(f"FATAL: Error building initial V3 stats: {e}")
        self.query_estimator = QueryEstimator(self.conn, self.histograms, self.column_types, self.correlated_pairs)
        self.feedback_handler = FeedbackHandlerML(self.conn, self.histograms, self.column_types, self.correlated_pairs) 
        print("ACAHv3Estimator components initialized.")

    # --- Interface Methods (same implementations as V3 placeholder version) ---
    def estimate_cardinality(self, query): # ... uses self.feedback_handler.get_cond_summary ...
        if not self.query_estimator or not self.feedback_handler: return 1.0 
        estimate, details = self.query_estimator.estimate(query, self.feedback_handler.get_cond_summary)
        self._last_estimation_details_map[query] = details 
        return estimate

    def learn_from_error(self, query, actual_card): # ... calls self.feedback_handler.learn_from_error ...
        if not self.feedback_handler: return
        details = self._last_estimation_details_map.get(query) 
        if not details: return
        estimated_card = details.get('final_card', -1)
        if estimated_card < 0: return
        self.feedback_handler.learn_from_error(query, estimated_card, actual_card, details)

    def apply_pending_refinements(self, save_stats_after=True): # ... calls self.feedback_handler.apply_refinements ...
        if not self.feedback_handler: return 0
        processed_count = self.feedback_handler.apply_refinements()
        if processed_count > 0 and save_stats_after: self.save_stats()
        return processed_count

    def save_stats(self): # ... (same logic) ...
         try:
            data_dir = os.path.dirname(config.STATS_FILE)
            if data_dir and not os.path.exists(data_dir): os.makedirs(data_dir)
            with open(config.STATS_FILE, "wb") as f:
                saved_data = {'histograms': self.histograms, 'column_types': self.column_types, 'correlated_pairs': self.correlated_pairs}
                pickle.dump(saved_data, f)
            # print(f"{self.name}: Base stats saved to {config.STATS_FILE}.") 
         except Exception as e: print(f"Error saving stats file: {e}")

# --- Example Usage (V3) ---
if __name__ == '__main__':
    DB_PATH = './imdb_simple.db' # ADJUST AS NEEDED
    estimator = None 
    try:
        # Set train_dummy_models=True on first run if model files don't exist
        estimator = ACAHv3Estimator(DB_PATH, train_dummy_models=True) 

        test_query = "SELECT COUNT(*) FROM title t JOIN movie_info mi ON t.id = mi.movie_id WHERE t.production_year > 2000 AND mi.info_type_id = 3" # ADJUST AS NEEDED

        # --- Test Cycle ---
        # ... (Same test cycle as V3 placeholder version) ...
        print("-" * 20)
        estimated_card = estimator.estimate_cardinality(test_query)
        print(f"Query: {test_query}")
        print(f"[{estimator.name}] Estimated Cardinality (1st pass): {estimated_card:.1f}")
        print("-" * 20)
        try:
            cursor = estimator.conn.cursor() 
            cursor.execute(test_query) 
            actual_card = cursor.fetchone()[0]
            print(f"Actual Cardinality (simulated): {actual_card}")
            estimator.learn_from_error(test_query, actual_card)
            processed = estimator.apply_pending_refinements(save_stats_after=True)
            print(f"Applied {processed} ML-suggested refinements.")
        except Exception as e: print(f"Error during feedback simulation: {e}")
        print("-" * 20)
        estimated_card_after = estimator.estimate_cardinality(test_query)
        print(f"[{estimator.name}] Estimated Cardinality AFTER Feedback: {estimated_card_after:.1f}")
        print("-" * 20)

    except Exception as main_e:
         print(f"Failed to initialize or run estimator V3: {main_e}")
         import traceback; traceback.print_exc()
    finally:
        if estimator is not None: estimator.close_connection()

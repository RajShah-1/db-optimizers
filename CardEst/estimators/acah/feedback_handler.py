# feedback_handler.py
"""Handles refinement tasks in ACAH V3 (fail-fast mode, no ML)."""

from collections import defaultdict
import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier

class FeedbackHandlerML:
    """Eager refinement manager for ACAH V3 (fail-fast, non-ML mode)."""

    def __init__(self, conn, stats_catalog):
        self.conn = conn
        self.stats_catalog = stats_catalog
        self.hist_catalog = stats_catalog.histogram_catalog
        self.cond_catalog = stats_catalog.cond_summary_catalog
        self.column_types = stats_catalog.column_types
        self.correlated_pairs = stats_catalog.correlated_pairs
        self.classifier = RandomForestClassifier(n_estimators=100)
        self.problem_areas = ['histogram', 'correlation', 'conditional_summary']

    def _classify_problem_area(self, feedback_data):
        """Classify the main problem area in the query estimation.
        
        Args:
            feedback_data (dict): The feedback data containing query information
            
        Returns:
            str: One of 'histogram', 'correlation', or 'conditional_summary'
        """
        # Extract features for classification
        features = self._extract_classification_features(feedback_data)
        
        # For now, use a simple rule-based approach
        # In a real implementation, this would use the trained classifier
        q_error = feedback_data['q_error']
        details = feedback_data['estimation_details']
        parsed = details.get("parsed", {})
        preds = parsed.get("preds", [])
        joins = parsed.get("joins", [])

        # Check for histogram problems
        if any(self._is_histogram_problem(table, col, val, q_error) 
               for table, col, _, val in preds):
            return 'histogram'

        # Check for correlation problems
        if any(self._is_correlation_problem(t1, c1, t2, c2, q_error) 
               for t1, c1, t2, c2 in joins):
            return 'correlation'

        # Check for conditional summary problems
        if any(self._is_conditional_summary_problem(table, col_a, col_b, q_error) 
               for table, col_a, col_b in self._get_column_pairs(preds, joins)):
            return 'conditional_summary'

        # Default to histogram if no clear problem area
        return 'histogram'

    def _extract_classification_features(self, feedback_data):
        """Extract features for the classifier from feedback data."""
        features = []
        details = feedback_data['estimation_details']
        parsed = details.get("parsed", {})
        
        # Number of predicates
        features.append(len(parsed.get("preds", [])))
        
        # Number of joins
        features.append(len(parsed.get("joins", [])))
        
        # Query error
        features.append(feedback_data['q_error'])
        
        # Add more features as needed
        return features

    def _is_histogram_problem(self, table, column, value, q_error):
        """Check if a histogram problem exists for given predicate."""
        bucket = self.hist_catalog.find_bucket(table, column, value)
        if not bucket:
            return False
        return q_error > 2.0 and bucket.count > 1000

    def _is_correlation_problem(self, table1, col1, table2, col2, q_error):
        """Check if a correlation problem exists for given join."""
        corr_score = self.correlated_pairs.get((table1, col1, col2), 0.0)
        return q_error > 2.0 and corr_score < 0.3

    def _is_conditional_summary_problem(self, table, col_a, col_b, q_error):
        """Check if a conditional summary problem exists for given columns."""
        return q_error > 2.0 and not self.cond_catalog.has_summary(table, col_a, col_b)

    def _get_column_pairs(self, preds, joins):
        """Get all relevant column pairs from predicates and joins."""
        pairs = set()
        for table, col, _, _ in preds:
            pairs.add((table, col))
        for t1, c1, t2, c2 in joins:
            pairs.add((t1, c1))
            pairs.add((t2, c2))
        return pairs

    def analyze_feedback(self, feedback_data):
        """Analyze feedback data and determine what statistics need adjustment.
        
        Args:
            feedback_data (dict): Contains:
                - query: The SQL query
                - estimated_cardinality: Estimated cardinality
                - true_cardinality: Actual cardinality
                - estimation_details: Details from the estimation process
                - q_error: The query error
                - execution_time: Time taken for estimation
        
        Returns:
            dict: Recommendations for statistics adjustments:
                {
                    'histogram_splits': [
                        {'table': str, 'column': str, 'value': any, 'priority': float}
                    ],
                    'conditional_summaries': [
                        {'table': str, 'col_a': str, 'col_b': str, 'priority': float}
                    ],
                    'correlation_updates': [
                        {'table': str, 'col_a': str, 'col_b': str, 'score': float}
                    ]
                }
        """
        recommendations = {
            'histogram_splits': [],
            'conditional_summaries': [],
            'correlation_updates': []
        }

        # First, classify the main problem area
        problem_area = self._classify_problem_area(feedback_data)
        
        # Extract data from feedback
        query = feedback_data['query']
        est_card = feedback_data['estimated_cardinality']
        true_card = feedback_data['true_cardinality']
        q_error = feedback_data['q_error']
        details = feedback_data['estimation_details']

        # Analyze predicates
        parsed = details.get("parsed", {})
        preds = parsed.get("preds", [])
        joins = parsed.get("joins", [])

        # Focus analysis on the identified problem area
        if problem_area == 'histogram':
            self._analyze_histogram_problems(preds, q_error, recommendations)
        elif problem_area == 'correlation':
            self._analyze_correlation_problems(joins, q_error, recommendations)
        elif problem_area == 'conditional_summary':
            self._analyze_conditional_summary_problems(preds, joins, q_error, recommendations)

        return recommendations

    def _analyze_histogram_problems(self, preds, q_error, recommendations):
        """Analyze and recommend histogram splits."""
        for table, column, op, value in preds:
            priority = self._calculate_split_priority(table, column, value, q_error)
            if priority > 0:
                recommendations['histogram_splits'].append({
                    'table': table,
                    'column': column,
                    'value': value,
                    'priority': priority
                })

    def _analyze_correlation_problems(self, joins, q_error, recommendations):
        """Analyze and recommend correlation updates."""
        for t1, c1, t2, c2 in joins:
            new_score = self._calculate_correlation(t1, c1, c2)
            if new_score >= 0.3:  # Correlation threshold
                recommendations['correlation_updates'].append({
                    'table': t1,
                    'col_a': c1,
                    'col_b': c2,
                    'score': new_score
                })

    def _analyze_conditional_summary_problems(self, preds, joins, q_error, recommendations):
        """Analyze and recommend conditional summaries."""
        pred_cols_by_table = defaultdict(set)
        for t, c, _, _ in preds:
            pred_cols_by_table[t].add(c)
        for t1, c1, t2, c2 in joins:
            pred_cols_by_table[t1].add(c1)
            pred_cols_by_table[t2].add(c2)

        for table, cols in pred_cols_by_table.items():
            cols_list = list(cols)
            for col_a in cols_list:
                for col_b in cols_list:
                    if col_a == col_b:
                        continue
                    priority = self._calculate_summary_priority(table, col_a, col_b, q_error)
                    if priority > 0:
                        recommendations['conditional_summaries'].append({
                            'table': table,
                            'col_a': col_a,
                            'col_b': col_b,
                            'priority': priority
                        })

    def _calculate_split_priority(self, table, column, value, q_error):
        """Calculate priority for splitting a histogram bucket."""
        # Higher q_error means higher priority
        # Also consider value distribution in the bucket
        bucket = self.hist_catalog.find_bucket(table, column, value)
        if not bucket:
            return 0.0
        
        # Priority increases with:
        # 1. Higher q_error
        # 2. More values in the bucket
        # 3. Wider bucket range
        bucket_size = bucket.count
        bucket_range = bucket.max_val - bucket.min_val
        return q_error * (bucket_size / 1000) * (bucket_range / 1000)

    def _calculate_summary_priority(self, table, col_a, col_b, q_error):
        """Calculate priority for building conditional summaries."""
        # Higher q_error means higher priority
        # Also consider if columns are correlated
        corr_score = self.correlated_pairs.get((table, col_a, col_b), 0.0)
        return q_error * (1 + corr_score)

    def _calculate_correlation(self, table, col1, col2):
        """Calculate correlation between two columns."""
        # Implementation from StatsBuilder
        cursor = self.conn.cursor()
        try:
            cursor.execute(f'SELECT "{col1}", "{col2}" FROM "{table}" WHERE "{col1}" IS NOT NULL AND "{col2}" IS NOT NULL')
            data = cursor.fetchall()
            if len(data) < 20:
                return 0.0

            col1_data = np.array([row[0] for row in data], dtype=float)
            col2_data = np.array([row[1] for row in data], dtype=float)
            
            std1 = np.std(col1_data)
            std2 = np.std(col2_data)
            if math.isclose(std1, 0) or math.isclose(std2, 0):
                return 0.0

            corr_matrix = np.corrcoef(col1_data, col2_data)
            corr = corr_matrix[0, 1]
            return abs(corr) if not np.isnan(corr) else 0.0
        except Exception as e:
            print(f"Correlation calc warning for {table}.({col1}, {col2}): {e}")
            return 0.0

    def apply_recommendations(self, recommendations):
        """Apply the recommended statistics adjustments."""
        # Apply histogram splits
        for split in sorted(recommendations['histogram_splits'], 
                          key=lambda x: x['priority'], 
                          reverse=True):
            self.hist_catalog.split_bucket_on_value(
                table=split['table'],
                column=split['column'],
                value=split['value']
            )

        # Apply conditional summaries
        for summary in sorted(recommendations['conditional_summaries'],
                            key=lambda x: x['priority'],
                            reverse=True):
            self.cond_catalog.build_all_summaries_for(
                summary['table'],
                summary['col_a'],
                summary['col_b'],
                self.hist_catalog
            )

        # Update correlations
        for corr in recommendations['correlation_updates']:
            self.correlated_pairs[(corr['table'], corr['col_a'], corr['col_b'])] = corr['score']

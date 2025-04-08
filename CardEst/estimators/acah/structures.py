# structures.py
"""Data structures (e.g., Bucket) for ACAH Estimator V3."""

import time
import math

class Bucket:
    """Represents a single bucket in a base or conditional mini-histogram."""
    def __init__(self, min_val, max_val, count=0, distinct_count=0):
        self.min_val = min_val
        self.max_val = max_val
        self.count = max(0, count) # Ensure non-negative
        self.distinct_count = max(0, distinct_count) 

    def get_range(self):
        """Calculates the numerical range of the bucket."""
        try:
            if isinstance(self.min_val, (int, float)) and isinstance(self.max_val, (int, float)):
                # Use math.isclose for floating point comparison
                if math.isclose(self.max_val, self.min_val):
                    return 0.0
                return self.max_val - self.min_val
            else:
                return 0.0 # Non-numeric range is 0
        except TypeError:
            return 0.0

    def contains(self, value, is_last_bucket=False):
        """Checks if a numeric value falls within the bucket's range."""
        try:
            num_value = float(value)
            # Handle single point buckets using floating point comparison
            if math.isclose(self.min_val, self.max_val):
                return math.isclose(num_value, self.min_val)
            # Standard range check: inclusive min, exclusive max (unless last bucket)
            elif is_last_bucket:
                 return self.min_val <= num_value <= self.max_val
            else:
                 # Check against max_val with a small tolerance if needed, or strict <
                 return self.min_val <= num_value < self.max_val
        except (ValueError, TypeError):
            return False # Cannot compare non-numeric

    def __repr__(self):
        """Provides a string representation of the bucket."""
        min_repr = f"{self.min_val:.2f}" if isinstance(self.min_val, float) else str(self.min_val)
        max_repr = f"{self.max_val:.2f}" if isinstance(self.max_val, float) else str(self.max_val)
        return (f"Bucket[{min_repr}-{max_repr}], "
                f"Cnt:{self.count}, NDV:{self.distinct_count}")

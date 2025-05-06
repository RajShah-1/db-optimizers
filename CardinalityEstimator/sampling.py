import random

def estimate_cardinality_sampling(data, sample_size, predicate_type=None, predicate_value=None):
    """
    Estimates cardinality using random sampling.

    Args:
        data: A list or other iterable representing the dataset.
        sample_size: The number of samples to take.
        predicate_type: string of lt, gt or eq
        predicate_value: number for comparision 

    Returns:
        An estimated cardinality.
    """
    if not data:
        return 0

    n = len(data)
    if sample_size >= n:
        if predicate_type:
            filtered_data = [x for x in data if apply_predicate(x, predicate_type, predicate_value)]
            return len(set(filtered_data))
        else:
            return len(set(data))

    sample = random.sample(data, sample_size)

    if predicate_type:
        filtered_sample = [x for x in sample if apply_predicate(x, predicate_type, predicate_value)]
        sample_cardinality = len(set(filtered_sample))
    else:
        sample_cardinality = len(set(sample))

    estimated_cardinality = (sample_cardinality / sample_size) * n
    return int(estimated_cardinality)

def apply_predicate(value, predicate_type, predicate_value):
    """
    Applies a predicate to a value.

    Args:
        value: The value to compare.
        predicate_type: The type of predicate ('lt', 'gt', 'eq').
        predicate_value: The value to compare against.

    Returns:
        True if the predicate is satisfied, False otherwise.
    """
    if predicate_type == 'lt':
        return value < predicate_value
    elif predicate_type == 'gt':
        return value > predicate_value
    elif predicate_type == 'eq':
        return value == predicate_value
    else:
        return True # if no predicate, every value passes.

# Example usage:
data = [1, 2, 2, 3, 4, 4, 4, 5, 6, 7, 8, 8, 8, 8, 9, 10, 10]
sample_size = 5

# No predicate
estimated_cardinality = estimate_cardinality_sampling(data, sample_size)
real_cardinality = len(set(data))
print(f"Estimated cardinality (no predicate): {estimated_cardinality}, Real: {real_cardinality}")

# Predicate: value < 5
estimated_cardinality_lt = estimate_cardinality_sampling(data, sample_size, 'lt', 5)
real_cardinality_lt = len(set([x for x in data if apply_predicate(x, 'lt', 5)]))
print(f"Estimated cardinality (value < 5): {estimated_cardinality_lt}, Real: {real_cardinality_lt}")

# Predicate: value > 7
estimated_cardinality_gt = estimate_cardinality_sampling(data, sample_size, 'gt', 7)
real_cardinality_gt = len(set([x for x in data if apply_predicate(x, 'gt', 7)]))
print(f"Estimated cardinality (value > 7): {estimated_cardinality_gt}, Real: {real_cardinality_gt}")

# Predicate: value == 8
estimated_cardinality_eq = estimate_cardinality_sampling(data, sample_size, 'eq', 8)
real_cardinality_eq = len(set([x for x in data if apply_predicate(x, 'eq', 8)]))
print(f"Estimated cardinality (value == 8): {estimated_cardinality_eq}, Real: {real_cardinality_eq}")

data2 = [random.randint(0,1000) for _ in range(10000)]
sample_size2 = 1000

estimated_cardinality2 = estimate_cardinality_sampling(data2, sample_size2, 'gt', 500)
real_cardinality2 = len(set([x for x in data2 if apply_predicate(x, 'gt', 500)]))
print(f"Estimated cardinality for larger dataset, gt 500: {estimated_cardinality2}, Real: {real_cardinality2}")
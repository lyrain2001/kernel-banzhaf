import itertools

def get_data_and_explicand(data_size, base_data, random_state=42):
    data = base_data.sample(n=data_size + 1, random_state=random_state)
    explicand = data.iloc[[0]]
    data = data.iloc[1:]

    return data, explicand


def powerset(n):
    """Generate all combinations of 0s and 1s for a set of size n."""
    return list(itertools.product([0, 1], repeat=n))
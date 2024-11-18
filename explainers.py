import numpy as np
from sklearn.linear_model import LinearRegression
from utils import powerset


class ExactBanzhaf:
    """
    Exact Banzhaf estimator.
    """

    def __init__(self, features, imputer):
        self.features = features
        self.imputer = imputer

    def __call__(self):
        n = len(self.features)
        weight = 1 / 2 ** (n - 1)
        S = np.array(powerset(n))
        banzhaf_values = {feature: 0 for feature in self.features}

        predictions = self.imputer(S)

        for feature_idx, feature in enumerate(self.features):
            with_feature = S[:, feature_idx]
            without_feature = 1 - with_feature

            with_indices = np.where(with_feature)[0]
            without_indices = np.where(without_feature)[0]

            with_sum = np.sum(predictions[with_indices], axis=0)
            without_sum = np.sum(predictions[without_indices], axis=0)

            difference = with_sum - without_sum

            banzhaf_values[feature] = (weight * difference).tolist()

        return banzhaf_values, S, predictions


class MCBanzhaf:
    """
    Monte Carlo Banzhaf estimator.
    """

    def __init__(self, features, S_size, imputer):
        self.features = features
        self.m = S_size // len(self.features)
        self.imputer = imputer

    def __call__(self):
        n = len(self.features)
        banzhaf_values = {feature: 0 for feature in self.features}

        S = np.random.randint(2, size=(self.m, n))

        for feature_idx, feature in enumerate(self.features):
            m = self.m
            weight = 1 / m

            S_ = np.concatenate((S, S), axis=0)
            S_[:m, feature_idx] = 1
            S_[m:, feature_idx] = 0
            predictions = self.imputer(S_)

            sum_a = np.sum(predictions[:m], axis=0)
            sum_b = np.sum(predictions[m:], axis=0)

            difference = sum_a - sum_b
            banzhaf_values[feature] = (weight * difference).tolist()

        return banzhaf_values


class MSRBanzhaf:
    """
    Monte Carlo Banzhaf estimator with Maxiumum Sample Reuse.
    """

    def __init__(self, features, S_size, imputer):
        self.features = features
        self.m = S_size
        self.imputer = imputer

    def __call__(self):
        n = len(self.features)
        banzhaf_values = {feature: 0 for feature in self.features}

        S = np.random.randint(2, size=(self.m, n))
        predictions = self.imputer(S)

        for feature_idx, feature in enumerate(self.features):
            with_feature = S[:, feature_idx]
            without_feature = 1 - with_feature

            with_indices = np.where(with_feature)[0]
            without_indices = np.where(without_feature)[0]

            if len(with_indices) > 0 and len(without_indices) > 0:
                value = np.mean(predictions[with_indices], axis=0) - np.mean(
                    predictions[without_indices], axis=0
                )
                banzhaf_values[feature] = value.tolist()
            else:
                print(f"Feature {feature} has no samples in one of the subsets.")

        return banzhaf_values


class KernelBanzhaf:
    """
    Linear regression Banzhaf estimator with paired sampling.
    """

    def __init__(self, features, S_size, imputer):
        self.features = features
        self.m = S_size
        self.imputer = imputer

    def __call__(self):
        n = len(self.features)
        weight = 1 / 2
        banzhaf_values = {feature: 0 for feature in self.features}

        S_original = np.random.randint(2, size=(self.m // 2, n))
        S_complement = 1 - S_original

        # Combine original and complement samples for paired sampling
        S = np.vstack((S_original, S_complement))

        # Convert binary vectors to {-1/2, 1/2} vectors
        lr_features = S - weight
        lr_output = self.imputer(S)

        model = LinearRegression().fit(lr_features, lr_output)
        values = model.coef_.T

        for i, feature in enumerate(self.features):
            banzhaf_values[feature] = values[i].tolist()

        return banzhaf_values


class KernelBanzhafWOPS:
    """
    Linear regression Banzhaf estimator without paired sampling.
    """

    def __init__(self, features, S_size, imputer):
        self.features = features
        self.m = S_size
        self.imputer = imputer

    def __call__(self):
        n = len(self.features)
        weight = 1 / 2
        banzhaf_values = {feature: 0 for feature in self.features}

        S = np.random.randint(2, size=(self.m, n))

        # Convert binary vectors to {-1/2, 1/2} vectors
        lr_features = S - weight
        lr_output = self.imputer(S)

        model = LinearRegression().fit(lr_features, lr_output)
        values = model.coef_.T

        for i, feature in enumerate(self.features):
            banzhaf_values[feature] = values[i].tolist()

        return banzhaf_values
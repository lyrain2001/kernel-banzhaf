import numpy as np
import pandas as pd
from tqdm import tqdm
import json


class RawImputer:
    def __init__(self, data, explicand, features, model):
        self.data = data
        self.explicand = explicand
        self.features = features
        self.model = model

    def __call__(self, S, batch_size=64):
        """
        Create datasets based on each binary feature vector in matrix S,
        replace specified features with values from all data points, make predictions,
        and average these predictions using batch processing.

        Parameters:
        - S (np.array): Matrix where each row indicates which features to keep (1)
                        or replace with values from all data points (0).
        - batch_size (int): Number of configurations to process per batch.

        Returns:
        - np.array: Averaged prediction probabilities for each configuration in S.
        """
        data = self.data[self.features].to_numpy()
        explicand = self.explicand[self.features].to_numpy()

        n_rows, _ = data.shape
        n_configs = S.shape[0]
        all_predictions = []

        for start in tqdm(range(0, n_configs, batch_size)):
            end = start + batch_size
            batch_S = S[start:end]
            batch_size_actual = batch_S.shape[0]

            tiled_explicand = np.tile(explicand, (batch_size_actual * n_rows, 1))

            tiled_data = np.tile(data, (batch_size_actual, 1))

            mask = np.repeat(batch_S, n_rows, axis=0)

            X = np.where(mask, tiled_explicand, tiled_data)

            X_df = pd.DataFrame(X, columns=self.features)
            X_df = X_df.astype({f: self.data[f].dtype for f in self.features})

            raw_predictions = (
                self.model.predict(X_df)
            )
            if not isinstance(raw_predictions, np.ndarray):
                raw_predictions = raw_predictions.to_numpy()
            raw_predictions = raw_predictions.reshape(batch_size_actual, n_rows, -1)
            predictions = np.mean(raw_predictions, axis=1)

            all_predictions.extend(predictions)

        return np.array(all_predictions)


class TreeImputer:
    def __init__(self, explicand, features, model):
        """
        Initialize with a trained tree-based model (e.g., XGBoost).
        The model should have tree structures accessible as JSON.

        Parameters:
        - model: A tree-based model (like xgboost.XGBRegressor or XGBClassifier)
        """
        self.explicand = explicand.iloc[0]
        self.features = features
        self.model = model
        # Get JSON dump of all trees
        self.trees = [
            json.loads(tree)
            for tree in model.get_booster().get_dump(
                dump_format="json", with_stats=True
            )
        ]

    def __call__(self, S):
        """
        Predict the output by averaging over all trees.

        Parameters:
        - S (np.array): Matrix where each row indicates which features to keep (1)
                        or replace with values from all data points (0).
        - x (np.array): Feature values for a single instance.

        Returns:
        - float: The averaged predicted value.
        """
        feature_map = {f: i for i, f in enumerate(self.features)}

        all_predictions = []
        for s in S:
            prediction = 0
            for tree in self.trees:
                prediction += self.traverse_tree(tree, s, self.explicand, feature_map)
            all_predictions.append(prediction)
            if np.all(s):
                print(f"Prediction: {all_predictions[-1]}")
        return np.array(all_predictions)

    def is_leaf(self, node):
        """Check if the node is a leaf node"""
        return "leaf" in node

    def traverse_tree(self, node, S, x, feature_map):
        """
        Recursively traverse the tree based on feature set S and instance x.

        Parameters:
        - node (dict): The current node in the tree.
        - S (set): Set of feature indices that are used.
        - x (np.array): Feature values for a single instance.
        - feature_map (dict): Mapping of feature names to indices.

        Returns:
        - float: The calculated value from the tree.
        """
        if self.is_leaf(node):
            return float(node["leaf"])
        else:
            split_feature_name = node["split"]
            split_feature_index = feature_map[split_feature_name]
            if S[split_feature_index]:
                if float(x.iloc[split_feature_index]) < float(node["split_condition"]):
                    return self.traverse_tree(
                        node["children"][0], S, x, feature_map
                    )  # Left child
                else:
                    return self.traverse_tree(
                        node["children"][1], S, x, feature_map
                    )  # Right child
            else:
                # Feature not in S, average both branches
                left_value = self.traverse_tree(node["children"][0], S, x, feature_map)
                right_value = self.traverse_tree(node["children"][1], S, x, feature_map)
                node_weight = node.get("cover", 1)
                if node_weight is None:
                    print(f"No 'cover' found in node: {node}")
                    node_weight = 1
                left_weight = node["children"][0].get("cover", None)
                if left_weight is None:
                    print(f"No 'cover' found in left child: {node['children'][0]}")
                    left_weight = 1

                right_weight = node["children"][1].get("cover", None)
                if right_weight is None:
                    print(f"No 'cover' found in right child: {node['children'][1]}")
                    right_weight = 1

                return (
                    left_value * left_weight + right_value * right_weight
                ) / node_weight
"""A small, executable decision-tree classifier for teaching.

The implementation is intentionally direct rather than efficient. It supports
numeric features, multiclass labels, binary threshold splits, class-probability
predictions, and a few pre-pruning conditions. It does not support missing
values, categorical features, sample weights, or post-pruning.

At a node containing ``n`` examples and ``d`` features, the code tests up to
``d * (n - 1)`` thresholds. Each test scans the examples again, so finding one
split is roughly O(d * n**2). Across a balanced tree, the total remains roughly
O(d * N**2); a pathologically unbalanced tree can approach O(d * N**3).
Production libraries use sorting and caching to avoid much of this repeated
work. Prediction takes O(tree depth) per example.

The comments marked BEGIN/END SNIPPET delimit sections that can be copied into
the Quarto slides.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


# --- BEGIN SNIPPET: entropy -------------------------------------------------
def class_probabilities(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Return the fraction of examples belonging to each class."""
    return np.array([np.mean(y == label) for label in classes])


def entropy(y: np.ndarray, classes: np.ndarray) -> float:
    """Measure how mixed the classes are; zero means a pure node."""
    probabilities = class_probabilities(y, classes)
    probabilities = probabilities[probabilities > 0]
    if len(probabilities) == 1:
        return 0.0
    return float(-np.sum(probabilities * np.log2(probabilities)))
# --- END SNIPPET: entropy ---------------------------------------------------


# --- BEGIN SNIPPET: split-score ---------------------------------------------
def weighted_entropy(
    y_left: np.ndarray,
    y_right: np.ndarray,
    classes: np.ndarray,
) -> float:
    """Return the weighted entropy produced by a split."""
    n_left = len(y_left)
    n_right = len(y_right)
    n_parent = n_left + n_right

    return (
        n_left / n_parent * entropy(y_left, classes)
        + n_right / n_parent * entropy(y_right, classes)
    )


def candidate_thresholds(values: np.ndarray) -> np.ndarray:
    """Return midpoints between consecutive distinct feature values."""
    values = np.unique(values)
    return (values[:-1] + values[1:]) / 2
# --- END SNIPPET: split-score -----------------------------------------------


# --- BEGIN SNIPPET: best-split ----------------------------------------------
def find_best_split(
    X: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray,
    min_samples_leaf: int,
) -> tuple[tuple[int, float, np.ndarray] | None, int]:
    """Return the best split and the number of candidates evaluated."""
    best_split = None
    best_score = entropy(y, classes)
    n_candidates = 0

    for feature in range(X.shape[1]):
        for threshold in candidate_thresholds(X[:, feature]):
            go_left = X[:, feature] <= threshold
            n_left = np.sum(go_left)
            n_right = len(y) - n_left

            if min(n_left, n_right) < min_samples_leaf:
                continue

            n_candidates += 1
            score = weighted_entropy(y[go_left], y[~go_left], classes)
            if score < best_score:
                best_score = score
                best_split = (feature, float(threshold), go_left)

    return best_split, n_candidates
# --- END SNIPPET: best-split ------------------------------------------------


# --- BEGIN SNIPPET: node -----------------------------------------------------
@dataclass
class Node:
    """One node in the learned decision tree."""

    probabilities: np.ndarray
    n_samples: int
    loss: float
    feature: int | None = None
    threshold: float | None = None
    left: Node | None = None
    right: Node | None = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None
# --- END SNIPPET: node -------------------------------------------------------


class SimpleDecisionTreeClassifier:
    """A didactic classifier with a small scikit-learn-like interface."""

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    # --- BEGIN SNIPPET: fit --------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> SimpleDecisionTreeClassifier:
        """Learn a tree from numeric features X and class labels y."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self._validate_training_data(X, y)

        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.n_candidate_splits_ = 0
        self.tree_ = self._grow_tree(X, y, depth=0)
        return self
    # --- END SNIPPET: fit ----------------------------------------------------

    # --- BEGIN SNIPPET: grow-tree -------------------------------------------
    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """Build a tree recursively, beginning with the current node."""
        probabilities = class_probabilities(y, self.classes_)
        node = Node(
            probabilities=probabilities,
            n_samples=len(y),
            loss=entropy(y, self.classes_),
        )

        depth_limit = self.max_depth is not None and depth >= self.max_depth
        pure_node = np.count_nonzero(probabilities) == 1
        too_small = len(y) < self.min_samples_split

        if pure_node or depth_limit or too_small:
            return node

        split, n_candidates = find_best_split(
            X, y, self.classes_, self.min_samples_leaf
        )
        self.n_candidate_splits_ += n_candidates
        if split is None:
            return node

        node.feature, node.threshold, go_left = split
        node.left = self._grow_tree(X[go_left], y[go_left], depth + 1)
        node.right = self._grow_tree(X[~go_left], y[~go_left], depth + 1)
        return node
    # --- END SNIPPET: grow-tree ---------------------------------------------

    # --- BEGIN SNIPPET: predict ---------------------------------------------
    def _find_leaf(self, x: np.ndarray) -> Node:
        """Follow the learned decisions until reaching a leaf."""
        node = self.tree_
        while not node.is_leaf:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return the class proportions stored in the reached leaves."""
        X = self._validate_prediction_data(X)
        return np.vstack([self._find_leaf(x).probabilities for x in X])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the most frequent training class in each reached leaf."""
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]
    # --- END SNIPPET: predict ------------------------------------------------

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return classification accuracy."""
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def export_text(self, feature_names: list[str] | None = None) -> str:
        """Return readable if/else rules for the learned tree."""
        if feature_names is None:
            feature_names = [f"x[{j}]" for j in range(self.n_features_in_)]
        if len(feature_names) != self.n_features_in_:
            raise ValueError("feature_names must match the number of features")

        lines: list[str] = []

        def visit(node: Node, indent: str) -> None:
            if node.is_leaf:
                prediction = self.classes_[np.argmax(node.probabilities)]
                if isinstance(prediction, np.generic):
                    prediction = prediction.item()
                probabilities = np.round(node.probabilities, 3)
                lines.append(
                    f"{indent}predict {prediction!r} "
                    f"(p={probabilities}, n={node.n_samples})"
                )
                return

            name = feature_names[node.feature]
            lines.append(f"{indent}if {name} <= {node.threshold:.3f}:")
            visit(node.left, indent + "    ")
            lines.append(f"{indent}else:")
            visit(node.right, indent + "    ")

        visit(self.tree_, "")
        return "\n".join(lines)

    def _validate_training_data(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("X must be 2-D and y must be 1-D with matching rows")
        if len(y) == 0:
            raise ValueError("the training set cannot be empty")
        if not np.isfinite(X).all():
            raise ValueError("missing and non-finite feature values are unsupported")
        if self.max_depth is not None and self.max_depth < 0:
            raise ValueError("max_depth must be non-negative or None")
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1")

    def _validate_prediction_data(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X has the wrong number of features")
        if not np.isfinite(X).all():
            raise ValueError("missing and non-finite feature values are unsupported")
        return X


# --- BEGIN SNIPPET: decision-boundary ---------------------------------------
def plot_decision_boundary(
    X: np.ndarray,
    y: np.ndarray,
    model: SimpleDecisionTreeClassifier,
    feature_names: list[str],
) -> None:
    """Plot a classifier trained with exactly two numeric features."""
    x0 = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 300)
    x1 = np.linspace(X[:, 1].min() - 200, X[:, 1].max() + 200, 300)
    xx0, xx1 = np.meshgrid(x0, x1)

    grid = np.column_stack([xx0.ravel(), xx1.ravel()])
    predicted = model.predict(grid)
    regions = np.array(
        [np.flatnonzero(model.classes_ == label)[0] for label in predicted]
    ).reshape(xx0.shape)

    plt.figure(figsize=(9, 5))
    plt.contourf(xx0, xx1, regions, alpha=0.25, cmap="Set2")
    for label in model.classes_:
        selected = y == label
        plt.scatter(
            X[selected, 0],
            X[selected, 1],
            edgecolor="black",
            label=label,
        )
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title("Decision Tree Classification Regions")
    plt.legend()
    plt.tight_layout()
# --- END SNIPPET: decision-boundary -----------------------------------------


# --- BEGIN SNIPPET: penguins-example ----------------------------------------
def main() -> None:
    """Train and evaluate the classifier on Palmer Penguins."""
    from palmerpenguins import load_penguins
    from sklearn.model_selection import train_test_split

    feature_names = ["bill_depth_mm", "body_mass_g"]
    penguins = load_penguins()
    penguins = penguins[feature_names + ["species"]].dropna().copy()

    X = penguins[feature_names].to_numpy()
    y = np.where(
        penguins["species"].to_numpy() == "Gentoo",
        "Gentoo",
        "Not Gentoo",
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = SimpleDecisionTreeClassifier(max_depth=2, min_samples_leaf=5)
    model.fit(X_train, y_train)

    print(f"Test accuracy: {model.score(X_test, y_test):.3f}")
    print(f"Candidate splits evaluated: {model.n_candidate_splits_}")
    print(f"Class order: {model.classes_}")
    print("\nLearned rules:\n")
    print(model.export_text(feature_names))

    plot_decision_boundary(X_train, y_train, model, feature_names)
    plt.show()
# --- END SNIPPET: penguins-example ------------------------------------------


if __name__ == "__main__":
    main()

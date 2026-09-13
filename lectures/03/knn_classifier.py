"""A small, executable k-nearest-neighbours classifier for teaching.

The implementation deliberately favours directness over efficiency. Learning
consists of storing the training examples. For each query, the classifier
computes every Euclidean distance, sorts the distances, and lets the nearest
examples vote. It supports multiclass labels and uniform or distance-weighted
voting.

With ``N`` training examples and ``D`` features, storing a copy of the training
set takes O(ND) memory and time. A prediction computes distances in O(ND) and
uses a full O(N log N) sort. Production implementations can avoid the full sort
and may use specialized search structures.

The comments marked BEGIN/END SNIPPET delimit sections that can be copied into
the Quarto slides. The decision-boundary figure intentionally uses
scikit-learn, keeping visualization code separate from the teaching model.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


# --- BEGIN SNIPPET: neighbours ----------------------------------------------
def nearest_neighbors(
    X_train: np.ndarray,
    x: np.ndarray,
    n_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the nearest training indices and their distances from x."""
    distances = np.sqrt(np.sum((X_train - x) ** 2, axis=1))
    indices = np.argsort(distances, kind="stable")[:n_neighbors]
    return indices, distances[indices]
# --- END SNIPPET: neighbours ------------------------------------------------


# --- BEGIN SNIPPET: voting ---------------------------------------------------
def voting_weights(distances: np.ndarray, mode: str) -> np.ndarray:
    """Return uniform weights or inverse-distance weights."""
    if mode == "uniform":
        return np.ones(len(distances))

    exact_matches = distances == 0
    if np.any(exact_matches):
        return exact_matches.astype(float)

    return 1 / distances


def vote_probabilities(
    labels: np.ndarray,
    weights: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    """Convert the neighbours' weighted votes into probabilities."""
    scores = np.array([
        np.sum(weights[labels == label])
        for label in classes
    ])
    return scores / np.sum(scores)
# --- END SNIPPET: voting -----------------------------------------------------


class SimpleKNeighborsClassifier:
    """A didactic classifier with a small scikit-learn-like interface."""

    def __init__(self, n_neighbors: int = 5, weights: str = "uniform") -> None:
        self.n_neighbors = n_neighbors
        self.weights = weights

    # --- BEGIN SNIPPET: fit --------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> SimpleKNeighborsClassifier:
        """Store the numeric features and their class labels."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self._validate_training_data(X, y)

        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        return self
    # --- END SNIPPET: fit ----------------------------------------------------

    # --- BEGIN SNIPPET: predict-one -----------------------------------------
    def _predict_proba_one(self, x: np.ndarray) -> np.ndarray:
        """Predict class probabilities for one example."""
        indices, distances = nearest_neighbors(
            self.X_train_, x, self.n_neighbors
        )
        labels = self.y_train_[indices]
        weights = voting_weights(distances, self.weights)
        return vote_probabilities(labels, weights, self.classes_)
    # --- END SNIPPET: predict-one -------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for one or more examples."""
        X = self._validate_prediction_data(X)
        return np.vstack([self._predict_proba_one(x) for x in X])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the class with the largest share of the vote."""
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return classification accuracy."""
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def _validate_training_data(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("X must be 2-D and y must be 1-D with matching rows")
        if len(y) == 0:
            raise ValueError("the training set cannot be empty")
        if not np.isfinite(X).all():
            raise ValueError("missing and non-finite feature values are unsupported")
        if not isinstance(self.n_neighbors, (int, np.integer)):
            raise ValueError("n_neighbors must be an integer")
        if not 1 <= self.n_neighbors <= len(y):
            raise ValueError("n_neighbors must be between 1 and len(y)")
        if self.weights not in {"uniform", "distance"}:
            raise ValueError("weights must be 'uniform' or 'distance'")

    def _validate_prediction_data(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "X_train_"):
            raise ValueError("call fit before making predictions")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X has the wrong number of features")
        if not np.isfinite(X).all():
            raise ValueError("missing and non-finite feature values are unsupported")
        return X


# --- BEGIN SNIPPET: penguins-data -------------------------------------------
def load_penguins_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load two bill measurements and the three penguin species."""
    from palmerpenguins import load_penguins

    feature_names = ["bill_length_mm", "bill_depth_mm"]
    penguins = load_penguins()
    penguins = penguins[feature_names + ["species"]].dropna().copy()

    X = penguins[feature_names].to_numpy()
    y = penguins["species"].to_numpy()
    return X, y, feature_names
# --- END SNIPPET: penguins-data ---------------------------------------------


def plot_sklearn_boundaries(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
) -> None:
    """Compare KNN boundaries while keeping the axes in original units."""
    from sklearn.inspection import DecisionBoundaryDisplay
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    configurations = [
        (1, "uniform"),
        (5, "uniform"),
        (25, "uniform"),
        (5, "distance"),
    ]
    classes, y_encoded = np.unique(y_train, return_inverse=True)
    axis_labels = {
        "bill_length_mm": "Bill length (mm)",
        "bill_depth_mm": "Bill depth (mm)",
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax, (n_neighbors, weights) in zip(axes.ravel(), configurations):
        model = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
            ),
        )
        model.fit(X_train, y_encoded)
        DecisionBoundaryDisplay.from_estimator(
            model,
            X_train,
            response_method="predict",
            alpha=0.25,
            multiclass_colors="Set2",
            ax=ax,
        )

        for label, name in enumerate(classes):
            selected = y_encoded == label
            ax.scatter(
                X_train[selected, 0],
                X_train[selected, 1],
                edgecolor="black",
                s=24,
                color=plt.get_cmap("Set2")(label),
                label=name,
            )

        ax.set_title(f"k={n_neighbors}, weights='{weights}'")
        ax.set_xlabel(axis_labels[feature_names[0]])
        ax.set_ylabel(axis_labels[feature_names[1]])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(
        "How k and Voting Weights Change the KNN Decision Boundary",
        y=0.99,
    )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncols=len(classes),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))


# --- BEGIN SNIPPET: penguins-example ----------------------------------------
def main() -> None:
    """Train the teaching classifier and draw boundaries with scikit-learn."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X, y, feature_names = load_penguins_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # KNN distances are sensitive to feature scale. Fit the scaler only on the
    # training data, then apply the same transformation to the test data.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for mode in ("uniform", "distance"):
        model = SimpleKNeighborsClassifier(n_neighbors=5, weights=mode)
        model.fit(X_train_scaled, y_train)
        print(f"k=5, weights='{mode}': {model.score(X_test_scaled, y_test):.3f}")

    plot_sklearn_boundaries(X_train, y_train, feature_names)
    plt.show()
# --- END SNIPPET: penguins-example ------------------------------------------


if __name__ == "__main__":
    main()

"""
Single-layer Perceptron implementation from scratch (NumPy only).
"""

import numpy as np
from typing import Optional, Tuple, Dict, List


class Perceptron:
    """
    Single-layer Perceptron classifier.

    Parameters
    ----------
    learning_rate : float, default=0.01
    n_iterations : int, default=1000
    random_state : int, optional
    verbose : bool, default=False
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.verbose = verbose

        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0

        # Training history
        self.loss_history: List[float] = []
        self.weight_history: List[np.ndarray] = []
        self.bias_history: List[float] = []
        self.misclassified_history: List[int] = []

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #
    def _initialize_weights(self, n_features: int) -> None:
        rng = np.random.default_rng(self.random_state)
        self.weights = rng.standard_normal(n_features) * 0.01
        self.bias = 0.0

    # ------------------------------------------------------------------ #
    # Activations
    # ------------------------------------------------------------------ #
    @staticmethod
    def _step_activation(z: np.ndarray) -> np.ndarray:
        return (z >= 0).astype(int)

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z))

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model not trained. Call fit() first.")
        z = X @ self.weights + self.bias
        return self._step_activation(z)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model not trained. Call fit() first.")
        z = X @ self.weights + self.bias
        return self._sigmoid(z)

    # ------------------------------------------------------------------ #
    # Loss & gradients
    # ------------------------------------------------------------------ #
    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        signed_y = 2.0 * y - 1.0
        margins = signed_y * (X @ self.weights + self.bias)
        return float(np.mean(np.maximum(0.0, -margins)))

    def _compute_gradients(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        n = X.shape[0]
        signed_y = 2.0 * y - 1.0
        margins = signed_y * (X @ self.weights + self.bias)
        misclassified = margins <= 0
        grad_w = -(X[misclassified].T @ signed_y[misclassified]) / n
        grad_b = float(-np.sum(signed_y[misclassified]) / n)
        return grad_w, grad_b

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        n_samples, n_features = X.shape
        self._initialize_weights(n_features)

        # Reset history
        self.loss_history, self.weight_history = [], []
        self.bias_history, self.misclassified_history = [], []

        for it in range(self.n_iterations):
            grad_w, grad_b = self._compute_gradients(X, y)

            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

            preds = self.predict(X)
            loss = self._compute_loss(X, y)
            self.loss_history.append(loss)
            self.weight_history.append(self.weights.copy())
            self.bias_history.append(self.bias)

            mis = int(np.sum(preds != y.astype(int)))
            self.misclassified_history.append(mis)

            if self.verbose and (it % 100 == 0 or it == self.n_iterations - 1):
                acc = 1.0 - mis / n_samples
                print(f"[iter {it:4d}] loss={loss:.4f}  acc={acc:.4f}  "
                      f"misclassified={mis}")

            if mis == 0:
                if self.verbose:
                    print(f"Converged at iteration {it}.")
                break

        return self

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))

    def get_params(self) -> Dict:
        return {
            "weights": self.weights,
            "bias": self.bias,
            "learning_rate": self.learning_rate,
            "n_iterations": self.n_iterations,
        }

    def get_training_history(self) -> Dict:
        return {
            "loss": self.loss_history,
            "weights": self.weight_history,
            "bias": self.bias_history,
            "misclassified": self.misclassified_history,
        }
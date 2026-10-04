"""
Training and cross-validation helpers.
"""

import os
import sys
import numpy as np
from typing import Optional, Tuple

try:
    from .perceptron import Perceptron
    from .utils import analyze_convergence
except ImportError:
    # Support direct execution: python train.py from the src directory.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.perceptron import Perceptron
    from src.utils import analyze_convergence


def train_perceptron(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    learning_rate: float = 0.01,
    n_iterations: int = 1000,
    verbose: bool = True,
) -> Tuple[Perceptron, dict]:
    """Train a perceptron and evaluate on train/test sets."""
    model = Perceptron(
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        random_state=42,
        verbose=verbose,
    )

    print("Training Perceptron...")
    print("-" * 50)
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    print(f"Training Accuracy : {train_acc:.4f}")

    test_acc = None
    if X_test is not None and y_test is not None:
        test_acc = model.score(X_test, y_test)
        print(f"Test Accuracy     : {test_acc:.4f}")

    analysis = analyze_convergence(model.get_training_history())
    print("\nConvergence Analysis:")
    if analysis.get("converged"):
        print(f"  ✓ Converged at iteration {analysis['convergence_iteration']}")
    else:
        print("  ✗ Did not reach zero misclassifications")
    print(f"  Initial loss   : {analysis.get('initial_loss', float('nan')):.4f}")
    print(f"  Final loss     : {analysis.get('final_loss', float('nan')):.4f}")
    print(f"  Loss reduction : {analysis.get('loss_reduction', 0) * 100:.2f}%")

    metrics = {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "convergence_analysis": analysis,
        "final_weights": model.weights.copy() if model.weights is not None else None,
        "final_bias": model.bias,
    }
    return model, metrics


def cross_validate_perceptron(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    learning_rate: float = 0.01,
    n_iterations: int = 1000,
    random_state: int = 42,
) -> dict:
    """K-fold cross-validation for the Perceptron."""
    n = len(X)
    rng = np.random.default_rng(random_state)
    idx = rng.permutation(n)
    X_s, y_s = X[idx], y[idx]

    fold_size = n // n_folds
    accuracies = []

    print(f"\nPerforming {n_folds}-fold Cross-Validation...")
    print("-" * 50)

    for k in range(n_folds):
        start = k * fold_size
        end = start + fold_size if k < n_folds - 1 else n

        X_val, y_val = X_s[start:end], y_s[start:end]
        X_tr = np.vstack([X_s[:start], X_s[end:]])
        y_tr = np.hstack([y_s[:start], y_s[end:]])

        model = Perceptron(
            learning_rate=learning_rate,
            n_iterations=n_iterations,
            random_state=random_state,
            verbose=False,
        )
        model.fit(X_tr, y_tr)
        acc = model.score(X_val, y_val)
        accuracies.append(acc)
        print(f"  Fold {k + 1}: accuracy = {acc:.4f}")

    accuracies = np.array(accuracies)
    print("-" * 50)
    print(f"  Mean accuracy : {accuracies.mean():.4f} "
          f"(+/- {accuracies.std():.4f})")

    return {
        "accuracies": accuracies.tolist(),
        "mean_accuracy": float(accuracies.mean()),
        "std_accuracy": float(accuracies.std()),
    }
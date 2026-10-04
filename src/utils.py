"""
Visualization and analysis utilities.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe headless default; show() still works interactively
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Optional, Dict


CMAP_LIGHT = ListedColormap(["#FFAAAA", "#AAAAFF"])
CMAP_POINTS = ListedColormap(["#FF0000", "#0000FF"])


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def plot_decision_boundary(
    perceptron,
    X: np.ndarray,
    y: np.ndarray,
    title: str = "Perceptron Decision Boundary",
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """Plot decision regions, boundary line, and data points."""
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = perceptron.predict(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=CMAP_LIGHT)

    if perceptron.weights is not None:
        w1, w2 = perceptron.weights
        b = perceptron.bias
        if abs(w2) > 1e-10:
            x_line = np.array([x_min, x_max])
            y_line = -(w1 * x_line + b) / w2
            ax.plot(x_line, y_line, "k--", linewidth=2, label="Decision Boundary")

    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=CMAP_POINTS,
               edgecolor="k", s=45, alpha=0.85)

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(
    history: Dict,
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """4-panel plot: loss, misclassifications, weight evolution, bias evolution."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].plot(history["loss"], color="tab:blue", linewidth=2)
    axes[0, 0].set_title("Loss vs Iteration", fontweight="bold")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Binary Cross-Entropy")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history["misclassified"], color="tab:red", linewidth=2)
    axes[0, 1].set_title("Misclassifications vs Iteration", fontweight="bold")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("# Misclassified")
    axes[0, 1].grid(True, alpha=0.3)

    if history["weights"]:
        w = np.array(history["weights"])
        axes[1, 0].plot(w[:, 0], label="w1", linewidth=2)
        axes[1, 0].plot(w[:, 1], label="w2", linewidth=2)
        axes[1, 0].set_title("Weight Evolution", fontweight="bold")
        axes[1, 0].set_xlabel("Iteration")
        axes[1, 0].set_ylabel("Weight Value")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    if history["bias"]:
        axes[1, 1].plot(history["bias"], color="tab:green", linewidth=2)
        axes[1, 1].set_title("Bias Evolution", fontweight="bold")
        axes[1, 1].set_xlabel("Iteration")
        axes[1, 1].set_ylabel("Bias Value")
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_training_animation(
    history: Dict,
    X: np.ndarray,
    y: np.ndarray,
    save_path: Optional[str] = None,
    step: int = 10,
    show: bool = False,
) -> None:
    """Static overlay of decision boundaries at multiple training iterations."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=CMAP_POINTS,
               edgecolor="k", s=45, alpha=0.85)

    if not history["weights"]:
        plt.close(fig)
        return

    n_show = min(10, max(1, len(history["weights"]) // max(1, step)))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_show))
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5

    for i, idx in enumerate(range(0, len(history["weights"]), max(1, step))):
        if i >= n_show:
            break
        w = history["weights"][idx]
        b = history["bias"][idx]
        if abs(w[1]) > 1e-10:
            x_line = np.array([x_min, x_max])
            y_line = -(w[0] * x_line + b) / w[1]
            ax.plot(x_line, y_line, color=colors[i], linewidth=1.8,
                    alpha=0.75, label=f"iter {idx}")

    w = history["weights"][-1]
    b = history["bias"][-1]
    if abs(w[1]) > 1e-10:
        x_line = np.array([x_min, x_max])
        y_line = -(w[0] * x_line + b) / w[1]
        ax.plot(x_line, y_line, "k-", linewidth=3, label="final")

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Decision Boundary Evolution", fontweight="bold")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def analyze_convergence(history: Dict) -> Dict:
    """Return convergence diagnostics from training history."""
    out: Dict = {}

    mis = np.asarray(history.get("misclassified", []), dtype=int)
    zeros = np.where(mis == 0)[0]
    out["converged"] = len(zeros) > 0
    out["convergence_iteration"] = int(zeros[0]) if len(zeros) else None

    loss = history.get("loss", [])
    if loss:
        out["initial_loss"] = float(loss[0])
        out["final_loss"] = float(loss[-1])
        denom = loss[0] if loss[0] != 0 else 1.0
        out["loss_reduction"] = float((loss[0] - loss[-1]) / denom)

    if len(loss) > 100:
        out["loss_stability"] = float(np.std(loss[-100:]))

    return out
"""
Generate linearly separable (and non-separable) binary datasets
for perceptron training and analysis.

Supported dataset types:
    - 'and'      : quadrant-based AND gate (linearly separable)
    - 'or'       : quadrant-based OR gate  (linearly separable)
    - 'nand'     : quadrant-based NAND gate (linearly separable)
    - 'nor'      : quadrant-based NOR gate  (linearly separable)
    - 'custom'   : two Gaussian blobs (linearly separable)
    - 'moons'    : two interleaving half-moons (NOT linearly separable)
    - 'circles'  : concentric circles (NOT linearly separable)
    - 'xor'      : XOR pattern (NOT linearly separable)

Every generator returns a tuple (X, y) where:
    X : np.ndarray of shape (n_samples, 2), dtype=float64
    y : np.ndarray of shape (n_samples,), dtype=int  -> values in {0, 1}
"""

from __future__ import annotations

from typing import Optional, Tuple, List

import numpy as np


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
SUPPORTED_DATASETS: List[str] = [
    "and", "or", "nand", "nor",
    "custom",
    "xor", "moons", "circles",
]

LINEARLY_SEPARABLE: List[str] = ["and", "or", "nand", "nor", "custom"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _validate_inputs(n_samples: int, noise: float) -> None:
    if not isinstance(n_samples, (int, np.integer)) or n_samples < 2:
        raise ValueError("n_samples must be an integer >= 2.")
    if not isinstance(noise, (int, float, np.floating)) or noise < 0:
        raise ValueError("noise must be a non-negative number.")


def _balance_split(n_samples: int) -> Tuple[int, int]:
    """Return (n_class0, n_class1) that sums to n_samples."""
    n0 = n_samples // 2
    n1 = n_samples - n0
    return n0, n1


def _quadrant_gate(
    n_samples: int,
    gate: str,
    noise: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample noisy binary feature values and label them with a logic gate.

    Sampling arbitrary quadrants does not produce a linearly separable AND or
    OR dataset.  Points around the four binary corners preserve the intended
    truth-table interpretation and leave a margin for the perceptron.
    """
    bits = rng.integers(0, 2, size=(n_samples, 2))
    s1 = bits[:, 0].astype(bool)
    s2 = bits[:, 1].astype(bool)

    if gate == "and":
        y = (s1 & s2)
    elif gate == "or":
        y = (s1 | s2)
    elif gate == "nand":
        y = ~(s1 & s2)
    elif gate == "nor":
        y = ~(s1 | s2)
    else:
        raise ValueError(f"Unknown gate: {gate}")

    y = y.astype(int)

    # Keep jitter below 0.25 so the four clusters remain separated by the
    # corresponding gate boundary, even when the caller requests more noise.
    jitter = min(float(noise), 0.24)
    X = bits.astype(float)
    if jitter > 0:
        X += rng.uniform(-jitter, jitter, size=X.shape)
        X = np.clip(X, 0.0, 1.0)

    return X, y


def _two_gaussian_blobs(
    n_samples: int,
    noise: float,
    rng: np.random.Generator,
    center0: Tuple[float, float] = (-1.0, -1.0),
    center1: Tuple[float, float] = (1.0, 1.0),
    spread: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Two Gaussian blobs, one per class. Linearly separable in expectation."""
    n0, n1 = _balance_split(n_samples)
    X0 = rng.normal(loc=center0, scale=spread, size=(n0, 2))
    X1 = rng.normal(loc=center1, scale=spread, size=(n1, 2))
    y0 = np.zeros(n0, dtype=int)
    y1 = np.ones(n1, dtype=int)

    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])

    if noise > 0:
        X = X + rng.normal(0.0, noise, size=X.shape)

    idx = rng.permutation(n_samples)
    return X[idx], y[idx]


def _moons(
    n_samples: int,
    noise: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Two interleaving half-circles (sklearn-style, implemented manually)."""
    n0, n1 = _balance_split(n_samples)

    # Outer moon (class 0)
    t0 = np.linspace(0, np.pi, n0)
    X0 = np.column_stack([np.cos(t0), np.sin(t0)])

    # Inner moon (class 1)
    t1 = np.linspace(0, np.pi, n1)
    X1 = np.column_stack([1 - np.cos(t1), 1 - np.sin(t1) - 0.5])

    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])

    if noise > 0:
        X = X + rng.normal(0.0, noise, size=X.shape)

    idx = rng.permutation(n_samples)
    return X[idx], y[idx]


def _circles(
    n_samples: int,
    noise: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Concentric circles (NOT linearly separable)."""
    n0, n1 = _balance_split(n_samples)

    # Inner circle
    t0 = rng.uniform(0, 2 * np.pi, size=n0)
    r0 = rng.uniform(0.0, 0.4, size=n0)
    X0 = np.column_stack([r0 * np.cos(t0), r0 * np.sin(t0)])

    # Outer ring
    t1 = rng.uniform(0, 2 * np.pi, size=n1)
    r1 = rng.uniform(0.8, 1.0, size=n1)
    X1 = np.column_stack([r1 * np.cos(t1), r1 * np.sin(t1)])

    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])

    if noise > 0:
        X = X + rng.normal(0.0, noise, size=X.shape)

    idx = rng.permutation(n_samples)
    return X[idx], y[idx]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def generate_binary_data(
    dataset_type: str = "and",
    n_samples: int = 100,
    noise: float = 0.1,
    random_state: Optional[int] = 42,
    *,
    shuffle: bool = True,
    return_meta: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, dict]:
    """
    Generate a binary classification dataset.

    Parameters
    ----------
    dataset_type : str
        One of SUPPORTED_DATASETS.
    n_samples : int
        Total number of samples.
    noise : float
        Standard deviation of additive Gaussian noise.
    random_state : int or None
        Seed for reproducibility.
    shuffle : bool
        Whether to shuffle the rows before returning.
    return_meta : bool
        If True, also return a dict with dataset metadata.

    Returns
    -------
    X : np.ndarray of shape (n_samples, 2)
    y : np.ndarray of shape (n_samples,)   dtype=int, values in {0, 1}
    meta : dict (only if return_meta=True)
    """
    _validate_inputs(n_samples, noise)

    dtype = (dataset_type or "").lower().strip()
    if dtype not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unknown dataset_type={dataset_type!r}. "
            f"Choose from {SUPPORTED_DATASETS}."
        )

    rng = np.random.default_rng(random_state)

    # ---- Generate --------------------------------------------------------- #
    if dtype in ("and", "or", "nand", "nor"):
        X, y = _quadrant_gate(n_samples, dtype, noise, rng)

    elif dtype == "custom":
        X, y = _two_gaussian_blobs(n_samples, noise, rng)

    elif dtype == "xor":
        # Random uniform XOR pattern in [-1, 1]^2
        X = rng.uniform(-1.0, 1.0, size=(n_samples, 2))
        y = ((X[:, 0] >= 0) ^ (X[:, 1] >= 0)).astype(int)
        if noise > 0:
            X = X + rng.normal(0.0, noise, size=X.shape)

    elif dtype == "moons":
        X, y = _moons(n_samples, noise, rng)

    elif dtype == "circles":
        X, y = _circles(n_samples, noise, rng)

    else:  # pragma: no cover (guarded above)
        raise RuntimeError(f"Unhandled dataset_type={dtype!r}")

    # ---- Shuffle ---------------------------------------------------------- #
    if shuffle:
        idx = rng.permutation(n_samples)
        X, y = X[idx], y[idx]

    # ---- Sanity check ----------------------------------------------------- #
    assert X.shape == (n_samples, 2), f"Bad X shape: {X.shape}"
    assert y.shape == (n_samples,), f"Bad y shape: {y.shape}"
    assert set(np.unique(y)).issubset({0, 1}), "y must be binary {0,1}"

    X = X.astype(np.float64, copy=False)
    y = y.astype(np.int64, copy=False)

    if return_meta:
        meta = {
            "dataset_type": dtype,
            "n_samples": int(n_samples),
            "noise": float(noise),
            "random_state": random_state,
            "linearly_separable": dtype in LINEARLY_SEPARABLE,
            "class_balance": np.bincount(y, minlength=2).tolist(),
        }
        return X, y, meta

    return X, y


def generate_xor_data(
    n_samples: int = 100,
    noise: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backwards-compatible XOR generator. Equivalent to
    generate_binary_data('xor', n_samples, noise, random_state).
    """
    return generate_binary_data(
        "xor", n_samples=n_samples, noise=noise, random_state=random_state
    )


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    *,
    stratify: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple train/test split without sklearn.

    Parameters
    ----------
    test_size : float in [0, 1)
    stratify : bool
        If True, preserve class proportions in train and test.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    if not 0.0 <= test_size < 1.0:
        raise ValueError("test_size must be in [0, 1).")
    if len(X) != len(y):
        raise ValueError("X and y must have the same length.")

    rng = np.random.default_rng(random_state)
    n = len(X)

    if stratify:
        train_idx: List[int] = []
        test_idx: List[int] = []
        for cls in np.unique(y):
            cls_idx = np.where(y == cls)[0]
            cls_idx = rng.permutation(cls_idx)
            n_test_cls = int(round(len(cls_idx) * test_size))
            test_idx.extend(cls_idx[:n_test_cls].tolist())
            train_idx.extend(cls_idx[n_test_cls:].tolist())
        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)
    else:
        idx = rng.permutation(n)
        n_test = int(round(n * test_size))
        test_idx, train_idx = idx[:n_test], idx[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def make_linearly_separable(
    X: np.ndarray,
    y: np.ndarray,
    margin: float = 0.3,
    direction: Optional[np.ndarray] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Warp a 2D dataset so that the two classes become strictly linearly
    separable along `direction` with a given margin.

    Useful for turning 'moons' / 'circles' into separable variants for
    controlled experiments.
    """
    if X.shape[1] != 2:
        raise ValueError("make_linearly_separable expects 2D features.")

    rng = np.random.default_rng(random_state)
    if direction is None:
        direction = rng.standard_normal(2)
    direction = direction / (np.linalg.norm(direction) + 1e-12)

    proj = X @ direction
    # Shift each class along the chosen direction so classes separate
    shift = np.where(y == 1, +margin, -margin)
    # Remove mean of projection then push classes apart
    X_new = X + (shift - proj)[:, None] * direction
    return X_new.astype(np.float64), y.astype(np.int64)


def describe_dataset(X: np.ndarray, y: np.ndarray) -> dict:
    """Return summary statistics useful for reports and logging."""
    return {
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "class_counts": np.bincount(y, minlength=2).tolist(),
        "class_balance": (np.bincount(y, minlength=2) / len(y)).round(4).tolist(),
        "feature_means": X.mean(axis=0).round(4).tolist(),
        "feature_stds": X.std(axis=0).round(4).tolist(),
        "feature_min": X.min(axis=0).round(4).tolist(),
        "feature_max": X.max(axis=0).round(4).tolist(),
    }


# --------------------------------------------------------------------------- #
# CLI demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("Available datasets:", ", ".join(SUPPORTED_DATASETS))
    print()
    for kind in SUPPORTED_DATASETS:
        X, y, meta = generate_binary_data(kind, n_samples=200, noise=0.1,
                                          return_meta=True)
        stats = describe_dataset(X, y)
        print(f"[{kind:8s}]  X={X.shape}  y={y.shape}  "
              f"counts={stats['class_counts']}  "
              f"separable={meta['linearly_separable']}")
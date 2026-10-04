"""
Perceptron classifier package.
"""

from .perceptron import Perceptron
from .train import train_perceptron, cross_validate_perceptron
from .utils import (
    plot_decision_boundary,
    plot_convergence,
    plot_training_animation,
    analyze_convergence,
)

__all__ = [
    "Perceptron",
    "train_perceptron",
    "cross_validate_perceptron",
    "plot_decision_boundary",
    "plot_convergence",
    "plot_training_animation",
    "analyze_convergence",
]
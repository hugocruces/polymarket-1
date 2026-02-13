"""
Bias detection module for identifying markets affected by Polymarket demographic biases.
"""

from .classifier import classify_market
from .models import BiasCategory, BiasClassification

__all__ = ["BiasCategory", "BiasClassification", "classify_market"]

"""
Model architectures for SeqofSeq Pipeline
"""

from .conditional_sequence_generator import ConditionalSequenceGenerator
from .conditional_duration_predictor import ConditionalDurationPredictor

__all__ = [
    'ConditionalSequenceGenerator',
    'ConditionalDurationPredictor'
]

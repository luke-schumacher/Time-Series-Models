"""
Training modules for SeqofSeq Pipeline
"""

from .train_sequence_model import train_sequence_model
from .train_duration_model import train_duration_model

__all__ = [
    'train_sequence_model',
    'train_duration_model'
]

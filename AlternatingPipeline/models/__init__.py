"""
Models for the Alternating Pipeline.

Exchange Model: Predicts body region transitions
Examination Model: Generates event sequences for a body region
"""
from .layers import PositionalEncoding, create_attention_mask, create_key_padding_mask
from .exchange_model import ExchangeModel, create_exchange_model
from .examination_model import ExaminationModel, create_examination_model

__all__ = [
    'PositionalEncoding',
    'create_attention_mask',
    'create_key_padding_mask',
    'ExchangeModel',
    'create_exchange_model',
    'ExaminationModel',
    'create_examination_model',
]

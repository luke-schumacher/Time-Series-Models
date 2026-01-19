"""
Generation modules for the Alternating Pipeline.

- bucket_generator: Pre-generate samples for each body region transition
- day_simulator: Simulate a full day by sampling from buckets
- orchestrator: Main entry point for generation
"""
from .bucket_generator import BucketGenerator
from .day_simulator import DaySimulator

__all__ = [
    'BucketGenerator',
    'DaySimulator',
]

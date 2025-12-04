"""
Mean-Variance Optimal Delta-Hedging Package
===========================================
HKUST IEDA3330 Introduction to Financial Engineering - Fall 2025
Prof. Wei JIANG

Main package for short strangle delta-hedging analysis.

THREE METHODOLOGIES (implemented in separate files):
- M1: Option Strategy and Delta-Hedging (Lecture 6) - _1_Option_Delta.py
- M2: Dynamic Covariance Estimation via EWMA (Lecture 7) - _2_Covariance_Estimation.py
- M3: Mean-Variance Optimal Portfolio Construction (Lecture 5) - _3_MV_Optimization.py

The backtest compares all three methodologies head-to-head.
"""

__version__ = "1.0.0"
__author__ = "HKUST Financial Engineering Students"

# Core modules (used by backtest)
from . import data
from . import returns
from . import backtest    # Runs M1, M2, M3 comparison
from . import plots
from . import robo_advisor

# Methodology modules (M1, M2, M3)
from . import _1_Option_Delta
from . import _2_Covariance_Estimation
from . import _3_MV_Optimization

__all__ = [
    'data',
    'returns',
    'backtest',
    'plots',
    'robo_advisor',
    '_1_Option_Delta',
    '_2_Covariance_Estimation',
    '_3_MV_Optimization'
]


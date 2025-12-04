"""
Mean-Variance Optimal Delta-Hedging Package
===========================================
HKUST IEDA3330 Introduction to Financial Engineering - Fall 2025
Prof. Wei JIANG

Main package for short strangle delta-hedging analysis.

THREE METHODOLOGIES (compared in backtest):
- M1: Option Strategy and Delta-Hedging (Lecture 6) - covariance.py
- M2: Dynamic Covariance Estimation via EWMA (Lecture 7) - covariance.py
- M3: Mean-Variance Optimal Portfolio Construction (Lecture 5) - optimize.py

Standalone methodology demonstrations:
- _1_Option_Delta.py
- _2_Covariance_Estimation.py
- _3_MV_Optimization.py
"""

__version__ = "1.0.0"
__author__ = "HKUST Financial Engineering Students"

# Core modules (used by backtest)
from . import data
from . import returns
from . import covariance  # M2: EWMA estimation
from . import optimize    # M3: MV optimization
from . import backtest    # Runs M1, M2, M3 comparison
from . import plots
from . import robo_advisor

__all__ = [
    'data',
    'returns',
    'covariance',
    'optimize',
    'backtest',
    'plots',
    'robo_advisor'
]


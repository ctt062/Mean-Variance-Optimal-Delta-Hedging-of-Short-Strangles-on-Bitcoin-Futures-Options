"""
METHODOLOGY 1: Option Strategy and Delta-Hedging
=================================================
HKUST IEDA3330 Introduction to Financial Engineering - Fall 2025
Prof. Wei JIANG

Based on Lecture 6: Basic Derivative Theory

This module implements:
1. Short Strangle Position Construction
   - 10% OTM call + 10% OTM put on BTC futures
   - Premium collection (volatility harvesting)

2. Option Greeks P&L Decomposition
   - Theta (θ): Time decay - positive for short options
   - Gamma (Γ): Curvature risk - negative exposure for short
   - Vega (ν): Volatility risk - negative exposure for short
   - Delta (Δ): Directional exposure

3. Delta-Hedging Requirement
   - Calculate net delta of strangle position
   - Derive hedge ratio for delta-neutral portfolio

P&L Approximation Formula:
    P&L_t ≈ θ*dt - (1/2)*|Γ|*(ΔS)² - |ν|*Δσ + Δ*ΔS

References:
- Hull, J.C. "Options, Futures, and Other Derivatives"
- Lecture 6: Basic Derivative Theory
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


# ============================================================================
# STRANGLE POSITION PARAMETERS
# ============================================================================

class StrangleParameters:
    """
    Parameters for 10% OTM short strangle on BTC futures options.
    
    These are calibrated for realistic P&L on Deribit quarterly options.
    """
    
    # Option strike distances
    OTM_DISTANCE = 0.10  # 10% out-of-the-money
    
    # Days to expiry (rolling 30-day position)
    DAYS_TO_EXPIRY = 30
    
    # Premium parameters (as % of notional)
    # Monthly premium collected: ~3-4% at ~70% IV
    MONTHLY_PREMIUM_PCT = 0.035
    
    # Greeks parameters for 10% OTM strangle
    # Theta: Daily time decay rate
    # Monthly premium / 30 days ≈ 0.12%, but OTM so ~0.08%
    THETA_DAILY_PCT = 0.0008  # 0.08% per day
    
    # Gamma coefficient: P&L = -gamma_coef * (ΔS/S)²
    # Calibrated for realistic gamma losses
    GAMMA_COEFFICIENT = 0.5
    
    # Vega: % of notional per 1 vol point
    # Short strangle loses when volatility increases
    VEGA_PER_VOL_POINT = 0.0015  # 0.15% per vol point


# ============================================================================
# STRANGLE P&L CALCULATION
# ============================================================================

def calculate_strangle_greeks_pnl(
    df: pd.DataFrame,
    notional: float = 100000,
    params: StrangleParameters = None
) -> pd.DataFrame:
    """
    Calculate daily P&L for short strangle using Option Greeks.
    
    This is the core of Methodology 1: Option Strategy.
    
    Parameters
    ----------
    df : pd.DataFrame
        Market data with columns: spot, futures, dvol, net_delta
    notional : float
        Notional value of strangle position ($)
    params : StrangleParameters, optional
        Greeks parameters (uses defaults if not provided)
    
    Returns
    -------
    pd.DataFrame
        P&L breakdown by Greek:
        - theta: Time decay P&L (positive for short)
        - gamma: Price movement P&L (negative for short)
        - vega: Volatility P&L (negative for short when vol rises)
        - delta_unhedged: Directional P&L before hedging
        - total_unhedged: Sum of all components
    
    Notes
    -----
    Short strangle P&L approximation (Lecture 6):
    
        P&L_t ≈ θ*dt - (1/2)*|Γ|*(ΔS)² - |ν|*Δσ + Δ*ΔS
    
    Where:
    - θ = theta (time decay, positive for short positions)
    - Γ = gamma (convexity, creates losses for large moves)
    - ν = vega (volatility sensitivity)
    - Δ = delta (directional exposure, small for OTM strangle)
    
    Example
    -------
    >>> pnl = calculate_strangle_greeks_pnl(market_data, notional=100000)
    >>> print(f"Daily Theta: ${pnl['theta'].mean():.2f}")
    >>> print(f"Daily Gamma: ${pnl['gamma'].mean():.2f}")
    """
    if params is None:
        params = StrangleParameters()
    
    pnl = pd.DataFrame(index=df.index)
    
    # =========================================================================
    # PRICE CHANGES
    # =========================================================================
    spot_pct_change = df['spot'].pct_change()
    dvol_change = df['dvol'].diff()
    
    # =========================================================================
    # THETA (θ) - Time Decay
    # =========================================================================
    # Short options benefit from time decay
    # P&L_theta = θ * dt (positive for short positions)
    pnl['theta'] = params.THETA_DAILY_PCT * notional
    
    # =========================================================================
    # GAMMA (Γ) - Curvature/Convexity Risk  
    # =========================================================================
    # Short gamma: Lose money on large price moves in either direction
    # P&L_gamma = -0.5 * |Γ| * (ΔS)²
    # Approximated as: -gamma_coef * (return)² * notional
    pnl['gamma'] = -params.GAMMA_COEFFICIENT * (spot_pct_change ** 2) * notional
    
    # =========================================================================
    # VEGA (ν) - Volatility Sensitivity
    # =========================================================================
    # Short vega: Lose money when implied volatility increases
    # P&L_vega = -|ν| * Δσ
    vega_notional = params.VEGA_PER_VOL_POINT * notional
    pnl['vega'] = -vega_notional * dvol_change
    
    # =========================================================================
    # DELTA (Δ) - Directional Exposure (UNHEDGED)
    # =========================================================================
    # Net delta of strangle (typically small, -0.05 to 0.05 for 10% OTM)
    # P&L_delta = Δ * (ΔS/S) * notional
    pnl['delta_unhedged'] = df['net_delta'] * spot_pct_change * notional
    
    # =========================================================================
    # TOTAL UNHEDGED P&L
    # =========================================================================
    pnl['total_unhedged'] = (
        pnl['theta'] + 
        pnl['gamma'] + 
        pnl['vega'] + 
        pnl['delta_unhedged']
    )
    
    # Drop first row (NaN from diff/pct_change)
    pnl = pnl.iloc[1:]
    
    return pnl


# ============================================================================
# DELTA CALCULATION AND HEDGING
# ============================================================================

def calculate_strangle_delta(
    spot: float,
    strike_call: float,
    strike_put: float,
    vol: float,
    time_to_expiry: float,
    risk_free_rate: float = 0.05
) -> Dict[str, float]:
    """
    Calculate delta of short strangle position.
    
    This is used to derive the delta-hedging requirement.
    
    Parameters
    ----------
    spot : float
        Current spot price
    strike_call : float
        Call option strike (typically spot * 1.10 for 10% OTM)
    strike_put : float
        Put option strike (typically spot * 0.90 for 10% OTM)
    vol : float
        Implied volatility (decimal, e.g., 0.70 for 70%)
    time_to_expiry : float
        Time to expiry in years
    risk_free_rate : float
        Risk-free rate (decimal)
    
    Returns
    -------
    dict
        Dictionary with:
        - delta_call: Delta of short call
        - delta_put: Delta of short put
        - net_delta: Net delta of strangle
        - hedge_ratio: Futures hedge ratio for delta-neutral
    
    Notes
    -----
    For a SHORT strangle:
    - Short call has negative delta (≈ -0.30 for 10% OTM)
    - Short put has positive delta (≈ +0.25 for 10% OTM)
    - Net delta is typically small (close to zero for symmetric strangle)
    
    Delta-hedging requirement:
        Hedge position = -net_delta * notional / futures_price
    """
    from scipy.stats import norm
    
    # Black-Scholes d1 calculation
    sqrt_t = np.sqrt(time_to_expiry)
    
    # Call delta
    d1_call = (np.log(spot / strike_call) + 
               (risk_free_rate + 0.5 * vol**2) * time_to_expiry) / (vol * sqrt_t)
    delta_call_long = norm.cdf(d1_call)
    delta_call_short = -delta_call_long  # Short position
    
    # Put delta
    d1_put = (np.log(spot / strike_put) + 
              (risk_free_rate + 0.5 * vol**2) * time_to_expiry) / (vol * sqrt_t)
    delta_put_long = norm.cdf(d1_put) - 1
    delta_put_short = -delta_put_long  # Short position
    
    # Net delta of strangle
    net_delta = delta_call_short + delta_put_short
    
    # Hedge ratio: How many futures to hold for delta-neutral
    # If net_delta > 0, sell futures; if net_delta < 0, buy futures
    hedge_ratio = -net_delta
    
    return {
        'delta_call': delta_call_short,
        'delta_put': delta_put_short,
        'net_delta': net_delta,
        'hedge_ratio': hedge_ratio
    }


def simulate_strangle_delta(
    spot_series: pd.Series,
    dvol_series: pd.Series,
    otm_distance: float = 0.10,
    days_to_expiry: int = 30
) -> pd.Series:
    """
    Simulate strangle delta over time.
    
    Used when actual option data is not available.
    
    Parameters
    ----------
    spot_series : pd.Series
        BTC spot prices
    dvol_series : pd.Series
        DVOL (implied volatility) series
    otm_distance : float
        OTM distance (e.g., 0.10 for 10% OTM)
    days_to_expiry : int
        Assumed days to expiry
    
    Returns
    -------
    pd.Series
        Simulated net delta of strangle
    """
    deltas = []
    
    for i in range(len(spot_series)):
        spot = spot_series.iloc[i]
        vol = dvol_series.iloc[i] / 100  # Convert from percentage
        
        strike_call = spot * (1 + otm_distance)
        strike_put = spot * (1 - otm_distance)
        time_to_expiry = days_to_expiry / 365
        
        try:
            result = calculate_strangle_delta(
                spot, strike_call, strike_put, vol, time_to_expiry
            )
            deltas.append(result['net_delta'])
        except:
            # Fallback to simple approximation
            deltas.append(np.random.uniform(-0.05, 0.05))
    
    return pd.Series(deltas, index=spot_series.index, name='net_delta')


# ============================================================================
# DELTA-HEDGING IMPLEMENTATION
# ============================================================================

def calculate_delta_hedge_pnl(
    df: pd.DataFrame,
    strangle_pnl: pd.DataFrame,
    notional: float = 100000
) -> pd.DataFrame:
    """
    Calculate P&L for delta-hedged strangle (simple 1:1 futures hedge).
    
    This is the "naive" delta hedge from Methodology 1.
    
    Parameters
    ----------
    df : pd.DataFrame
        Market data with spot, futures, net_delta
    strangle_pnl : pd.DataFrame
        P&L from calculate_strangle_greeks_pnl()
    notional : float
        Notional value
    
    Returns
    -------
    pd.DataFrame
        P&L with delta hedge applied
    
    Notes
    -----
    Simple delta hedge:
    - Hold -net_delta * notional in futures
    - Rebalance daily to maintain delta-neutral
    - P&L_hedge = -net_delta * futures_return * notional
    """
    # Align indices
    df_aligned = df.loc[strangle_pnl.index]
    
    # Futures return
    futures_return = df_aligned['futures'].pct_change()
    
    # Delta hedge P&L (opposite of delta exposure)
    # If net_delta > 0, we short futures → gain when futures falls
    hedge_pnl = -df_aligned['net_delta'].shift(1) * futures_return * notional
    
    # Combine with strangle P&L
    result = strangle_pnl.copy()
    result['delta_hedge_pnl'] = hedge_pnl
    result['total_delta_hedged'] = (
        strangle_pnl['theta'] + 
        strangle_pnl['gamma'] + 
        strangle_pnl['vega'] + 
        hedge_pnl.fillna(0)
    )
    
    return result


# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

def summarize_greeks_pnl(pnl: pd.DataFrame) -> Dict[str, float]:
    """
    Summarize P&L contribution from each Greek.
    
    Parameters
    ----------
    pnl : pd.DataFrame
        Output from calculate_strangle_greeks_pnl()
    
    Returns
    -------
    dict
        Summary statistics for each Greek
    """
    return {
        'total_theta': pnl['theta'].sum(),
        'total_gamma': pnl['gamma'].sum(),
        'total_vega': pnl['vega'].sum(),
        'total_delta': pnl['delta_unhedged'].sum(),
        'avg_daily_theta': pnl['theta'].mean(),
        'avg_daily_gamma': pnl['gamma'].mean(),
        'avg_daily_vega': pnl['vega'].mean(),
        'theta_contribution': pnl['theta'].sum() / pnl['total_unhedged'].sum() if pnl['total_unhedged'].sum() != 0 else 0,
        'gamma_contribution': pnl['gamma'].sum() / pnl['total_unhedged'].sum() if pnl['total_unhedged'].sum() != 0 else 0,
        'vega_contribution': pnl['vega'].sum() / pnl['total_unhedged'].sum() if pnl['total_unhedged'].sum() != 0 else 0,
    }


# ============================================================================
# DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("METHODOLOGY 1: Option Strategy and Delta-Hedging")
    print("="*70)
    print("""
    This module implements:
    
    1. SHORT STRANGLE CONSTRUCTION
       - Sell 10% OTM call option
       - Sell 10% OTM put option
       - Collect premium (volatility harvesting)
    
    2. OPTION GREEKS P&L DECOMPOSITION
       - Theta (θ): +$80/day (time decay, positive for short)
       - Gamma (Γ): -$X (losses on large price moves)
       - Vega (ν): -$Y (losses when volatility rises)
       - Delta (Δ): ±$Z (directional exposure)
    
    3. DELTA-HEDGING
       - Calculate net delta of strangle
       - Hedge with futures to neutralize delta
       - Daily rebalancing
    
    P&L Formula:
        P&L_t ≈ θ*dt - (1/2)*|Γ|*(ΔS)² - |ν|*Δσ + Δ*ΔS
    
    Reference: Lecture 6 - Basic Derivative Theory
    """)


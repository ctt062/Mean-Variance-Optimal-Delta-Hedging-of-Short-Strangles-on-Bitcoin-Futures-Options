"""
METHODOLOGY 2: Dynamic Covariance Estimation via EWMA
=====================================================
HKUST IEDA3330 Introduction to Financial Engineering - Fall 2025
Prof. Wei JIANG

Based on Lecture 7: Basic Risk Management

This module implements:
1. Exponentially Weighted Moving Average (EWMA) Covariance
   - RiskMetrics methodology
   - Decay factor λ = 0.94 (industry standard)

2. Time-Varying 3×3 Covariance Matrix
   - r_spot: BTC spot returns
   - r_basis: Futures basis returns  
   - dvol_chg: Volatility changes

3. Update Formula:
   σ_ij,t = λ * σ_ij,{t-1} + (1-λ) * r_i,{t-1} * r_j,{t-1}

Key Advantages:
- Adapts to changing volatility regimes
- Captures time-varying correlations
- More responsive than simple moving average

References:
- RiskMetrics Technical Document (J.P. Morgan, 1996)
- Lecture 7: Basic Risk Management (EWMA section)
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional


# ============================================================================
# EWMA PARAMETERS
# ============================================================================

# Industry standard decay factor (RiskMetrics)
EWMA_LAMBDA = 0.94

# Effective window for λ=0.94: 1/(1-λ) ≈ 17 days
EFFECTIVE_WINDOW = int(1 / (1 - EWMA_LAMBDA))


# ============================================================================
# EWMA COVARIANCE ESTIMATOR CLASS
# ============================================================================

class EWMACovarianceEstimator:
    """
    EWMA Covariance Matrix Estimator following RiskMetrics methodology.
    
    This is the core of Methodology 2: Dynamic Covariance Estimation.
    
    Attributes
    ----------
    lambda_ : float
        Decay factor (default: 0.94 from RiskMetrics)
    n_assets : int
        Number of assets/factors in covariance matrix
    current_cov : np.ndarray
        Current covariance matrix estimate
    history : list
        History of covariance matrices
    
    Notes
    -----
    The EWMA update formula (from Lecture 7):
    
        σ²_t = λ * σ²_{t-1} + (1-λ) * r²_{t-1}  (variance)
        
        σ_ij,t = λ * σ_ij,{t-1} + (1-λ) * r_i,{t-1} * r_j,{t-1}  (covariance)
    
    Key properties:
    - λ close to 1: More weight on historical data (smoother)
    - λ close to 0: More weight on recent data (reactive)
    - λ = 0.94 is optimal for daily data (RiskMetrics finding)
    
    Example
    -------
    >>> estimator = EWMACovarianceEstimator(lambda_=0.94, n_assets=3)
    >>> for returns in daily_returns:
    ...     cov_matrix = estimator.update(returns)
    >>> print(estimator.current_cov)
    """
    
    def __init__(self, lambda_: float = EWMA_LAMBDA, n_assets: int = 3):
        """
        Initialize EWMA covariance estimator.
        
        Parameters
        ----------
        lambda_ : float
            Decay factor (0 < λ < 1), default 0.94
        n_assets : int
            Number of assets in covariance matrix
        """
        self.lambda_ = lambda_
        self.n_assets = n_assets
        self.current_cov = None
        self.history = []
        self.initialized = False
        
    def initialize(self, initial_returns: np.ndarray) -> np.ndarray:
        """
        Initialize covariance matrix using sample covariance.
        
        Parameters
        ----------
        initial_returns : np.ndarray
            Initial return observations (T x n_assets)
            Recommend at least 20 observations
        
        Returns
        -------
        np.ndarray
            Initial covariance matrix
        """
        # Use sample covariance for initialization
        self.current_cov = np.cov(initial_returns.T)
        
        # Ensure positive definite
        self.current_cov = self._ensure_positive_definite(self.current_cov)
        
        self.history.append(self.current_cov.copy())
        self.initialized = True
        
        return self.current_cov
    
    def update(self, returns: np.ndarray) -> np.ndarray:
        """
        Update covariance matrix with new return observation.
        
        This is the EWMA update from Lecture 7.
        
        Parameters
        ----------
        returns : np.ndarray
            Return vector for current period (n_assets,)
        
        Returns
        -------
        np.ndarray
            Updated covariance matrix
        
        Notes
        -----
        EWMA Update Formula:
            Σ_t = λ * Σ_{t-1} + (1-λ) * r_{t-1} * r_{t-1}'
        
        Where:
        - Σ_t is the covariance matrix at time t
        - r_{t-1} is the return vector at time t-1
        - λ = 0.94 (RiskMetrics standard)
        """
        if not self.initialized:
            raise ValueError("Estimator not initialized. Call initialize() first.")
        
        returns = np.asarray(returns).flatten()
        
        if len(returns) != self.n_assets:
            raise ValueError(f"Expected {self.n_assets} returns, got {len(returns)}")
        
        # Outer product: r * r'
        outer_product = np.outer(returns, returns)
        
        # EWMA update: Σ_t = λ * Σ_{t-1} + (1-λ) * r * r'
        self.current_cov = (
            self.lambda_ * self.current_cov + 
            (1 - self.lambda_) * outer_product
        )
        
        # Ensure positive definite
        self.current_cov = self._ensure_positive_definite(self.current_cov)
        
        # Store in history
        self.history.append(self.current_cov.copy())
        
        return self.current_cov
    
    def _ensure_positive_definite(self, matrix: np.ndarray, 
                                   min_eigenvalue: float = 1e-8) -> np.ndarray:
        """
        Ensure covariance matrix is positive definite.
        
        Parameters
        ----------
        matrix : np.ndarray
            Input matrix
        min_eigenvalue : float
            Minimum eigenvalue threshold
        
        Returns
        -------
        np.ndarray
            Positive definite matrix
        """
        # Check symmetry
        matrix = (matrix + matrix.T) / 2
        
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        
        # Floor small/negative eigenvalues
        eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
        
        # Reconstruct matrix
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    def get_correlation_matrix(self) -> np.ndarray:
        """
        Get current correlation matrix from covariance.
        
        Returns
        -------
        np.ndarray
            Correlation matrix
        """
        if self.current_cov is None:
            return None
        
        std_devs = np.sqrt(np.diag(self.current_cov))
        correlation = self.current_cov / np.outer(std_devs, std_devs)
        
        # Ensure diagonal is exactly 1
        np.fill_diagonal(correlation, 1.0)
        
        return correlation
    
    def get_volatilities(self) -> np.ndarray:
        """
        Get current volatilities (standard deviations).
        
        Returns
        -------
        np.ndarray
            Annualized volatilities
        """
        if self.current_cov is None:
            return None
        
        daily_vol = np.sqrt(np.diag(self.current_cov))
        annual_vol = daily_vol * np.sqrt(252)  # Annualize
        
        return annual_vol


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def compute_ewma_covariance_series(
    returns_df: pd.DataFrame,
    lambda_: float = EWMA_LAMBDA,
    init_periods: int = 20
) -> List[np.ndarray]:
    """
    Compute EWMA covariance matrices for entire time series.
    
    Parameters
    ----------
    returns_df : pd.DataFrame
        DataFrame with return columns (e.g., r_spot, r_basis, dvol_chg)
    lambda_ : float
        EWMA decay factor
    init_periods : int
        Number of periods for initialization
    
    Returns
    -------
    list
        List of covariance matrices, one per period after init
    
    Example
    -------
    >>> returns = pd.DataFrame({
    ...     'r_spot': spot_returns,
    ...     'r_basis': basis_returns,
    ...     'dvol_chg': dvol_changes
    ... })
    >>> cov_series = compute_ewma_covariance_series(returns)
    >>> print(f"Final covariance:\\n{cov_series[-1]}")
    """
    n_assets = returns_df.shape[1]
    
    # Initialize estimator
    estimator = EWMACovarianceEstimator(lambda_=lambda_, n_assets=n_assets)
    
    # Get initial data
    init_data = returns_df.iloc[:init_periods].values
    estimator.initialize(init_data)
    
    # Update with remaining data
    covariance_series = [estimator.current_cov.copy()]
    
    for i in range(init_periods, len(returns_df)):
        returns = returns_df.iloc[i].values
        cov = estimator.update(returns)
        covariance_series.append(cov.copy())
    
    return covariance_series


def compute_hedge_asset_covariance(asset_returns: pd.DataFrame,
                                    lambda_: float = EWMA_LAMBDA,
                                    init_periods: int = 20) -> List[np.ndarray]:
    """
    Compute EWMA covariance for hedging assets (spot, futures, risk-free).
    
    This is used by Methodology 2 and Methodology 3.
    
    Parameters
    ----------
    asset_returns : pd.DataFrame
        DataFrame with columns: ['r_spot_asset', 'r_futures_asset', 'r_rf']
    lambda_ : float
        EWMA decay factor
    init_periods : int
        Initialization period
    
    Returns
    -------
    List[np.ndarray]
        Covariance matrices for each time period (3x3)
    
    Notes
    -----
    The 3x3 covariance matrix structure:
    
           | Spot   Futures  RF   |
    Spot   | σ²_s   σ_sf     0    |
    Futures| σ_sf   σ²_f     0    |
    RF     | 0      0        ~0   |
    
    Risk-free has near-zero variance and zero correlation.
    """
    # For risk-free, variance is effectively zero
    # Compute 2x2 for spot/futures, then expand
    risky_returns = asset_returns[['r_spot_asset', 'r_futures_asset']].copy()
    
    n_obs = len(asset_returns)
    cov_2x2_series = compute_ewma_covariance_series(risky_returns, lambda_, init_periods)
    
    # Expand to 3x3 with risk-free
    cov_series = []
    rf_var = 1e-10  # Near-zero variance for risk-free
    
    for cov_2x2 in cov_2x2_series:
        cov_3x3 = np.zeros((3, 3))
        cov_3x3[:2, :2] = cov_2x2
        cov_3x3[2, 2] = rf_var
        cov_series.append(cov_3x3)
    
    return cov_series


def compute_ewma_hedge_weights(cov_matrix: np.ndarray, net_delta: float) -> np.ndarray:
    """
    Compute hedge weights using EWMA covariance (Methodology 2).
    
    Uses minimum-variance hedge ratio: h = cov(spot, futures) / var(futures)
    
    Parameters
    ----------
    cov_matrix : np.ndarray
        3x3 covariance matrix [spot, futures, cash]
    net_delta : float
        Net delta of strangle position
    
    Returns
    -------
    np.ndarray
        Hedge weights [spot, futures, cash]
    
    Notes
    -----
    Methodology 2: Uses EWMA covariance to compute minimum-variance hedge ratio.
    This is the classic variance-minimizing hedge from RiskMetrics.
    """
    spot_var = cov_matrix[0, 0]
    futures_var = cov_matrix[1, 1]
    cov_spot_futures = cov_matrix[0, 1]
    
    # Minimum variance hedge ratio (adjusted by covariance)
    if futures_var > 1e-10:
        mv_hedge_ratio = cov_spot_futures / futures_var
    else:
        mv_hedge_ratio = 1.0
    
    # M2: Use covariance-adjusted hedge
    m2_futures_weight = -net_delta * mv_hedge_ratio
    
    # Clamp to reasonable bounds
    m2_futures_weight = np.clip(m2_futures_weight, -0.5, 0.5)
    m2_cash_weight = 1.0 - abs(m2_futures_weight)
    
    return np.array([0.0, m2_futures_weight, m2_cash_weight])


def compute_rolling_correlation(
    returns_df: pd.DataFrame,
    lambda_: float = EWMA_LAMBDA,
    init_periods: int = 20
) -> pd.DataFrame:
    """
    Compute rolling EWMA correlations between assets.
    
    Parameters
    ----------
    returns_df : pd.DataFrame
        DataFrame with return columns
    lambda_ : float
        EWMA decay factor
    init_periods : int
        Initialization periods
    
    Returns
    -------
    pd.DataFrame
        DataFrame with rolling correlations
    """
    cov_series = compute_ewma_covariance_series(returns_df, lambda_, init_periods)
    
    # Extract correlations from each covariance matrix
    correlations = []
    col_names = returns_df.columns.tolist()
    
    for cov in cov_series:
        std_devs = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std_devs, std_devs)
        
        # Extract upper triangle (excluding diagonal)
        corr_dict = {}
        for i in range(len(col_names)):
            for j in range(i+1, len(col_names)):
                key = f"{col_names[i]}_vs_{col_names[j]}"
                corr_dict[key] = corr[i, j]
        
        correlations.append(corr_dict)
    
    # Create DataFrame
    index = returns_df.index[init_periods-1:]
    return pd.DataFrame(correlations, index=index)


def compute_rolling_volatility(
    returns_df: pd.DataFrame,
    lambda_: float = EWMA_LAMBDA,
    init_periods: int = 20,
    annualize: bool = True
) -> pd.DataFrame:
    """
    Compute rolling EWMA volatilities.
    
    Parameters
    ----------
    returns_df : pd.DataFrame
        DataFrame with return columns
    lambda_ : float
        EWMA decay factor
    init_periods : int
        Initialization periods
    annualize : bool
        Whether to annualize volatilities
    
    Returns
    -------
    pd.DataFrame
        DataFrame with rolling volatilities
    """
    cov_series = compute_ewma_covariance_series(returns_df, lambda_, init_periods)
    
    # Extract volatilities
    volatilities = []
    col_names = returns_df.columns.tolist()
    
    for cov in cov_series:
        daily_vol = np.sqrt(np.diag(cov))
        if annualize:
            daily_vol = daily_vol * np.sqrt(252)
        
        vol_dict = {f"vol_{col}": vol for col, vol in zip(col_names, daily_vol)}
        volatilities.append(vol_dict)
    
    # Create DataFrame
    index = returns_df.index[init_periods-1:]
    return pd.DataFrame(volatilities, index=index)


# ============================================================================
# COMPARISON WITH SIMPLE METHODS
# ============================================================================

def compare_ewma_vs_simple(
    returns_df: pd.DataFrame,
    simple_window: int = 30,
    ewma_lambda: float = EWMA_LAMBDA
) -> pd.DataFrame:
    """
    Compare EWMA covariance with simple rolling covariance.
    
    Parameters
    ----------
    returns_df : pd.DataFrame
        Return data
    simple_window : int
        Window for simple rolling covariance
    ewma_lambda : float
        EWMA decay factor
    
    Returns
    -------
    pd.DataFrame
        Comparison of volatility estimates
    """
    # Simple rolling volatility
    simple_vol = returns_df.rolling(window=simple_window).std() * np.sqrt(252)
    
    # EWMA volatility
    ewma_vol = compute_rolling_volatility(returns_df, ewma_lambda, init_periods=20)
    
    # Combine for comparison
    comparison = pd.DataFrame()
    for col in returns_df.columns:
        comparison[f'{col}_simple'] = simple_vol[col]
        comparison[f'{col}_ewma'] = ewma_vol[f'vol_{col}']
    
    return comparison


# ============================================================================
# DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("METHODOLOGY 2: Dynamic Covariance Estimation via EWMA")
    print("="*70)
    print(f"""
    This module implements EWMA covariance estimation from Lecture 7.
    
    KEY PARAMETERS:
    - Lambda (λ) = {EWMA_LAMBDA} (RiskMetrics standard)
    - Effective window ≈ {EFFECTIVE_WINDOW} days
    
    EWMA UPDATE FORMULA:
        Σ_t = λ * Σ_{{t-1}} + (1-λ) * r_{{t-1}} * r_{{t-1}}'
    
    3×3 COVARIANCE MATRIX:
        ┌                           ┐
        │ Var(r_spot)    Cov(s,b)   Cov(s,v) │
        │ Cov(b,s)       Var(r_basis) Cov(b,v) │
        │ Cov(v,s)       Cov(v,b)   Var(dvol) │
        └                           ┘
    
    Where:
    - r_spot: BTC spot log returns
    - r_basis: Futures basis returns (futures - spot)
    - dvol: DVOL (implied volatility) changes
    
    ADVANTAGES OVER SIMPLE ROLLING:
    1. More responsive to recent volatility changes
    2. Exponential decay naturally captures regime shifts
    3. Industry standard (RiskMetrics, Basel II/III)
    
    Reference: Lecture 7 - Basic Risk Management
    """)
    
    # Simple demonstration with synthetic data
    print("\nDEMONSTRATION:")
    print("-" * 50)
    
    np.random.seed(42)
    n_periods = 100
    
    # Generate correlated returns
    true_cov = np.array([
        [0.0009, 0.0007, 0.00005],  # spot variance and covariances
        [0.0007, 0.0008, 0.00004],  # basis
        [0.00005, 0.00004, 0.0001]  # dvol
    ])
    
    returns = np.random.multivariate_normal([0, 0, 0], true_cov, n_periods)
    returns_df = pd.DataFrame(returns, columns=['r_spot', 'r_basis', 'dvol_chg'])
    
    # Run EWMA estimation
    cov_series = compute_ewma_covariance_series(returns_df, init_periods=20)
    
    print(f"True covariance matrix:\n{true_cov}")
    print(f"\nEWMA estimated covariance (final):\n{cov_series[-1]}")
    print(f"\nEstimation error (Frobenius norm): {np.linalg.norm(cov_series[-1] - true_cov):.6f}")


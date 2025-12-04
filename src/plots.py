"""
Visualization Module
====================
HKUST IEDA3330 Introduction to Financial Engineering - Fall 2025
Prof. Wei JIANG

This module visualizes results for THREE METHODOLOGIES:

METHODOLOGY 1: Option Strategy and Delta-Hedging (Lecture 6)
   - File: 1_Option_Delta.py
   - Blue color for Delta Hedge strategy

METHODOLOGY 2: Dynamic Covariance Estimation via EWMA (Lecture 7)
   - File: 2_Covariance_Estimation.py
   - Used internally by M1+M2+M3 strategy

METHODOLOGY 3: Mean-Variance Optimal Portfolio Construction (Lecture 5)
   - File: 3_MV_Optimization.py
   - Green color for MV Optimal strategy

Backtest Strategies Compared:
- Strangle Only (Red): No hedging, baseline
- M1: Delta Hedge (Blue): Methodology 1 with simple hedge
- M1+M2+M3: MV Optimal (Green): All three methodologies combined

Plots Generated:
- Cumulative P&L curves
- P&L distribution histograms
- Efficient frontier (Markowitz)
- Rolling volatility comparison
- Drawdown analysis
- Performance metrics bar charts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Custom color scheme - M1, M2, M3 Methodologies
# M1: Delta Hedge (Lecture 6) - Red
# M2: EWMA Hedge (Lecture 7) - Blue
# M3: MV Optimal (Lecture 5) - Green
COLORS = {
    'm1': '#e74c3c',             # Red - M1: Delta Hedge
    'm2': '#3498db',             # Blue - M2: EWMA Hedge
    'm3': '#2ecc71',             # Green - M3: MV Optimal
    'unhedged': '#95a5a6',       # Gray - Unhedged (if needed)
    'naive': '#e74c3c',          # Alias for M1
    'mv_optimal': '#2ecc71',     # Alias for M3
    'spot': '#f39c12',           # Orange
    'futures': '#9b59b6',        # Purple
    'cash': '#1abc9c',           # Teal
    'frontier': '#34495e',       # Dark gray
    'highlight': '#e67e22'       # Bright orange
}

# Strategy names for plot legends
STRATEGY_NAMES = {
    'm1': 'M1: Delta Hedge\n(Lecture 6)',
    'm2': 'M2: EWMA Hedge\n(Lecture 7)',
    'm3': 'M3: MV Optimal\n(Lecture 5)'
}


# ============================================================================
# CUMULATIVE P&L PLOTS
# ============================================================================

def plot_cumulative_pnl(pnl_history: pd.DataFrame,
                        title: str = "Cumulative P&L: M1 vs M2 vs M3",
                        figsize: tuple = (14, 8),
                        save_path: str = None) -> plt.Figure:
    """
    Plot cumulative P&L comparing three methodologies (M1, M2, M3).
    
    Parameters
    ----------
    pnl_history : pd.DataFrame
        DataFrame with cumulative P&L columns
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    
    Notes
    -----
    Compares three methodologies:
    - M1: Delta Hedge (Lecture 6) - Simple 1:1 futures hedge
    - M2: EWMA Hedge (Lecture 7) - Volatility-adjusted hedge
    - M3: MV Optimal (Lecture 5) - Full Markowitz optimization
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot M1, M2, M3
    ax.plot(pnl_history.index, pnl_history['cum_m1'], 
            color=COLORS['m1'], linewidth=2, 
            label='M1: Delta Hedge (Lecture 6)', alpha=0.8)
    ax.plot(pnl_history.index, pnl_history['cum_m2'], 
            color=COLORS['m2'], linewidth=2, 
            label='M2: EWMA Hedge (Lecture 7)', alpha=0.8)
    ax.plot(pnl_history.index, pnl_history['cum_m3'], 
            color=COLORS['m3'], linewidth=2.5, 
            label='M3: MV Optimal (Lecture 5)', alpha=0.9)
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Formatting
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative P&L ($)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    
    # Tight layout
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_pnl_distribution(pnl_history: pd.DataFrame,
                          figsize: tuple = (14, 5),
                          save_path: str = None) -> plt.Figure:
    """
    Plot distribution of daily P&L for each strategy.
    
    Parameters
    ----------
    pnl_history : pd.DataFrame
        DataFrame with daily P&L columns
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Three methodologies: M1, M2, M3
    strategies = [
        ('pnl_m1', 'M1: Delta Hedge\n(Lecture 6)', COLORS['m1']),
        ('pnl_m2', 'M2: EWMA Hedge\n(Lecture 7)', COLORS['m2']),
        ('pnl_m3', 'M3: MV Optimal\n(Lecture 5)', COLORS['m3'])
    ]
    
    for ax, (col, name, color) in zip(axes, strategies):
        data = pnl_history[col].dropna()
        
        # Histogram with KDE
        ax.hist(data, bins=50, density=True, alpha=0.6, color=color)
        data.plot.kde(ax=ax, color=color, linewidth=2)
        
        # Add VaR lines
        var_95 = np.percentile(data, 5)
        ax.axvline(var_95, color='red', linestyle='--', alpha=0.7, 
                   label=f'95% VaR: ${var_95:.0f}')
        
        # Stats
        mean = data.mean()
        std = data.std()
        ax.axvline(mean, color='black', linestyle='-', alpha=0.5, label=f'Mean: ${mean:.0f}')
        
        ax.set_title(f'{name}\nStd: ${std:.0f}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Daily P&L ($)', fontsize=10)
        ax.legend(fontsize=8)
    
    plt.suptitle('Daily P&L Distribution: M1 vs M2 vs M3', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ============================================================================
# EFFICIENT FRONTIER
# ============================================================================

def plot_efficient_frontier(frontier_df: pd.DataFrame,
                            current_portfolio: tuple = None,
                            figsize: tuple = (10, 8),
                            save_path: str = None) -> plt.Figure:
    """
    Plot the efficient frontier.
    
    Parameters
    ----------
    frontier_df : pd.DataFrame
        Efficient frontier data with 'return' and 'volatility' columns
    current_portfolio : tuple, optional
        (volatility, return) of current portfolio to highlight
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    
    Notes
    -----
    From Lecture 4: The efficient frontier shows the best possible
    risk-return tradeoff available to investors.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Annualize for display
    frontier_df = frontier_df.copy()
    frontier_df['ann_return'] = frontier_df['return'] * 365
    frontier_df['ann_vol'] = frontier_df['volatility'] * np.sqrt(365)
    
    # Plot frontier
    ax.plot(frontier_df['ann_vol'] * 100, frontier_df['ann_return'] * 100,
            color=COLORS['frontier'], linewidth=2.5, marker='o', markersize=4,
            label='Efficient Frontier (Delta-Neutral)')
    
    # Color by Sharpe ratio
    scatter = ax.scatter(frontier_df['ann_vol'] * 100, frontier_df['ann_return'] * 100,
                        c=frontier_df['sharpe'], cmap='RdYlGn', s=50, zorder=5)
    plt.colorbar(scatter, label='Sharpe Ratio')
    
    # Highlight current/optimal portfolio
    if current_portfolio:
        vol, ret = current_portfolio
        ax.scatter([vol * 100], [ret * 100], s=200, c=COLORS['highlight'],
                   marker='*', edgecolor='black', linewidth=1.5,
                   label='Current MV Optimal', zorder=10)
    
    # Formatting
    ax.set_xlabel('Annualized Volatility (%)', fontsize=12)
    ax.set_ylabel('Annualized Return (%)', fontsize=12)
    ax.set_title('Efficient Frontier (Subject to Delta-Neutral Constraint)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ============================================================================
# WEIGHT EVOLUTION
# ============================================================================

def plot_weight_evolution(weights_history: pd.DataFrame,
                          figsize: tuple = (14, 9),
                          save_path: str = None) -> plt.Figure:
    """
    Plot evolution of portfolio weights over time for M1, M2, M3.
    
    Parameters
    ----------
    weights_history : pd.DataFrame
        DataFrame with weight columns over time
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # M1: Delta Hedge weights
    ax1 = axes[0]
    ax1.stackplot(weights_history.index,
                  weights_history['m1_w_spot'].abs(),
                  weights_history['m1_w_futures'].abs(),
                  weights_history['m1_w_cash'].abs(),
                  labels=['Spot', 'Futures', 'Cash'],
                  colors=[COLORS['spot'], COLORS['futures'], COLORS['cash']],
                  alpha=0.8)
    ax1.set_ylabel('Weight', fontsize=11)
    ax1.set_title('M1: Delta Hedge Weights (Lecture 6)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_ylim(0, 1.1)
    
    # M2: EWMA Hedge weights
    ax2 = axes[1]
    ax2.stackplot(weights_history.index,
                  weights_history['m2_w_spot'].abs(),
                  weights_history['m2_w_futures'].abs(),
                  weights_history['m2_w_cash'].abs(),
                  labels=['Spot', 'Futures', 'Cash'],
                  colors=[COLORS['spot'], COLORS['futures'], COLORS['cash']],
                  alpha=0.8)
    ax2.set_ylabel('Weight', fontsize=11)
    ax2.set_title('M2: EWMA Hedge Weights (Lecture 7)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_ylim(0, 1.1)
    
    # M3: MV Optimal weights
    ax3 = axes[2]
    ax3.stackplot(weights_history.index,
                  weights_history['m3_w_spot'].abs(),
                  weights_history['m3_w_futures'].abs(),
                  weights_history['m3_w_cash'].abs(),
                  labels=['Spot', 'Futures', 'Cash'],
                  colors=[COLORS['spot'], COLORS['futures'], COLORS['cash']],
                  alpha=0.8)
    ax3.set_ylabel('Weight', fontsize=11)
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_title('M3: MV Optimal Weights (Lecture 5)', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.set_ylim(0, 1.1)
    
    plt.suptitle('Hedge Weight Evolution: M1 vs M2 vs M3', fontsize=14, fontweight='bold', y=1.02)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ============================================================================
# ROLLING VOLATILITY
# ============================================================================

def plot_rolling_volatility(pnl_history: pd.DataFrame,
                            window: int = 30,
                            figsize: tuple = (14, 6),
                            save_path: str = None) -> plt.Figure:
    """
    Plot rolling volatility comparison.
    
    Parameters
    ----------
    pnl_history : pd.DataFrame
        DataFrame with P&L columns
    window : int
        Rolling window size
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate rolling volatility for M1, M2, M3 (annualized)
    roll_vol_m1 = pnl_history['ret_m1'].rolling(window).std() * np.sqrt(365) * 100
    roll_vol_m2 = pnl_history['ret_m2'].rolling(window).std() * np.sqrt(365) * 100
    roll_vol_m3 = pnl_history['ret_m3'].rolling(window).std() * np.sqrt(365) * 100
    
    ax.plot(pnl_history.index, roll_vol_m1, 
            color=COLORS['m1'], linewidth=1.5, label='M1: Delta Hedge', alpha=0.8)
    ax.plot(pnl_history.index, roll_vol_m2, 
            color=COLORS['m2'], linewidth=1.5, label='M2: EWMA Hedge', alpha=0.8)
    ax.plot(pnl_history.index, roll_vol_m3, 
            color=COLORS['m3'], linewidth=2, label='M3: MV Optimal', alpha=0.9)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(f'{window}-Day Rolling Volatility (% annualized)', fontsize=12)
    ax.set_title(f'Rolling Volatility: M1 vs M2 vs M3 ({window}-Day Window)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ============================================================================
# DRAWDOWN ANALYSIS
# ============================================================================

def plot_drawdowns(pnl_history: pd.DataFrame,
                   figsize: tuple = (14, 6),
                   save_path: str = None,
                   initial_capital: float = 100000) -> plt.Figure:
    """
    Plot drawdown analysis (as percentage) for each strategy.
    
    Parameters
    ----------
    pnl_history : pd.DataFrame
        DataFrame with cumulative P&L columns
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    initial_capital : float
        Initial capital for percentage calculation
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for col, name, color in [
        ('cum_m1', 'M1: Delta Hedge', COLORS['m1']),
        ('cum_m2', 'M2: EWMA Hedge', COLORS['m2']),
        ('cum_m3', 'M3: MV Optimal', COLORS['m3'])
    ]:
        cum_pnl = pnl_history[col]
        # Calculate portfolio value (initial + cumulative P&L)
        portfolio_value = initial_capital + cum_pnl
        # Running maximum of portfolio value
        running_max = portfolio_value.cummax()
        # Drawdown as percentage
        drawdown_pct = (running_max - portfolio_value) / running_max * 100
        
        ax.fill_between(pnl_history.index, 0, -drawdown_pct, 
                       color=color, alpha=0.4, label=name)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.set_title('Drawdown Analysis: M1 vs M2 vs M3', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{-x:.1f}%'))
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ============================================================================
# PERFORMANCE COMPARISON BAR CHART
# ============================================================================

def plot_metrics_comparison(metrics_df: pd.DataFrame,
                            figsize: tuple = (12, 8),
                            save_path: str = None) -> plt.Figure:
    """
    Create bar chart comparing key metrics across strategies.
    
    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame with metrics for each strategy
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # M1, M2, M3 labels for x-axis
    strategies = ['M1:\nDelta Hedge', 'M2:\nEWMA Hedge', 'M3:\nMV Optimal']
    colors = [COLORS['m1'], COLORS['m2'], COLORS['m3']]
    
    # Column names in metrics_df
    col_names = ['M1: Delta Hedge', 'M2: EWMA Hedge', 'M3: MV Optimal']
    
    # Convert string percentages to floats if needed
    def parse_pct(val):
        if isinstance(val, str):
            return float(val.strip('%')) / 100
        return val
    
    # Helper to get value
    def get_metric(metric_name, col_idx):
        try:
            return metrics_df[metrics_df['Metric'] == metric_name][col_names[col_idx]].values[0]
        except (KeyError, IndexError):
            return 0
    
    # Sharpe Ratio
    ax1 = axes[0, 0]
    sharpe_vals = [float(get_metric('Sharpe Ratio', i)) for i in range(3)]
    ax1.bar(strategies, sharpe_vals, color=colors)
    ax1.set_ylabel('Sharpe Ratio', fontsize=11)
    ax1.set_title('Sharpe Ratio Comparison', fontsize=12, fontweight='bold')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Volatility
    ax2 = axes[0, 1]
    vol_vals = [parse_pct(get_metric('Annualized Volatility', i)) for i in range(3)]
    ax2.bar(strategies, [v * 100 for v in vol_vals], color=colors)
    ax2.set_ylabel('Volatility (%)', fontsize=11)
    ax2.set_title('Annualized Volatility', fontsize=12, fontweight='bold')
    
    # Max Drawdown
    ax3 = axes[1, 0]
    dd_vals = [parse_pct(get_metric('Max Drawdown', i)) for i in range(3)]
    ax3.bar(strategies, [d * 100 for d in dd_vals], color=colors)
    ax3.set_ylabel('Max Drawdown (%)', fontsize=11)
    ax3.set_title('Maximum Drawdown', fontsize=12, fontweight='bold')
    
    # Win Rate
    ax4 = axes[1, 1]
    wr_vals = [parse_pct(get_metric('Win Rate', i)) for i in range(3)]
    ax4.bar(strategies, [w * 100 for w in wr_vals], color=colors)
    ax4.set_ylabel('Win Rate (%)', fontsize=11)
    ax4.set_title('Daily Win Rate', fontsize=12, fontweight='bold')
    
    plt.suptitle('Performance Metrics: M1 vs M2 vs M3', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ============================================================================
# BTC PRICE WITH EVENTS
# ============================================================================

def plot_btc_price_context(df: pd.DataFrame,
                           pnl_history: pd.DataFrame = None,
                           figsize: tuple = (14, 10),
                           save_path: str = None) -> plt.Figure:
    """
    Plot BTC price with key events and optionally overlay P&L.
    
    Parameters
    ----------
    df : pd.DataFrame
        Market data with spot prices
    pnl_history : pd.DataFrame, optional
        P&L history to overlay
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True,
                            gridspec_kw={'height_ratios': [2, 1]})
    
    # BTC Price
    ax1 = axes[0]
    ax1.plot(df.index, df['spot'], color=COLORS['spot'], linewidth=1.5)
    ax1.set_ylabel('BTC Price (USD)', fontsize=12)
    ax1.set_title('BTC Spot Price', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add key events
    events = [
        ('2022-05-09', 'Terra/Luna\nCollapse'),
        ('2022-11-08', 'FTX\nBankruptcy'),
        ('2024-01-10', 'BTC ETF\nApproved'),
    ]
    
    for date, label in events:
        try:
            if date in df.index.astype(str).tolist() or pd.Timestamp(date) in df.index:
                ax1.axvline(pd.Timestamp(date), color='red', linestyle='--', alpha=0.5)
                ax1.annotate(label, xy=(pd.Timestamp(date), df['spot'].max() * 0.9),
                           fontsize=8, ha='center', color='red')
        except:
            pass
    
    # DVOL
    ax2 = axes[1]
    ax2.plot(df.index, df['dvol'], color=COLORS['futures'], linewidth=1.5)
    ax2.set_ylabel('DVOL (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_title('BTC Implied Volatility (DVOL)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ============================================================================
# GENERATE ALL PLOTS
# ============================================================================

def generate_all_plots(engine, output_dir: str = "figures/"):
    """
    Generate all standard plots from backtest results.
    
    Parameters
    ----------
    engine : BacktestEngine
        Completed backtest engine
    output_dir : str
        Directory to save figures
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating plots...")
    
    # 1. Cumulative P&L
    fig1 = plot_cumulative_pnl(engine.pnl_history, 
                               save_path=f"{output_dir}cumulative_pnl.png")
    plt.close(fig1)
    print("  - Cumulative P&L plot saved")
    
    # 2. P&L Distribution
    fig2 = plot_pnl_distribution(engine.pnl_history,
                                 save_path=f"{output_dir}pnl_distribution.png")
    plt.close(fig2)
    print("  - P&L distribution plot saved")
    
    # 3. Weight Evolution
    fig3 = plot_weight_evolution(engine.weights_history,
                                save_path=f"{output_dir}weight_evolution.png")
    plt.close(fig3)
    print("  - Weight evolution plot saved")
    
    # 4. Rolling Volatility
    fig4 = plot_rolling_volatility(engine.pnl_history,
                                   save_path=f"{output_dir}rolling_volatility.png")
    plt.close(fig4)
    print("  - Rolling volatility plot saved")
    
    # 5. Drawdowns
    fig5 = plot_drawdowns(engine.pnl_history,
                         save_path=f"{output_dir}drawdowns.png")
    plt.close(fig5)
    print("  - Drawdown plot saved")
    
    # 6. BTC Context
    fig6 = plot_btc_price_context(engine.df,
                                  save_path=f"{output_dir}btc_context.png")
    plt.close(fig6)
    print("  - BTC context plot saved")
    
    # 7. Metrics Comparison
    try:
        summary = engine.get_summary_table()
        fig7 = plot_metrics_comparison(summary,
                                       save_path=f"{output_dir}metrics_comparison.png")
        plt.close(fig7)
        print("  - Metrics comparison plot saved")
    except Exception as e:
        print(f"  - Skipped metrics comparison: {e}")
    
    print(f"All plots saved to {output_dir}")


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    from .backtest import run_full_backtest
    
    # Run backtest
    print("Running backtest for plot testing...")
    engine, summary = run_full_backtest("2022-01-01", "2025-12-03")
    
    # Generate all plots
    generate_all_plots(engine, output_dir="figures/")
    
    print("\nAll plots generated successfully!")


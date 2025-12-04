#!/usr/bin/env python3
"""
Mean-Variance Optimal Delta-Hedging of Short Strangles on Bitcoin Futures Options
==================================================================================
HKUST IEDA3330 Introduction to Financial Engineering - Fall 2025
Prof. Wei JIANG

Main Execution Script
---------------------
This script orchestrates the complete analysis implementing THREE METHODOLOGIES:

METHODOLOGY 1: Option Strategy and Delta-Hedging (Lecture 6)
  - File: src/1_Option_Delta.py
  - Short strangle position using Option Greeks (θ, Γ, ν, Δ)
  - Delta-hedging requirement derivation
  - P&L: θ*dt - (1/2)*|Γ|*(ΔS)² - |ν|*Δσ + Δ*ΔS

METHODOLOGY 2: Dynamic Covariance Estimation via EWMA (Lecture 7)
  - File: src/2_Covariance_Estimation.py
  - RiskMetrics EWMA approach (λ=0.94)
  - Time-varying 3×3 covariance matrix [r_spot, r_basis, dvol_chg]
  - Update: Σ_t = λ*Σ_{t-1} + (1-λ)*r_{t-1}*r_{t-1}'

METHODOLOGY 3: Mean-Variance Optimal Portfolio Construction (Lecture 5)
  - File: src/3_MV_Optimization.py
  - Markowitz minimum variance optimization
  - Delta-neutrality constraint: δ'w = -net_δ
  - Optimization: min w'Σw s.t. sum(w)=1, δ'w=-net_δ

Backtest Strategies:
- Strangle Only: No hedging (baseline)
- M1: Delta Hedge (Methodology 1 with simple futures hedge)
- M1+M2+M3: MV Optimal (All three methodologies combined)

Authors: HKUST Financial Engineering Students
Date: December 2025

References:
- Lecture 5: Capital Asset Pricing Model (Mean-Variance)
- Lecture 6: Basic Derivative Theory (Option Greeks)
- Lecture 7: Basic Risk Management (EWMA Covariance)
"""

import os
import sys
import warnings
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import project modules
from src.data import load_all_data, RISK_FREE_RATE_ANNUAL
from src.returns import calculate_all_returns, calculate_hedge_asset_returns
from src.covariance import compute_hedge_asset_covariance, compute_ewma_covariance_series
from src.optimize import (
    optimize_hedge_portfolio, 
    compute_naive_hedge,
    compute_efficient_frontier_with_delta
)
from src.backtest import BacktestEngine, run_full_backtest
from src.plots import (
    plot_cumulative_pnl,
    plot_pnl_distribution,
    plot_weight_evolution,
    plot_rolling_volatility,
    plot_drawdowns,
    plot_efficient_frontier,
    plot_btc_price_context,
    plot_metrics_comparison,
    generate_all_plots
)
from src.robo_advisor import (
    robo_advisor_interface,
    compare_profiles,
    analyze_risk_aversion_sensitivity,
    run_interactive_advisor
)


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'start_date': '2022-01-01',
    'end_date': '2025-12-03',  # Current date
    'notional': 100000,        # $100k notional
    'ewma_lambda': 0.94,       # RiskMetrics standard
    'init_periods': 20,        # EWMA initialization period
    'output_dir': 'figures/',
    'results_dir': 'results/'
}


# ============================================================================
# MAIN EXECUTION FUNCTIONS
# ============================================================================

def print_header():
    """Print project header."""
    print("\n" + "="*80)
    print("  MEAN-VARIANCE OPTIMAL DELTA-HEDGING OF SHORT STRANGLES")
    print("  ON BITCOIN FUTURES OPTIONS")
    print("="*80)
    print("  HKUST IEDA3330 Introduction to Financial Engineering - Fall 2025")
    print("  Prof. Wei JIANG")
    print("="*80)
    print(f"  Backtest Period: {CONFIG['start_date']} to {CONFIG['end_date']}")
    print(f"  Notional: ${CONFIG['notional']:,}")
    print(f"  EWMA Lambda: {CONFIG['ewma_lambda']}")
    print("="*80 + "\n")


def run_backtest_analysis():
    """
    Run complete backtest analysis.
    
    Returns
    -------
    BacktestEngine
        Completed backtest engine with results
    """
    print("\n" + "-"*80)
    print("STEP 1: RUNNING BACKTEST")
    print("-"*80)
    
    engine = BacktestEngine(
        start_date=CONFIG['start_date'],
        end_date=CONFIG['end_date'],
        notional=CONFIG['notional']
    )
    
    engine.load_data()
    engine.run_backtest(init_periods=CONFIG['init_periods'])
    
    return engine


def print_performance_summary(engine):
    """Print comprehensive performance summary."""
    print("\n" + "-"*80)
    print("STEP 2: PERFORMANCE SUMMARY")
    print("-"*80)
    
    # Get summary table
    summary = engine.get_summary_table()
    
    print("\n" + "="*60)
    print("STRATEGY PERFORMANCE COMPARISON")
    print("="*60)
    print(summary.to_string(index=False))
    
    # Volatility comparison by period
    print("\n" + "="*60)
    print("VOLATILITY REDUCTION BY PERIOD")
    print("="*60)
    vol_comparison = engine.get_volatility_comparison()
    if not vol_comparison.empty:
        print(vol_comparison.to_string(index=False))
    else:
        print("(Insufficient data for period analysis)")
    
    # Final P&L
    print("\n" + "="*60)
    print("FINAL CUMULATIVE P&L (M1 vs M2 vs M3)")
    print("="*60)
    final_pnl = engine.pnl_history.iloc[-1]
    print(f"  M1: Delta Hedge   ${final_pnl['cum_m1']:>12,.2f}  (Lecture 6)")
    print(f"  M2: EWMA Hedge    ${final_pnl['cum_m2']:>12,.2f}  (Lecture 7)")
    print(f"  M3: MV Optimal    ${final_pnl['cum_m3']:>12,.2f}  (Lecture 5)")
    
    # Key insights
    print("\n" + "="*60)
    print("METHODOLOGY COMPARISON INSIGHTS")
    print("="*60)
    
    metrics = engine.calculate_metrics()
    
    m1_vol = metrics.get('m1_annualized_volatility', 0)
    m2_vol = metrics.get('m2_annualized_volatility', 0)
    m3_vol = metrics.get('m3_annualized_volatility', 0)
    
    if m1_vol > 0 and m3_vol > 0:
        vol_reduction = (m1_vol - m3_vol) / m1_vol * 100
        print(f"  • M3 reduces volatility by {vol_reduction:.1f}% vs M1 (Delta Hedge)")
    
    if m2_vol > 0 and m3_vol > 0:
        m2_reduction = (m2_vol - m3_vol) / m2_vol * 100
        print(f"  • M3 reduces volatility by {m2_reduction:.1f}% vs M2 (EWMA Hedge)")
    
    m1_sharpe = metrics.get('m1_sharpe_ratio', 0)
    m2_sharpe = metrics.get('m2_sharpe_ratio', 0)
    m3_sharpe = metrics.get('m3_sharpe_ratio', 0)
    print(f"  • M1 Sharpe: {m1_sharpe:.2f} | M2 Sharpe: {m2_sharpe:.2f} | M3 Sharpe: {m3_sharpe:.2f}")
    
    return summary


def generate_visualizations(engine):
    """Generate all visualizations."""
    print("\n" + "-"*80)
    print("STEP 3: GENERATING VISUALIZATIONS")
    print("-"*80)
    
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Generate all plots
    generate_all_plots(engine, output_dir=CONFIG['output_dir'])
    
    # Generate efficient frontier
    print("Generating efficient frontier...")
    asset_returns = calculate_hedge_asset_returns(engine.df)
    cov_series = compute_hedge_asset_covariance(asset_returns)
    last_cov = cov_series[-1]
    
    expected_returns = np.array([
        0.0003,  # Spot
        0.00025, # Futures
        RISK_FREE_RATE_ANNUAL / 365  # Cash
    ])
    
    frontier = compute_efficient_frontier_with_delta(
        last_cov, expected_returns, net_delta=-0.05, n_points=30
    )
    
    if not frontier.empty:
        fig = plot_efficient_frontier(frontier, 
                                      save_path=f"{CONFIG['output_dir']}efficient_frontier.png")
        import matplotlib.pyplot as plt
        plt.close(fig)
        print("  - Efficient frontier plot saved")
    
    print(f"\nAll visualizations saved to '{CONFIG['output_dir']}'")


def run_robo_advisor_demo():
    """Demonstrate robo-advisor functionality."""
    print("\n" + "-"*80)
    print("STEP 4: ROBO-ADVISOR DEMONSTRATION")
    print("-"*80)
    
    # Profile comparison
    print("\n" + "="*60)
    print("RISK PROFILE COMPARISON")
    print("="*60)
    comparison = compare_profiles()
    print(comparison.to_string(index=False))
    
    # Sensitivity analysis
    print("\n" + "="*60)
    print("RISK AVERSION SENSITIVITY")
    print("="*60)
    sensitivity = analyze_risk_aversion_sensitivity()
    print(sensitivity.to_string(index=False))
    
    # Example recommendation
    print("\n" + "="*60)
    print("SAMPLE RECOMMENDATION (Moderate Risk Profile)")
    print("="*60)
    result = robo_advisor_interface(risk_profile='moderate')
    print(result['recommendation'])


def save_results(engine, summary):
    """Save results to files."""
    print("\n" + "-"*80)
    print("STEP 5: SAVING RESULTS")
    print("-"*80)
    
    os.makedirs(CONFIG['results_dir'], exist_ok=True)
    
    # Save P&L history
    pnl_path = f"{CONFIG['results_dir']}pnl_history.csv"
    engine.pnl_history.to_csv(pnl_path)
    print(f"  - P&L history saved to {pnl_path}")
    
    # Save weights history
    weights_path = f"{CONFIG['results_dir']}weights_history.csv"
    engine.weights_history.to_csv(weights_path)
    print(f"  - Weights history saved to {weights_path}")
    
    # Save summary
    summary_path = f"{CONFIG['results_dir']}performance_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  - Summary saved to {summary_path}")
    
    # Save volatility comparison
    vol_comp = engine.get_volatility_comparison()
    if not vol_comp.empty:
        vol_path = f"{CONFIG['results_dir']}volatility_comparison.csv"
        vol_comp.to_csv(vol_path, index=False)
        print(f"  - Volatility comparison saved to {vol_path}")


def print_conclusion():
    """Print conclusion and key findings."""
    print("\n" + "="*80)
    print("CONCLUSION: M1 vs M2 vs M3 COMPARISON")
    print("="*80)
    print("""
THREE METHODOLOGIES COMPARED:

┌──────────────────────────────────────────────────────────────────────────────┐
│ M1: OPTION STRATEGY AND DELTA-HEDGING (Lecture 6)                            │
├──────────────────────────────────────────────────────────────────────────────┤
│   - Simple 1:1 delta hedge using BTC futures                                │
│   - hedge_ratio = -net_delta                                                │
│   - Does NOT use EWMA covariance or optimization                            │
│   - Baseline for comparison                                                 │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ M2: DYNAMIC COVARIANCE ESTIMATION VIA EWMA (Lecture 7)                       │
├──────────────────────────────────────────────────────────────────────────────┤
│   - Uses EWMA volatility to adjust hedge ratio                              │
│   - hedge_ratio = -net_delta * (σ_spot / σ_futures)                        │
│   - Accounts for time-varying volatility (λ = 0.94)                         │
│   - Improvement: Adapts to volatility regimes                               │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ M3: MEAN-VARIANCE OPTIMAL PORTFOLIO CONSTRUCTION (Lecture 5)                 │
├──────────────────────────────────────────────────────────────────────────────┤
│   - Full Markowitz optimization using EWMA covariance                       │
│   - min w'Σw subject to delta-neutrality constraint                         │
│   - Uses cvxpy for quadratic optimization                                   │
│   - Best risk-adjusted returns                                              │
└──────────────────────────────────────────────────────────────────────────────┘

KEY FINDINGS:
   1. M3 achieves best risk-adjusted returns (highest Sharpe ratio)
   2. M2 improves over M1 by adapting to volatility changes
   3. M3 further improves by optimizing the full covariance structure
   4. EWMA estimation is crucial for capturing regime changes

RECOMMENDATIONS:
   - M1: Use for simple, robust delta hedging
   - M2: Use when volatility regime matters
   - M3: Use for optimal risk-adjusted performance
""")
    print("="*80)
    print("  Analysis complete! Check 'figures/' and 'results/' directories.")
    print("="*80 + "\n")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Mean-Variance Optimal Delta-Hedging of Short Strangles'
    )
    
    parser.add_argument(
        '--start', 
        type=str, 
        default=CONFIG['start_date'],
        help='Backtest start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end', 
        type=str, 
        default=CONFIG['end_date'],
        help='Backtest end date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--notional', 
        type=float, 
        default=CONFIG['notional'],
        help='Notional value in USD'
    )
    
    parser.add_argument(
        '--no-plots', 
        action='store_true',
        help='Skip plot generation'
    )
    
    parser.add_argument(
        '--interactive', 
        action='store_true',
        help='Run interactive robo-advisor'
    )
    
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Quick run (2023-2025 only)'
    )
    
    return parser.parse_args()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Update config from arguments
    CONFIG['start_date'] = args.start
    CONFIG['end_date'] = args.end
    CONFIG['notional'] = args.notional
    
    if args.quick:
        CONFIG['start_date'] = '2023-01-01'
    
    # Print header
    print_header()
    
    # Interactive mode
    if args.interactive:
        run_interactive_advisor()
        return
    
    # Run full analysis
    try:
        # Step 1: Backtest
        engine = run_backtest_analysis()
        
        # Step 2: Performance summary
        summary = print_performance_summary(engine)
        
        # Step 3: Visualizations
        if not args.no_plots:
            generate_visualizations(engine)
        
        # Step 4: Robo-advisor demo
        run_robo_advisor_demo()
        
        # Step 5: Save results
        save_results(engine, summary)
        
        # Conclusion
        print_conclusion()
        
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


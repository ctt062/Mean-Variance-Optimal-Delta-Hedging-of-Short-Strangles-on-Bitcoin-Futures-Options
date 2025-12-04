# Code Analysis: Duplicate and Unused Files

## Summary

**UNUSED FILES (Can be removed):**
- `src/1_Option_Delta.py` - Standalone demonstration, NOT imported anywhere
- `src/2_Covariance_Estimation.py` - Standalone demonstration, NOT imported anywhere  
- `src/3_MV_Optimization.py` - Standalone demonstration, NOT imported anywhere

**USED FILES (Keep these):**
- `src/data.py` - Used by main.py, backtest.py
- `src/returns.py` - Used by main.py, backtest.py
- `src/covariance.py` - Used by main.py, backtest.py, optimize.py (M2: EWMA)
- `src/optimize.py` - Used by main.py, backtest.py (M3: MV Optimization)
- `src/backtest.py` - Used by main.py (contains M1, M2, M3 comparison)
- `src/plots.py` - Used by main.py
- `src/robo_advisor.py` - Used by main.py
- `src/__init__.py` - Package initialization

## Detailed Analysis

### Import Chain Analysis

**main.py imports:**
```
- src.data
- src.returns  
- src.covariance
- src.optimize
- src.backtest
- src.plots
- src.robo_advisor
```

**backtest.py imports:**
```
- .data
- .returns
- .covariance
- .optimize
```

**optimize.py imports:**
```
- .data
- .covariance
```

**Result:** The three methodology files (1_Option_Delta.py, 2_Covariance_Estimation.py, 3_MV_Optimization.py) are **NEVER imported** by any code.

### Functionality Mapping

| Methodology | Standalone File | Actual Implementation | Status |
|------------|----------------|---------------------|--------|
| **M1: Option Strategy** | `1_Option_Delta.py` | `backtest.py::calculate_strangle_pnl()` | **DUPLICATE** |
| **M2: EWMA Covariance** | `2_Covariance_Estimation.py` | `covariance.py::EWMACovariance` | **DUPLICATE** |
| **M3: MV Optimization** | `3_MV_Optimization.py` | `optimize.py::optimize_hedge_portfolio()` | **DUPLICATE** |

### Recommendation

**DELETE these files:**
- `src/1_Option_Delta.py` - Functionality exists in `backtest.py`
- `src/2_Covariance_Estimation.py` - Functionality exists in `covariance.py`
- `src/3_MV_Optimization.py` - Functionality exists in `optimize.py`

**Reason:** These files are standalone demonstrations/documentation but are not used by the actual backtest. The real implementation is in the core modules that are imported and used.

### Notes

- The standalone files contain good documentation and examples, but they duplicate functionality
- If you want to keep them for documentation purposes, consider moving them to a `docs/` or `examples/` folder
- The actual backtest uses the implementations in `backtest.py`, `covariance.py`, and `optimize.py`


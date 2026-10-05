# Collateralized Debt Obligation | Structured Finance
Mini-Project 3, Valuation for Financial Engineering
Team: Jiayi Chen, Binny Singla

# Simplified CDO Analysis: Correlated Defaults and Tranche Cash Flows

Monte Carlo simulation of a collateralized debt obligation (CDO) backed by 10 speculative-grade corporate bonds. The model generates correlated default times with a Gaussian copula, builds quarterly collateral cash flows, and distributes them through a two-tranche waterfall.

## The deal

| Component | Notional | Coupon | Priority |
| --- | --- | --- | --- |
| Collateral: 10 bonds | $10MM each | 6%, paid quarterly | Source of cash |
| Class A | $20MM | 2% | 1st |
| Class B | $10MM | 4% | 2nd |
| Equity (retained by bank) | Residual | — | 3rd |

Each bond has a 5-year maturity, a 4% annual default probability, a 60% loss given default, and a pairwise default correlation of 0.20.

## Method

1. Draw 1,000 × 10 standard normals (seed 42), apply moment matching, and sort scenarios from worst to best.
2. Correlate them with a Cholesky decomposition of the target correlation matrix.
3. Convert them into exponential default times with a Gaussian copula.
4. Build each bond's quarterly cash flows. A defaulting bond pays its coupon in the default quarter and its recovery in the next quarter.
5. Pay each quarter's pool cash to Class A, then Class B, then the equity. Shortfalls are not carried forward.
6. Report the distributions of collateral and tranche cash flows, and run sensitivities to correlation, PD and LGD.

## Key results (base case)

- The pool returns $115.97MM on average against $130MM promised.
- Class A and Class B are short-paid in only 0.2% and 0.6% of scenarios.
- The equity absorbs about 99% of the expected loss.
- Higher correlation mainly raises the tranches' tail risk. Under a combined stress (PD 8%, correlation 0.5), Class A is short-paid in 7.4% of scenarios and Class B in 12.2%.

## Repository structure

```
notebooks/CDO Risk Analysis.ipynb   Main analysis (Tasks 1–3)
src/cdo_funs.py                     Bond, portfolio and tranche cash flow functions
src/sensitivity_funs.py             Model runner and sensitivity analysis
data/raw/fixed_normals.csv          Fixed random draws used in every run
Output/                             Saved figures
```

## How to run

```bash
pip install numpy pandas scipy matplotlib
jupyter notebook "notebooks/CDO Risk Analysis.ipynb"
```

Run all cells from the top. The notebook sets the working directory to the project root automatically.

# Credit Default Swap | Structured Finance
# Simplified CDO Analysis: Correlated Defaults and Tranche Cash Flows

Monte Carlo simulation of a collateralized debt obligation (CDO) backed by 10 speculative-grade corporate bonds. The model generates correlated default times with a Gaussian copula, builds quarterly collateral cash flows, and distributes them through a two-tranche waterfall.

Course project for FRE 6103 Valuation for Financial Engineers (NYU Tandon, Fall 2026).

## Problem setup

**Collateral:** 10 corporate bonds, $10MM face each

| Parameter | Value |
|---|---|
| Maturity | 5 years (20 quarters) |
| Coupon | 6% annual, paid quarterly |
| Annual default probability | 4% |
| Loss given default | 60% |
| Pairwise default correlation | 0.20 |

**CDO structure:**

| Tranche | Notional | Coupon | Priority |
|---|---|---|---|
| Class A | $20MM | 2% | Paid first |
| Class B | $10MM | 4% | Paid second |
| Equity | Residual | — | Retained by the bank |

No carry-forwards: all cash is distributed as earned each quarter.

## Methodology (Task 1)

1. **Fixed random numbers:** 1,000 cases × 10 bonds of standard normals, saved once so every scenario and sensitivity run uses the same draws (common random numbers).
2. **Moment matching:** each bond's draws are rescaled to exactly mean 0 and SD 1.
3. **Sorting:** cases are ordered from worst to best by the sum of their draws, so case 1 is the most adverse scenario.
4. **Correlation:** Cholesky decomposition of the 10 × 10 correlation matrix (ρ = 0.20) turns independent normals into correlated normals.
5. **Default times:** Gaussian copula with exponential marginals, τ = −ln(1 − Φ(Y)) / λ, with λ = −ln(0.96) so the annual PD is exactly 4%.
6. **Bond cash flows:** coupons until default, 40% recovery in the default quarter, nothing afterwards; survivors repay face at maturity.
7. **Portfolio cash flows:** bond cash flows summed by quarter, giving a 1,000 × 20 matrix that feeds the waterfall.

## Key results so far

- Simulated 5-year default rate: 18.27% (target 18.46%).
- Correlation leaves the mean number of defaults unchanged (~1.85 per case) but fattens the tail: P(5+ defaults) is **8.4%**, versus 2.4% under independence.
- Simulated mean quarterly cash flows match the analytic expectation in every quarter within 3 standard errors.


## How to run

Requires Python 3.11 with `numpy`, `pandas`, `scipy`, and `matplotlib`.

```bash
pip install numpy pandas scipy matplotlib
```

Open the notebook and choose **Run All**. If `data/raw/fixed_normals.csv` exists, the same random draws are reused; otherwise they are generated with a fixed seed.

## Status

- [x] Task 1: Correlated default simulation and collateral cash flows
- [ ] Task 2: Waterfall allocation to Class A, Class B, and equity
- [ ] Task 3: Statistical analysis and sensitivities (PD, LGD, correlation)


## Status

Work in progress.

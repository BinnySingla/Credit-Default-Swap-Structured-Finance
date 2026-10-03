"""
sensitivity_funs.py

How it works:
    1. BASE holds the base-case parameters.
    2. GRID holds the values to test for each parameter.
    3. run_model() runs the full pipeline (correlation -> defaults -> cash flows -> waterfall)
       for ONE set of parameters, always using the SAME fixed random numbers Z.
    4. run_sensitivity() changes ONE parameter at a time, keeps the others at BASE,
       and returns a table of key metrics.
    5. run_stress() changes TWO parameters together (e.g. PD and rho in a recession).
 
"""
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
 
from src.cdo_funs import all_bond_cash_flows, portfolio_cash_flows
 
 
# ---------------------------------------------------------------------------
# 1. Parameters
# ---------------------------------------------------------------------------
BASE = {
    "rho": 0.20,          # pairwise default correlation
    "pd_annual": 0.04,    # annual default probability
    "LGD": 0.60,          # loss given default
}
 
GRID = {
    "rho":       [0.0, 0.1, 0.2, 0.3, 0.5],
    "pd_annual": [0.02, 0.04, 0.06, 0.08],
    "LGD":       [0.40, 0.60, 0.80],
}
 
# Deal terms (fixed in this project)
DEAL = {
    "face": 10.0,          # each bond, $MM
    "coupon_rate": 0.06,   # bond coupon, annual
    "n_quarters": 20,      # 5 years
    "A_notional": 20.0, "A_rate": 0.02,
    "B_notional": 10.0, "B_rate": 0.04,
}
 
# Metrics shown in the plots by default
KEY_METRICS = ["P(5+ defaults)", "P(A shortfall)", "P(B shortfall)", "Equity mean"]
 
 
# ---------------------------------------------------------------------------
# 2. Model pieces
# ---------------------------------------------------------------------------
def default_quarters_from_Z(Z, rho, pd_annual, n_quarters=20):
    """
    Z: fixed, moment-matched, sorted independent normals (cases x bonds).
    Returns default quarters (1..n_quarters), 0 = survives.
    Same steps as Task 1, Steps 4-5.
    """
    n_bonds = Z.shape[1]
 
    # Step 4: correlation via Cholesky (§5.7.2)
    R = np.full((n_bonds, n_bonds), rho)
    np.fill_diagonal(R, 1.0)
    L = np.linalg.cholesky(R)
    Y = (L @ Z.T).T
 
    # Step 5: default times via Gaussian copula + exponential marginals (§5.8.1)
    lam = -np.log(1 - pd_annual)
    tau_q = -np.log(1 - norm.cdf(Y)) / lam * 4
 
    return np.where(tau_q <= n_quarters, np.ceil(tau_q), 0).astype(int)
 
 
def waterfall(port_cf, deal=DEAL):
    """
    Pay Class A first, then Class B, residual to equity. No carry-forwards.
    Interest every quarter, principal at the final quarter.
    Returns (A_cf, B_cf, E_cf), each of shape (n_cases, n_quarters).
    """
    n_q = port_cf.shape[1]
 
    A_due = np.full(n_q, deal["A_notional"] * deal["A_rate"] / 4)
    B_due = np.full(n_q, deal["B_notional"] * deal["B_rate"] / 4)
    A_due[-1] += deal["A_notional"]
    B_due[-1] += deal["B_notional"]
 
    A_cf = np.minimum(port_cf, A_due)       # Class A paid first from all available cash
    after_A = port_cf - A_cf
    B_cf = np.minimum(after_A, B_due)       # Class B paid from what is left
    E_cf = after_A - B_cf                   # residual to equity
 
    return A_cf, B_cf, E_cf
 
 
def run_model(Z, rho, pd_annual, LGD, deal=DEAL):
    """Full pipeline (Task 1 Steps 4-7 + Task 2) for one parameter set."""
    default_q = default_quarters_from_Z(Z, rho, pd_annual, deal["n_quarters"])
    bond_cf = all_bond_cash_flows(default_q, face=deal["face"],
                                  coupon_rate=deal["coupon_rate"], LGD=LGD,
                                  n_quarters=deal["n_quarters"])
    port_cf = portfolio_cash_flows(bond_cf)
    A_cf, B_cf, E_cf = waterfall(port_cf, deal)
 
    return {"default_q": default_q, "port_cf": port_cf,
            "A_cf": A_cf, "B_cf": B_cf, "E_cf": E_cf}
 
 
# ---------------------------------------------------------------------------
# 3. Metrics
# ---------------------------------------------------------------------------
def summarize(res, deal=DEAL):
    """Key metrics for one model run (totals over 5 years, $MM)."""
    n_q = deal["n_quarters"]
    A_promised = deal["A_notional"] * deal["A_rate"] / 4 * n_q + deal["A_notional"]   # 22
    B_promised = deal["B_notional"] * deal["B_rate"] / 4 * n_q + deal["B_notional"]   # 12
 
    n_defaults = (res["default_q"] > 0).sum(axis=1)
    coll = res["port_cf"].sum(axis=1)
    A = res["A_cf"].sum(axis=1)
    B = res["B_cf"].sum(axis=1)
    E = res["E_cf"].sum(axis=1)
 
    return {
        "Avg defaults":          n_defaults.mean(),
        "P(5+ defaults)":        (n_defaults >= 5).mean(),
        "Collateral mean":       coll.mean(),
        "Collateral 1st pct":    np.percentile(coll, 1),
        "P(A shortfall)":        (A < A_promised - 1e-9).mean(),
        "A expected shortfall":  A_promised - A.mean(),
        "P(B shortfall)":        (B < B_promised - 1e-9).mean(),
        "B expected shortfall":  B_promised - B.mean(),
        "Equity mean":           E.mean(),
        "Equity 5th pct":        np.percentile(E, 5),
    }
 
 
# ---------------------------------------------------------------------------
# 4. Sensitivity runs
# ---------------------------------------------------------------------------
def run_sensitivity(Z, param, values=None, base=BASE, deal=DEAL):
    """
    Change ONE parameter over `values` (default: GRID[param]),
    keep the others at `base`. Returns a DataFrame (rows = tested values).
    """
    if values is None:
        values = GRID[param]
 
    rows = {}
    for v in values:
        params = dict(base)          # copy the base case
        params[param] = v            # change only this one
        rows[v] = summarize(run_model(Z, deal=deal, **params), deal)
 
    table = pd.DataFrame(rows).T
    table.index.name = param
    return table
 
 
def run_all_sensitivities(Z, grid=GRID, base=BASE, deal=DEAL):
    """run_sensitivity for every parameter in grid. Returns {param: DataFrame}."""
    return {param: run_sensitivity(Z, param, values, base, deal)
            for param, values in grid.items()}
 
 
def run_stress(Z, pd_values=(0.04, 0.08), rho_values=(0.2, 0.5), base=BASE, deal=DEAL):
    """
    Change PD and rho TOGETHER (a recession raises both).
    Returns a DataFrame indexed by (pd_annual, rho).
    """
    rows = {}
    for pd_ in pd_values:
        for rho_ in rho_values:
            params = dict(base, pd_annual=pd_, rho=rho_)
            rows[(pd_, rho_)] = summarize(run_model(Z, deal=deal, **params), deal)
 
    table = pd.DataFrame(rows).T
    table.index.names = ["pd_annual", "rho"]
    return table
 
 
def plot_sensitivity(table, metrics=KEY_METRICS, title=None, save_path=None):
    """Line chart of selected metrics against the tested parameter values."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.5))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, metrics):
        ax.plot(table.index, table[m], marker="o")
        ax.set_xlabel(table.index.name)
        ax.set_title(m)
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    plt.show()
 
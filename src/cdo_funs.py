"""
cdo_funs.py
Cash flow functions for Mini-Project 3: CDS Risk analysis 
"""
 
import numpy as np
 
 
def bond_cash_flows(default_quarter, face=10.0, coupon_rate=0.06, LGD=0.60,
                    n_quarters=20):
    """
    Quarterly cash flows of ONE bond (in $MM).
    default_quarter: 1..n_quarters = quarter of default, 0 = no default.
 
    Rule (§5.4):
        before default   -> coupon
        default quarter  -> recovery = face * (1 - LGD), no coupon
        after default    -> 0
        no default       -> coupon every quarter, plus face at maturity
    """
    coupon = face * coupon_rate / 4
    recovery = face * (1 - LGD)
    cf = np.zeros(n_quarters)
 
    for q in range(1, n_quarters + 1):
        i = q - 1                                   # Python index starts at 0
 
        if default_quarter == 0:                    # no default
            cf[i] = coupon
            if q == n_quarters:                     # maturity: add principal
                cf[i] += face
 
        elif q < default_quarter:                   # before default
            cf[i] = coupon
 
        elif q == default_quarter:                  # default quarter
            cf[i] = recovery
 
        # after default: stays 0
 
    return cf
 
 
def all_bond_cash_flows(default_q, face=10.0, coupon_rate=0.06, LGD=0.60,
                        n_quarters=20):
    """
    Cash flows for every case and bond.
    default_q: array (n_cases, n_bonds) of default quarters (0 = no default).
    Returns an array of shape (n_cases, n_bonds, n_quarters).
    """
    n_cases, n_bonds = default_q.shape
    bond_cf = np.zeros((n_cases, n_bonds, n_quarters))
 
    for case in range(n_cases):
        for bond in range(n_bonds):
            bond_cf[case, bond] = bond_cash_flows(
                default_q[case, bond], face, coupon_rate, LGD, n_quarters)
 
    return bond_cf
 
 
def portfolio_cash_flows(bond_cf):
    """Sum across the bonds: returns shape (n_cases, n_quarters)."""
    return bond_cf.sum(axis=1)
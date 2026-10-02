import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
from scipy.optimize import brentq


def previous_coupon_date(date, preserve_month_end):
    """
    Get the previous coupon date for a given date.
    If the maturity date is a month end, then the previous coupon date should also be a month end.
    """ 

    previous = pd.Timestamp(date) - pd.DateOffset(months=6)

    if preserve_month_end:
        previous = previous + MonthEnd(0)

    return previous


def coupon_schedule(maturity, valuation_date):
    """
    Get the coupon schedule for a given maturity and valuation date.

    Parameters:
    maturity (str): The maturity date of the bond in 'YYYY-MM-DD' format.
    valuation_date (str): The valuation date in 'YYYY-MM-DD' format.
    """
    maturity = pd.Timestamp(maturity).normalize()
    valuation_date = pd.Timestamp(valuation_date).normalize()
    preserve_month_end = maturity.is_month_end

    dates = []
    coupon_date = maturity

    while coupon_date > valuation_date:
        dates.append(coupon_date)
        coupon_date = previous_coupon_date(
            coupon_date,
            preserve_month_end,
        )

    return sorted(dates)

def current_coupon_dates(maturity, valuation_date):
    """
    Get the current coupon dates for a given maturity and valuation date.

    Parameters:
    maturity (str): The maturity date of the bond in 'YYYY-MM-DD' format.
    valuation_date (str): The valuation date in 'YYYY-MM-DD'
    
    """
    maturity = pd.Timestamp(maturity).normalize()
    valuation_date = pd.Timestamp(valuation_date).normalize()
    preserve_month_end = maturity.is_month_end

    next_coupon = maturity

    while previous_coupon_date(
        next_coupon,
        preserve_month_end,
    ) > valuation_date:
        next_coupon = previous_coupon_date(
            next_coupon,
            preserve_month_end,
        )

    previous_coupon = previous_coupon_date(
        next_coupon,
        preserve_month_end,
    )

    return pd.Series({
        "Accrual Start": previous_coupon,
        "Accrual End": next_coupon,
    })

# Model functions
def svensson_zero_rate(t, theta):
    """
    Calculate the Svensson zero rate for a given time and parameters.
    
    Parameters
    ----------
    t : array-like
        Time to maturity in years.
    theta : array-like
        Parameters of the Svensson model (beta_0, beta_1, beta_2, beta_3, tau_1, tau_2).
    
    Returns
    -------
    array-like
        The calculated zero rates for the given times and parameters.
    """
    beta_0, beta_1, beta_2, beta_3, tau_1, tau_2 = theta

    x_1 = t / tau_1
    x_2 = t / tau_2

    level_slope_1 = (1-np.exp(-x_1)) / x_1
    curvature_1 = level_slope_1 - np.exp(-x_1)

    level_slope_2 = (1-np.exp(-x_2)) / x_2
    curvature_2 = level_slope_2 - np.exp(-x_2)

    return (
        beta_0
        + beta_1 * level_slope_1
        + beta_2 * curvature_1
        + beta_3 * curvature_2
    )
def model_dirty_price(t, cashflows, theta):
    """
    Calculate the model dirty price of a bond given the cashflows, times, and Svensson parameters.

    Parameters
    ----------
    t : array-like
        Time to maturity in years.
    cashflows : array-like
        Cashflows of the bond.
    theta : array-like
        Parameters of the Svensson model (beta_0, beta_1, beta_2, beta_3, tau_1, tau_2).
    
    Returns
    -------
    float
        The calculated dirty price of the bond.
    """
    zero_rates = svensson_zero_rate(t, theta)
    discount_factors = np.exp(-zero_rates * t)
    return np.sum(cashflows * discount_factors)

def pricing_residuals(theta, bonds):
    residuals = []

    for bond in bonds:
        model_price = model_dirty_price(
            t=bond["times"],
            cashflows=bond["cash_flows"],
            theta=theta,
        )

        residual = model_price - bond["dirty_price"]
        residuals.append(residual)

    return np.asarray(residuals)

def transform_theta(z,  epsilon=1e-8):
    beta1 = z[1]
    beta0 = -beta1 + epsilon + np.exp(z[0])

    return np.array([
        beta0,
        beta1,
        z[2],
        z[3],
        z[4],
        z[5],
    ])

def constrained_residuals(z, bonds_data, epsilon=1e-8):
    theta = transform_theta(z, epsilon)
    return pricing_residuals(theta, bonds_data)

def ytm_from_price(price, times, cashflows, frequency=2):
    """
    Bond-equivalent YTM with semiannual compounding.
    """
    def price_error(yield_rate):
        discount_factors = (
            1 + yield_rate / frequency
        ) ** (-frequency * times)

        calculated_price = np.sum(cashflows * discount_factors)
        return calculated_price - price

    return brentq(
        price_error,
        a=-0.99,
        b=1.00,
    )


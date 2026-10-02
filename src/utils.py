# libraries
import numpy as np


def treasury_form_2_decimal(x):
    """
    Change the treasury form of a bond to decimal form
    for the first two decimals divide by 32
    for the next decimal divide by 256

    98.086 is 98 + 08/32 + 6/256
    """

    # first divide price on cents, thousandths and units

    units = np.trunc(x)
    decimals = x - units
    cents = np.trunc(decimals * 100)
    thousandths = np.trunc((decimals * 100 - cents) * 10)

    # convert to decimal
    decimal_price = units + cents / 32 + thousandths / 256
    return decimal_price



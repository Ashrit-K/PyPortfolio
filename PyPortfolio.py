import datetime as dt
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader
import scipy


def max_drawdown(return_series: pd.Series, start_value=1):
    """Computes the maximum drawdown for a given return series. Maximum drawdown for a
    time period t is the difference between the value of dollar index and cumulative
    maximum of the dollar index untill time t expressed as a percentage of the
    cumulative maximum of the dollar index untill time t

    see here for more: https://en.wikipedia.org/wiki/Drawdown_(economics)

    Parameters
    ----------
    return_series : pd.Series
        The series of price returns
    start_value : int, optional
        Starting value for computing the dollar index, by default 1

    Returns
    -------
    pd.DataFrame
        Dataframe with the dollar index, cumulative max in dollar index, and the max drawdown
    """

    dollar_index = start_value * (1+return_series).cumprod()
    prev_peak = dollar_index.cummax()
    drawdown = (dollar_index - prev_peak)/prev_peak
    return pd.DataFrame({
        'dollar_index': dollar_index,
        'cumulative_max': prev_peak,
        'max_drawdown': drawdown
    })


def semi_deviation(return_series, periodicity):
    """Computes the anualized semi-deviation for a given return series.
       Semi-deviation is calculated as the standard deviation of returns
       that are less tha 0.

       Parameters
       ----------
       return_series : pd.Series, pd.Dataframe
           Periodic returns for an asset or portfolio
       periodicity : str
           Periodicity of the returns. This will be used to compute annualized
           semi-deviation.

       Returns
       -------
       float
           Annulized semi-devition of the return series.

       Raises
       ------
       ValueError
           If periodicity is not 'M' or 'W' or 'D' or 'Y'.
       """

    # if the function is called on a pd.Dataframe with returns,
    # a recursive call is made to the semi_deviation function
    # using the pandas aggragate method. For more on pandas aggregate see:
    # https://www.w3resource.com/pandas/dataframe/dataframe-aggregate.php
    if isinstance(return_series, pd.DataFrame):
        return return_series.aggregate(semi_deviation)

    if periodicity == 'D':
        scale_factor = math.sqrt(252)
    elif periodicity == 'W':
        scale_factor = math.sqrt(52)
    elif periodicity == 'M':
        scale_factor = math.sqrt(12)
    elif periodicity == 'Y':
        scale_factor = 1
    else:
        raise ValueError('Invalid periodicity')

    negative_mask = return_series < 0
    semi_deviation_exit = return_series[negative_mask].std() * scale_factor
    return semi_deviation_exit

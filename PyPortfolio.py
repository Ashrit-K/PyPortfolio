import datetime as dt

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

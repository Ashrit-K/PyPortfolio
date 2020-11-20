import scipy
import pandas_datareader
import pandas as pd
import numpy as np
import datetime as dt
import math
import matplotlib.pyplot as plt
plt.style.use('seaborn')


def max_drawdown(return_series: pd.Series, start_value=1):
    """Computes the maximum drawdown for a given return series. Maximum drawdown for a
    time period t is the difference between the value of dollar index and cumulative
    maximum of the dollar index untill time t expressed as a percentage of the
    cumulative maximum of the dollar index untill time t

    for more: https://en.wikipedia.org/wiki/Drawdown_(economics)

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
    exit_df = pd.DataFrame({'dollar_index': dollar_index,
                            'cumulative_max': prev_peak,
                            'drawdown': drawdown
                            })
    return exit_df


def volatilty_scaling_helper(return_periodicity: str):
    """Checks for the periodicity of the returns and returns the appropriate
    scale factor for volatility. Annual volatility is calculated as 
    the product of volatility over time t and sqrt t.

    Parameters
    ----------
    return_periodicity : str
        The periodicity of the returns. 
        Supported periodicity: 'D' (t=252), 'W' (t=52),
                                 'M' (t=12), 'Y' (t=1)


    Returns
    -------
    float
        Scale factor for computing annual volatility from volatility
        computed for time periods < 1 year

    Raises
    ------
    ValueError
        If the return_periodicity is not in the list of specified periods
    """
    if return_periodicity == 'D':
        scale_factor = math.sqrt(252)
    elif return_periodicity == 'W':
        scale_factor = math.sqrt(52)
    elif return_periodicity == 'M':
        scale_factor = math.sqrt(12)
    elif return_periodicity == 'Y':
        scale_factor = 1
    else:
        raise ValueError('Invalid periodicity')

    return scale_factor


def return_annualizing_helper(return_periodicity: str, num_periods: float):
    """Returns the appropriate power term to raise the (1+r) term to compute
    annulized returns. 

    Usage: invoke this function to compute the appropriate exponent for the
    given (1+periodic_return) term.

    Parameters
    ----------
    rreturn_periodicity : str
        The periodicity of the returns. 
        Supported periodicity: 'D' (t=252), 'W' (t=52),
                                 'M' (t=12), 'Y' (t=1)
    num_periods : float
        Total number of periodic observations in the return series for an asset or portfolio

    Returns
    -------
    float   
        Returns the appropriate exponent for the (1+periodic_return) term.

    Raises
    ------
    ValueError
        If the return_periodicity is not in the list of specified periods
    """

    if return_periodicity == 'D':
        scale_factor = 252/num_periods
    elif return_periodicity == 'W':
        scale_factor = 52/num_periods
    elif return_periodicity == 'M':
        scale_factor = 12/num_periods
    elif return_periodicity == 'Y':
        scale_factor = 1/num_periods
    else:
        raise ValueError('Invalid periodicity')

    return scale_factor


def semi_deviation(return_series, periodicity):
    """Computes the anualized semi-deviation for a given return series.
       Semi-deviation is calculated as the standard deviation of returns
       that are less tha 0.

       Parameters
       ----------
       return_series : pd.Series, pd.Dataframe
           Periodic returns for an asset or portfolio
       periodicity : str
           Periodicity of the returns. 
           Supported periodicity: 'D' (t=252), 'W' (t=52),
                                     'M' (t=12), 'Y' (t=1)

       Returns
       -------
       float
           Annulized semi-devition of the return series.

       """

    # if the function is called on a pd.Dataframe with returns,
    # a recursive call is made to the semi_deviation function
    # using the pandas aggragate method. For more on pandas aggregate see:
    # https://www.w3resource.com/pandas/dataframe/dataframe-aggregate.php
    if isinstance(return_series, pd.DataFrame):
        return return_series.aggregate(semi_deviation)

    # invoke helper func to get scaling factor
    scale_factor = volatilty_scaling_helper(return_periodicity=periodicity)

    negative_mask = return_series < 0
    semi_deviation_exit = return_series[negative_mask].std() * scale_factor
    return semi_deviation_exit


def annualized_volatility(return_series, periodicity):
    periodic_vol = return_series.std()
    scale_factor = volatilty_scaling_helper(return_periodicity=periodicity)
    return periodic_vol * scale_factor


def annualized_return(return_series, periodicity):
    compunded_ret = (1+return_series).cumprod()
    annualizing_exponent = return_annualizing_helper(return_periodicity=periodicity,
                                                     num_periods=return_series.shape[0])

    return (compunded_ret ** annualizing_exponent) - 1


def sharpe_ratio(return_series, periodicity, risk_free_rates):
    try:  # do i need a try block?
        excess_returns = return_series - risk_free_rates
        annualized_excess_returns = annualized_return(return_series=excess_returns,
                                                      periodicity=periodicity)
        annual_vol = annualized_volatility(return_series=return_series,
                                           periodicity=periodicity)

        return annualized_excess_returns/annual_vol
    except:
        pass


def sortino_ratio(return_series, periodicity, risk_free_rates):
    excess_returns = return_series - risk_free_rates
    annualized_excess_returns = annualized_return(return_series=excess_returns,
                                                  periodicity=periodicity)
    semi_dev = semi_deviation(return_series=return_series,
                              periodicity=periodicity)
    return annualized_excess_returns/semi_dev


def calmar_ratio(return_series, periodicity, risk_free_rates):
    pass

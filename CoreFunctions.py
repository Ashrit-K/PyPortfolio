import scipy.stats
import pandas_datareader
import pandas as pd
import numpy as np
import datetime as dt
import math
import matplotlib.pyplot as plt
plt.style.use('seaborn')


def dollar_index(return_series, start_value=1):
    return start_value * (1+return_series).cumprod()


def Drawdown(return_series: pd.Series, start_value=1):
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

    wealth_index = (dollar_index(return_series=return_series,
                                 start_value=start_value))
    prev_peak = wealth_index.cummax()
    return (wealth_index - prev_peak)/prev_peak


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

    # invoke helper func to get scaling factor
    scale_factor = volatilty_scaling_helper(return_periodicity=periodicity)

    negative_return_mask = return_series < 0
    return return_series[negative_return_mask].std() * scale_factor


def annualized_volatility(return_series, periodicity):
    """computes annulized volatility for a periodic return series. 
    See documentation for volatility_scaling_helper() for details on 
    time period scaling

    Parameters
    ----------
    return_series : pd.Series, pd.DataFrame
        Periodic returns for an asset or portfolio
    periodicity : str
           Periodicity of the returns.
           Supported periodicity: 'D' (t=252), 'W' (t=52),
                                     'M' (t=12), 'Y' (t=1)

    Returns
    -------
    float   
        annualized volatility of the periodic return series
    """
    periodic_vol = return_series.std()
    scale_factor = volatilty_scaling_helper(return_periodicity=periodicity)
    return periodic_vol * scale_factor


def annualized_return(return_series, periodicity):
    """computes annulized returns for a periodic return series. 
    See documentation for return_annualizing_helper() for details on 
    time period scaling

    Parameters
    ----------
    return_series : pd.Series, pd.DataFrame
        Periodic returns for an asset or portfolio
    periodicity : str
           Periodicity of the returns.
           Supported periodicity: 'D' (t=252), 'W' (t=52),
                                     'M' (t=12), 'Y' (t=1)

    Returns
    -------
    float   
        annualized returns of the periodic return series
    """
    compunded_ret = (1+return_series).prod()
    annualizing_exponent = return_annualizing_helper(return_periodicity=periodicity,
                                                     num_periods=return_series.shape[0])

    return (compunded_ret ** annualizing_exponent) - 1


def sharpe_ratio(return_series, periodicity, risk_free_rates=None):
    """Computes annualized sharpe ratio for a given periodic return series.
    Sharpe ratio is computed as the ratio of the excess returns generated by an 
    asset or a portfolio over the volatility for the asset or portfolio

    Parameters
    ----------
    return_series : pd.Series, pd.DataFrame
        Periodic returns for an asset or a portfolio
    periodicity : str
           Periodicity of the returns.
           Supported periodicity: 'D' (t=252), 'W' (t=52),
                                     'M' (t=12), 'Y' (t=1)
    risk_free_rates : pd.Series, pd.DataFrame, optional
        Applicable risk-free rates for the asset or the portfolio, by default None

    Returns
    -------
    float   
        Annualized Sharpe Ratio of the return series
    """

    if risk_free_rates == None:
        risk_free_rates = 0

    excess_returns = return_series - risk_free_rates
    annualized_excess_returns = annualized_return(return_series=excess_returns,
                                                  periodicity=periodicity)
    annual_vol = annualized_volatility(return_series=return_series,
                                       periodicity=periodicity)

    return annualized_excess_returns/annual_vol


def sortino_ratio(return_series, periodicity, risk_free_rates=None):
    """Computes the annulized Sortino Ratio for a given periodic return series.
    Sortino ratio is computed as the ratio of the excess returns over a risk-free 
    rate to the semi-devition

    Parameters
    ----------
    return_series : pd.Series, pd.DataFrame
        Periodic returns for an asset or a portfolio
    periodicity : str
           Periodicity of the returns.
           Supported periodicity: 'D' (t=252), 'W' (t=52),
                                     'M' (t=12), 'Y' (t=1)
    risk_free_rates : pd.Series, pd.DataFrame, optional
        Applicable risk-free rates for the asset or the portfolio, by default None

    Returns
    -------
    float
        Annulized Sortino Ratio for the return series
    """

    if risk_free_rates == None:
        risk_free_rates = 0

    excess_returns = return_series - risk_free_rates
    annualized_excess_returns = annualized_return(return_series=excess_returns,
                                                  periodicity=periodicity)
    semi_dev = semi_deviation(return_series=return_series,
                              periodicity=periodicity)
    return annualized_excess_returns/semi_dev


def calmar_ratio(return_series, periodicity, risk_free_rates=None):
    raise NotImplementedError
    # ? do I need this ?


def historic_VaR(return_series, level):
    """Computes the historic Value at Risk for the time period of the return series at the 
    given level. Historic VaR is the minimum expected loss over the periodicity of the return
    series with a probability of 'level'. Returns the VaR as a positive value

    Parameters
    ----------
    return_series : pd.Series, pd.DataFrame
        Periodic returns for an asset or a portfolio
    level : float
        Probability level to compute the historic VaR

    Returns
    -------
    float, pd.Series
        Minimun expected loss in the periodicity of the return series 
        with a probability of 'level' (VaR)
    """

    return -1 * return_series.quantile(level)


def Gaussian_VaR(return_series, level):
    """Computes the Parametric Value at Risk for the time period of the return series at the 
    given level assuming returns follow Gaussian Distribution. Guassian VaR is the minimum 
    expected loss (assuming Gaussian Distribution) over the periodicity of the return series
     with a probability of 'level'. Returns the VaR as a positive value

    Parameters
    ----------
    return_series : pd.Series, pd.DataFrame
        Periodic returns for an asset or a portfolio
    level : float
        Probability level to compute the historic VaR

    Returns
    -------
    float, pd.Series
        Minimun expected loss in the periodicity of the return series 
        with a probability of 'level' (VaR) assuming returns are Gaussian
    """
    return -1 * (return_series.mean() + scipy.stats.norm.ppf(q=level) * return_series.std())

    # todo consider adding Cornish-Fisher Var
    '''
    Using the below method outputs the VaR in an unlabeled array
    while the above method outputs the VaR as a pd.Series
    
    return -1 * scipy.stats.norm.ppf(q=level,
                                     loc=return_series.mean(),
                                     scale=return_series.std()) '''


def conditional_VaR(return_series, level, VaR_method=None):
    """Computes the conditional VaR of the return series for the periodicity 
    with a probability of 'level' using the passed in VaR_method

    Parameters
    ----------
    return_series : pd.Series, pd.DataFrame
        Periodic returns for an asset or a portfolio
    level : float
        Probability level to compute the conditional VaR
    VaR_method : str
        Method to estimate VaR of the return series
        Valid options:
        - 'Guassian'
        - 'Historic' 
        - 'None' : Returns a dataframe with conditional var estimated using the above 2 methods


    Returns
    -------
    float, pd.Dataframe
        Average of the expected loss in the periodicity of the return series 
        with a probability greatear than or equal to 'level' (VaR)
    """
    if VaR_method == 'Gaussian':
        var_normal = Gaussian_VaR(return_series=return_series,
                                  level=level)
        bool_mask = return_series < -var_normal
        return -return_series[bool_mask].mean()

    elif VaR_method == 'Historic':
        hist_var = historic_VaR(return_series=return_series,
                                level=level)
        bool_mask = return_series < -hist_var
        return -return_series[bool_mask].mean()
    else:
        print('Warning:\nVaR Method not provided or not valid; returning pd.DataFrame with cVaR using Guassian & Historic method')
        var_normal = Gaussian_VaR(return_series=return_series,
                                  level=level)
        bool_mask1 = return_series < -var_normal
        normal = -return_series[bool_mask1].mean()

        hist_var = historic_VaR(return_series=return_series,
                                level=level)
        bool_mask2 = return_series < -hist_var
        hist = -return_series[bool_mask2].mean()

        return pd.DataFrame({'Gaussian Conditional VaR': normal,
                             'Historic Conditional VaR': hist})


def portfolio_returns(return_series, weights: np.array):
    pr = weights @ return_series.transpose()
    return pr


def portfolio_volatility(return_series, weights):
    pvar = weights @ return_series.cov @ weights
    return math.sqrt(pvar)

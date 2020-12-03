import scipy.stats
import scipy.optimize
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


def semi_deviation(return_series, periodicity, direction=None):
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
       direction : str
            valid options:
            - 'up' : positive semi-deviation
            - 'down' : negative semi-deviation
            - 'None' : Default. Returns a pd.DataFrame with positive and negative
                        semi-deviation


       Returns
       -------
       float, pd.Dataframe
           Annulized semi-devition of the return series.

       """

    # if the function is called on a pd.Dataframe with returns,
    # a recursive call is made to the semi_deviation function
    # using the pandas aggragate method. For more on pandas aggregate see:
    # https://www.w3resource.com/pandas/dataframe/dataframe-aggregate.php

    # invoke helper func to get scaling factor
    scale_factor = volatilty_scaling_helper(return_periodicity=periodicity)

    if direction == 'up':
        return_mask = return_series > 0
        return return_series[return_mask].std() * scale_factor

    elif direction == 'down':
        return_mask = return_series < 0
        return return_series[return_mask].std() * scale_factor

    else:
        up_return_mask = return_series > 0
        up = return_series[up_return_mask].std() * scale_factor

        down_return_mask = return_series < 0
        down = return_series[down_return_mask].std() * scale_factor

        return pd.DataFrame({'Positive semi-deviation': [up],
                             'Negative semi-deviation': [down]})


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
                              periodicity=periodicity,
                              direction='down')
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
        - 'None' : Returns a dataframe with conditional
                 VaR estimated using the above 2 methods


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
        print('Warning:\nVaR Method not provided or not valid;\
             returning pd.DataFrame with cVaR using Guassian & Historic method')
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


def portfolio_returns(weights: np.array, return_series):
    """Computes the periodic portfolio returns
    for a given return series and weights

    Parameters
    ----------
    return_series : [type]
        [description]
    weights : np.array
        Portfolio weights in the same order as the returns
        are arranged for assets in the return_series

    Returns
    -------
    pd.Series
        return series of the portfolio
    """
    pr = weights @ return_series.transpose()
    return pr


def portfolio_volatility(weights: np.array, return_series):
    """Computes the periodic volatility of the portfolio
    with given weights and the historical covariance
    matrix for the return_series

    Parameters
    ----------
    return_series : pd.Series, pd.DataFrame
        [description]
    weights : [type]
        [description]

    Returns
    -------
    float
        periodic portfolio volatility.

        # ? For the future clueless me:
        by periodic volatility I mean volatility for the return series over
        the periodicity of the passed in return series. i.e, if return_series is
        monthly, then the returned portfolio volatilty will be monthly.
    """
    pvar = weights.transpose() @ return_series.cov() @ weights
    return math.sqrt(pvar)


def minimum_volatility_weights(annualized_target_return: float,
                               return_series: pd.DataFrame,
                               allow_shorts=False):
    """Returns the portfolio weights that minimimzes the portfolio volatility for a
    given level of annualized return

    Parameters
    ----------
    annualized_target_return : float
        the level of return for which portfolio volatility will be minimized
    return_series : pd.DataFrame
        returns for the set of assets in a portfolio. Expects assets to be organized in
        columns and periodic returns along the rows.

    allow_shorts : boolean
        True will set the weight bounds to (-1,1)
        False will set the weight bounds to (0,1). Defualt mode for the optimization

    Returns
    -------
    tuple
        a tuple with the status of the optimization and the optimal weights
    """

    num_of_assets = return_series.shape[1]

    # * set bounds for the weights for all assets
    weight_bounds = [(0, 1)] * num_of_assets
    if allow_shorts:
        weight_bounds = [(-1, 1)] * num_of_assets

    # * set bounds for the weights for all assets
    weight_bounds = [(0, 1)] * num_of_assets

    # * set initial guess to start the optimization
    # todo add option to allow shorts and leverage
    initial_weights = np.repeat(a=1/num_of_assets, repeats=num_of_assets)

    # * define weight constraint
    # todo add option to allow shorts and leverage
    weights_add_one = {
        'type': 'eq',
        'args': (),
        'fun': lambda weights: np.sum(weights) - 1
    }

    def helper_pret(weights, return_series):
        '''returns annualized portfolio returns for given weights'''

        return annualized_return(return_series=portfolio_returns(weights=weights,
                                                                 return_series=return_series),
                                 periodicity='M')

    target_return_eq_pret = {
        'type': 'eq',
        'args': (return_series, annualized_target_return, ),
        'fun': lambda weights, return_series, annualized_target_return:
                abs(helper_pret(weights, return_series) - annualized_target_return)
    }

    # * invoke the optimizer
    optimal_weights = scipy.optimize.minimize(fun=portfolio_volatility,
                                              x0=initial_weights,
                                              args=(return_series,),
                                              constraints=[weights_add_one,
                                                           target_return_eq_pret],
                                              bounds=weight_bounds,
                                              method='SLSQP')

    return (optimal_weights.success, optimal_weights.x)


def EfficientFrontier(return_series, periodicity):
    """Returns a tuple containing two pd.DataFrames:
    - the first datframe consists of portfolio returns and portfolio vol along 
    efficient frontier
    - the second dataframe consists of the optimal weights for each asset for 
    a given level of portfolio returns

    Parameters
    ----------
    return_series : pd.DataFrame
        Periodic returns for the assets with which a portfolio will be constructed. 
        Note: the returns should be periodic (either M, W, D, or Y)
    periodicity : str
           Periodicity of the returns.
           Supported periodicity: 'D' (t=252), 'W' (t=52),
                                     'M' (t=12), 'Y' (t=1)

    Returns
    -------
    tuple
        tuple with 2 df; see above description
    """

    min_return = annualized_return(return_series=return_series,
                                   periodicity=periodicity).min()

    max_return = annualized_return(return_series=return_series,
                                   periodicity=periodicity).max()

    # the min portfolio return that can be generated is by allocating 100% to the lowest
    #  returning asset, similarly the max eturn that can be generated by the portfolio
    # is by allocating 100% to the max returning asset; any allocation in between will
    # be a trade-off between vol and return
    target_return_vector = np.linspace(start=min_return,
                                       stop=max_return,
                                       num=150)

    # minimum_volatility_weights returns a tuple
    # index into the tuple to get the weights
    optimal_weights = [minimum_volatility_weights(
        x, return_series)[1] for x in target_return_vector]

    # portfolio_return returns a return series
    # use annualized_return to compute annualized portfolio return for given weights
    portfolio_return = [annualized_return(return_series=portfolio_returns(weights=w,
                                                                          return_series=return_series),
                                          periodicity=periodicity) for w in optimal_weights]

    # portfolio_volatility returns periodic volatility
    # use volatility_scaling_helper to compute annualized volatility
    portfolio_vol = [portfolio_volatility(weights=w,
                                          return_series=return_series) *
                     volatilty_scaling_helper(return_periodicity=periodicity) for w in optimal_weights]

    # recover the weights for each level of return
    weights_df = pd.DataFrame(data=optimal_weights,
                              columns=return_series.columns)
    weights_df['Portfolio Return'] = portfolio_return

    return_risk_df = pd.DataFrame({'Portfolio Return': portfolio_return,
                                   'Portfolio Volatility': portfolio_vol})

    return (return_risk_df, weights_df)

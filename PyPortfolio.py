import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader
import scipy

# ----------------------------------------------------------------------------------------------


def get_resample(returns, period='M'):
    """ Returns the last monthly resampled data by default
    """
    resampled_r = returns.resample(period).last()
    return resampled_r

# ----------------------------------------------------------------------------------------------


def get_drawdown(return_series: pd.Series):
    """
    return series --> dollar_index, prev_peak, drawdown

    """
    import pandas as pd
    """ Takes a pandas return series and spits out a pd
    series containing dollar index, cumulative peaks & drawdown pct
    Returns drawdown pct as a negative value
    """
    dollar_index = 100*(1+return_series).cumprod()
    prev_peak = dollar_index.cummax()
    drawdown = (dollar_index - prev_peak)/prev_peak
    return pd.DataFrame({
        'dollar_index': dollar_index,
        'prev_peak': prev_peak,
        'drawdown': drawdown
    })

# ----------------------------------------------------------------------------------------------


def get_returns(price_series: pd.Series):
    """takes a price series and spits out a pd series of pct chnage"""
    returns = price_series.pct_change().dropna()
    return returns

# ----------------------------------------------------------------------------------------------


def get_skew(return_series):
    """
    Returns skewness of the series.
    Alternative to scipy.stats.skew
    """
    demeaned_r = return_series - return_series.mean()
    sigma_r = return_series.std(ddof=0)
    skew = (demeaned_r**3).mean()/sigma_r**3
    return skew


# ----------------------------------------------------------------------------------------------


def get_kurtosis(returns):
    """ returns total kurtosis of the series. Kurtosis is a measure of the thickness of the tails.
    Alternative to scipy.stats.kurtosis + 3
    """
    demeaned_r = returns - returns.mean()
    numerator = (demeaned_r**4).mean()
    sigma_r = returns.std(ddof=0)
    kurtosis = numerator/sigma_r**4
    return kurtosis

# ----------------------------------------------------------------------------------------------


def is_normal(returns, level=0.05):
    """ Returns True if series is normal at a 0.05 level of significance by default
    """
    import scipy.stats
    test_statictic, p_value = scipy.stats.jarque_bera(returns)
    if isinstance(returns, pd.DataFrame):
        # add a condition to handle when series has >1 return series
        return returns.aggregate(is_normal)
    else:
        return p_value > level

# ----------------------------------------------------------------------------------------------


def get_semi_deviation(returns, return_freq='M'):
    """ Returns semi deviation for a return series
    """
    if isinstance(returns, pd.DataFrame):
        return returns.aggregate(get_semi_deviation)

    is_neg = returns < 0
    if return_freq == 'M':
        scale = 12**0.5
    return returns[is_neg].std(ddof=0) * scale

# ----------------------------------------------------------------------------------------------


def historic_var(returns, level=0.05):
    """ Compute historic VaR for a given level of significance
        VaR is the minimum expected loss over the given period with a prob.= level
        Returns VaR as positive value
    """

    if isinstance(returns, pd.DataFrame):
        return returns.aggregate(historic_var)
    elif isinstance(returns, pd.Series):
        return -1 * np.percentile(returns, level*100)
    else:
        raise TypeError('Expected a pd.DataFrame or pd.Series')


# ----------------------------------------------------------------------------------------------


def guassian_var(returns, level=0.05, modified=False):
    from scipy.stats import norm
    """ Returns a guasssian VaR at 5% level. 
    MOdifief=True returns Cornish-Fisher VaR
    """
    z = norm.ppf(level)
    if modified == True:
        s = get_skew(returns)
        k = get_kurtosis(returns)
        z = z + (z**2 - 1) * s/6 + (z**3 - 3*z) * \
            (k-3)/24 - (2 * z**3 - 5*z) * s**2/36

        return -1 * (returns.mean() + z * returns.std(ddof=0))

    else:
        return -1 * (returns.mean() + z * returns.std(ddof=0))


# ----------------------------------------------------------------------------------------------


def annualized_returns(returns, periods_in_year=12):
    """
    Returns annualized returns. 
    Assumes monthly data input by default
    """
    comp_growth = (1+returns).prod()
    period_in_series = returns.shape[0]
    ar = (comp_growth ** (periods_in_year/period_in_series)) - 1
    return ar

# ----------------------------------------------------------------------------------------------


def annualized_vol(returns, periods_in_year=12):
    """
    Returns annaulized volatility. assumes monthly data input by default
    """
    period_vol = returns.std()
    av = period_vol * periods_in_year**0.5
    return av

# ----------------------------------------------------------------------------------------------


def assset_sharpe_ratio(returns, rf):
    """
    Returns annualized sharpe ratio for a given monthly return series 
    """
    ar = annualized_returns(returns)
    av = annualized_vol(returns)
    arf = annualized_returns(rf)
    sharpe = (ar - arf)/av
    return sharpe
# ----------------------------------------------------------------------------------------------


def portfolio_return(weight_vector, annual_return_vector):
    """
    Returns annual portfolio return.
    """
    return weight_vector.T @ annual_return_vector

# ----------------------------------------------------------------------------------------------


def portfolio_vol(weight_vector, cov_matrix, period='M'):
    """ 
    Returns portfolio volatility for the period. Does NOT scale to annual volatility!
    """
    return (weight_vector.T @ cov_matrix @ weight_vector)**0.5 * 12**0.5

# ----------------------------------------------------------------------------------------------


def plot_eff2(returns_vector, no_of_points=50):
    """
    takes in monthyl returns (not annual) and draws plots an effcient frontier
    """
    if returns_vector.shape[1] != 2:
        raise TypeError('expected return series for two assets only')
    weights_vector = [np.array([w, 1-w])
                      for w in np.linspace(0, 1, no_of_points)]
    rets = [portfolio_return(w, annualized_returns(returns_vector))
            for w in weights_vector]
    vols = [portfolio_vol(w, returns_vector.cov()) for w in weights_vector]
    simulated_portfolios = pd.DataFrame({'Returns': rets, 'Volatility': vols})
    plt.style.use('seaborn-talk')
    return simulated_portfolios.plot.scatter(x='Portfolio Volatility',
                                             y='Portfolio Returns',
                                             linewidths=0.05,
                                             color='darkviolet',
                                             marker='_')

# ----------------------------------------------------------------------------------------------


def min_vol_weights(tgt_return, returns):
    """
    Returns minimum portfolio vol weights for a given level of return 
    """

    import scipy.optimize as optimize
    n = returns.shape[1]
    bounds = ((0.0, 1.0),)*n
    init_guess = np.repeat(1/n, n)

    # weight constraint
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda optimal_weights: np.sum(optimal_weights) - 1
    }

    # return constraint
    tgt_return_met = {
        'type': 'eq',
        'args': (returns,),
        'fun': lambda optimal_weights, returns: tgt_return -
        portfolio_return(optimal_weights, annualized_returns(returns))
    }

    optimal_weights = optimize.minimize(portfolio_vol, init_guess,
                                        args=(returns.cov(),), method='SLSQP',
                                        constraints=(
                                            weights_sum_to_1, tgt_return_met),
                                        options={'disp': False})
    return optimal_weights.x


# ----------------------------------------------------------------------------------------------


def max_sharpe_portfolio_weights(returns, rf=0):
    """
    Returns weights for portfolio with max sharpe ratio 
    """
    import scipy.optimize as optimize
    n = returns.shape[1]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),)*n

    weight_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }

    def neg_sharpe(wts, returns, rf):
        ar = annualized_returns(returns, 12)
        pr = portfolio_return(wts, ar)
        pv = portfolio_vol(wts, returns.cov())
        neg_sharpe = (rf - pr) / pv
        return neg_sharpe

    msr_weights = optimize.minimize(neg_sharpe, x0=init_guess,
                                    bounds=bounds, constraints=(
                                        weight_sum_to_1,),
                                    args=(returns, rf), method='SLSQP',
                                    options={'disp': False})

    # the order of args matters!!

    return msr_weights.x


# ----------------------------------------------------------------------------------------------


def min_variance_portfolio_weights(cov_matrix, rf=0):
    """
    Returns weights for portfolio with max sharpe ratio 
    """
    from scipy.optimize import minimize
    n = cov_matrix.shape[0]
    init_guess = np.repeat(0.1/n, n)
    bounds = ((0.0, 1.0),)*n

    weight_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }

    def port_var(wts, cov_matrix):
        pv = portfolio_vol(wts, cov_matrix)**2
        return pv

    min_var_weights = minimize(port_var, x0=init_guess,
                               bounds=bounds, constraints=(weight_sum_to_1,),
                               args=(cov_matrix), method='SLSQP',
                               options={'disp': False})

    # the order of args matters!!

    return min_var_weights.x


# ----------------------------------------------------------------------------------------------


def plot_eff(returns, rf=0,
             sharpe_portfolio=False, min_variance_portfolio=True, no_of_points=150):

    target_returns = np.linspace(annualized_returns(returns).min(),
                                 annualized_returns(returns).max(),
                                 num=no_of_points)

    optimal_wts = [min_vol_weights(target_return, returns)
                   for target_return in target_returns]

    ret = [portfolio_return(wt, annualized_returns(returns))
           for wt in optimal_wts]

    vol = [portfolio_vol(wt, returns.cov()) for wt in optimal_wts]

    eff_n = pd.DataFrame(
        {'Portfolio Returns': ret, 'Portfolio Volatility': vol})

    eff_n['Sharpe Ratio'] = (
        eff_n['Portfolio Returns'] - rf) / eff_n['Portfolio Volatility']

    plt.style.use('seaborn-talk')

    plot = eff_n.plot.scatter(y='Portfolio Returns', x='Portfolio Volatility',
                              c='Sharpe Ratio', marker=".", edgecolor='white',
                              cmap='PuRd', s=35, legend=True)

    plot.set_xlim(left=0)
    plot.set_ylim(bottom=0)
    plot.scatter(x=0, y=rf,
                 color='gold', marker=',', linewidths=1)

    if sharpe_portfolio:
        sharpe_wts = max_sharpe_portfolio_weights(returns, rf=rf)
        sharpe_portfolio_ret = portfolio_return(sharpe_wts,
                                                annualized_returns(returns))
        sharpe_portfolio_vol = portfolio_vol(sharpe_wts, returns.cov())

        plot.scatter(x=[sharpe_portfolio_vol],
                     y=[sharpe_portfolio_ret],
                     color='red', marker=',',
                     s=30, label='Sharpe Portfolio')
        plot.legend()

    if min_variance_portfolio:
        n = returns.shape[1]
        wt = min_variance_portfolio_weights(returns.cov(), rf=rf)
        min_var_ret = portfolio_return(wt, annualized_returns(returns))
        # returns for min vol portfolio dont make a difference
        min_var_vol = portfolio_vol(wt, returns.cov())

        plot.scatter(x=[min_var_vol], y=[min_var_ret],
                     marker=',', color='darkgreen', s=30, label='Min Variance')
        plot.legend()

    return plot

# ----------------------------------------------------------------------------------------------


def get_snp500_tickers(reload=False):
    import pickle

    import bs4 as bs
    import requests

    """
    Gets the list of tickers from the wikipedia page. 
    If reload is set to false function will return tickers from the saved pickle
    
    """

    if reload:
        response = requests.get(
            'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        ''' 
        the response variable gets the page source from the specified link
        '''

        soup = bs.BeautifulSoup(response.text, 'lxml')
        '''
        the soup variable takes in the text version of the response and turns it into a
        bs object. The second argument specifies the parser
    
        '''

        table = soup.find('table', {'class': 'wikitable sortable',
                                    'id': 'constituents'})
        '''
        the first argument finds all the tables, but the second argument limits it to the 
        tables that match the critera entered
        '''

        tickers = []

        for row in table.findAll('tr')[1:]:
            ticker = row.findAll('td')[0].text
            ticker = ticker.strip()
            ticker = ticker.replace('.', '-')
            tickers.append(ticker)

        with open('snp500ticker.pickle', 'wb') as file:
            pickle.dump(tickers, file)

    with open('snp500ticker.pickle', 'rb') as file:
        tickers = pickle.load(file)

    return tickers


# __________________________________________________________________________________________________


def get_russel_1k_tickers():
    import pickle

    import bs4 as bs
    import requests

    response = requests.get(
        'https://en.wikipedia.org/wiki/Russell_1000_Index#Components')
    soup = bs.BeautifulSoup(response.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    # don't use 'findAll' here!
    tickers = []

    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[1].text
        tickers.append(ticker.strip())

    with open('russel_1k_tickers.pickle', 'wb') as file:
        pickle.dump(tickers, file)

    return tickers


# ----------------------------------------------------------------------------------------------


def save_quandl_key(key='aYZ7xHShmuXxR9Hwkx4s'):
    '''
    Save the quandl key as a pickle for future use
    '''
    import pickle
    with open('quandl_api_key.pickle', 'wb') as file:
        pickle.dump(key, file)


# ----------------------------------------------------------------------------------------------

def get_quandl_key():
    '''
    Loads saved api key
    '''
    import pickle
    with open('quandl_api_key.pickle', 'rb') as file:
        key = pickle.load(file)
    return key

# --------------------------------------------------------------------------------------------------


def get_cppi(risky_asset_returns, risk_free_returns,
             m=3, floor=0.8,
             drawdown_constraint=None,
             start_value=100):

    import numpy as np
    import pandas as pd

    """
    Args:
    risky_asset_returns --> returns for the risky asset (duh)
    risk_free_returs --> returns for the risk-free asset (duh, again!)
    m --> multiple of the cushion that will be allocated to risky asset
        set to 3 by default
    floor --> default set to 0.8
    start_value --> value of the portfolio as the start
    -----------------------------------------------------------------------
    Result:
    - Returns a dictionary containing the following. Individual series can be called
    by using ['example output']
        * 'Portfolio with CPPI'
        * 'Portfolio Without CPPI'
        * 'Weight in the Risky Asset'
        * 'Cushion Pct'
    
    """

    # create an exception to handle if the input return series is a pd.Series
    # fisrt convert the
    if isinstance(risky_asset_returns, pd.Series):
        risky_asset_returns = pd.DataFrame(risky_asset_returns, columns='R')

    account_value = start_value
    cppi_account_value = pd.DataFrame().reindex_like(risky_asset_returns)
    cppi_account_value.columns = ['CPPI']
    weight_in_risky = pd.DataFrame().reindex_like(risky_asset_returns)
    weight_in_risky.columns = ['Risky Allocation']
    floor_value_df = pd.DataFrame().reindex_like(risky_asset_returns)
    floor_value_df.columns = ['Floor $']
    dates = risky_asset_returns.index
    n_steps = len(dates)

    for step in range(n_steps):

        '''if drawdown_constraint is not None:
            peak = np.maximum(floor, drawdown_constraint)
            floor_value = peak * account_value
        else:
            floor_value = floor * account_value'''
        floor_value = floor * account_value
        floor_value_df.iloc[step] = floor_value
        cushion_pct = (account_value - floor_value)/account_value
        risk_alloc_pct = cushion_pct * m
        risk_alloc_pct = np.maximum(0, risk_alloc_pct)  # no shorting
        risk_alloc_pct = np.minimum(1, risk_alloc_pct)  # no leverage
        rf_alloc_pct = 1 - risk_alloc_pct
        risk_alloc = risk_alloc_pct * account_value
        rf_alloc = rf_alloc_pct * account_value

        # update
        account_value = rf_alloc * (1+risk_free_returns.iloc[step]) + \
            risk_alloc * (1+risky_asset_returns.iloc[step])

        cppi_account_value.iloc[step] = account_value
        weight_in_risky.iloc[step] = risk_alloc_pct

    no_cppi = (1+risky_asset_returns).cumprod() * start_value

    result = {
        'CPPI': cppi_account_value,
        'No CPPI': no_cppi,
        'Risky Allocation': weight_in_risky,
        'Floor $': floor_value_df
    }
    return result


# ----------------------

def summary_stats(returns, rf, return_freq=12):
    """
    Args:
    Returns --> a return series
    rf ---> a return series for the rf asset
------------------------------------------------------------------
    Result:
    Returns a Df with: 
        * Annualized returns 
        * Annualized vol
        * Skewness
        * Excess Kurtosis
        * Sharpe Ratio
        * Sortino Ratio
        * Max Drawdown
        * Cornish Fisher VaR at 0.05 level
    """

    """
    Revisit and refactor sharpe ratio and sortino ration code. currently using mean rf for both
    
    """

    annual_rets = annualized_returns(returns, periods_in_year=return_freq)
    annual_vol = annualized_vol(returns, periods_in_year=return_freq)
    skewness = get_skew(return_series=returns)
    Excess_Kurtosis = get_kurtosis(returns) - 3
    rf_mean_returns = rf.mean()
    Sharpe_Ratio = (annual_rets - rf_mean_returns) / annual_vol
    Sortino_Ratio = (annual_rets - rf_mean_returns) / \
        get_semi_deviation(returns=returns)

    Max_Drawdown = returns.aggregate(
        lambda r: get_drawdown(r).drawdown.min())

    Cornish_Fisher_VaR = guassian_var(
        returns=returns, modified=True, level=0.05)

    result = pd.DataFrame({
        'Annualized Returns': annual_rets,
        'Annualized Volatility': annual_vol,
        'Skew': skewness,
        'Excess Kurtosis': Excess_Kurtosis,
        'Sharpe Ratio': Sharpe_Ratio,
        'Sortino Ratio': Sortino_Ratio,
        'Max Drawdown': Max_Drawdown,
        'Cornish Fisher Var': Cornish_Fisher_VaR
    })
    return result

# --------------------------------------------------------------------------------------------------


def gbm(mu, sigma, n_years, steps_per_year, scenarios=1000, starting_price=1):
    '''
    Returns as df with simulated prices
    ----------------------------------------------------------------------------------------
    Args:
    mu: drift component
    sigma: standard deviation
    n_years: number of years to simualte prices
    step_per_year: number of steps to simulate in one year
    scenarios: number of paths that will be simulated
    starting price: the price of the stock at the start

    '''
    n_steps = int(n_years * steps_per_year)
    dt = 1/steps_per_year
    returns_plus_1 = np.random.normal(loc=(1+mu*dt),
                                      scale=sigma*np.sqrt(dt),
                                      size=(n_steps+1, scenarios))  # add 1 so that the next step doesnt affect
    returns_plus_1[0] = 1  # for aethetics only
    prices = starting_price * pd.DataFrame(returns_plus_1).cumprod()
    return prices


# --------------------------------------------------------------------------------------------------

def show_gbm(mu, sigma, n_years=1):

    gbm = gbm(mu=mu, sigma=sigma,
              n_years=n_years, steps_per_year=252, starting_price=1)

    ax = gbm.plot(legend=False, color='violet', alpha=0.35, linewidth=0.45)
    ax.axhline(y=1, linestyle='-', color='blue', linewidth='0.75')
    ax.set_xlabel('Days')
    ax.set_ylabel('Stock Price')
    ax.set_title('Simulation of Stock Price Using GBM')
    ax.set_xlim(left=0, right=n_years*252)

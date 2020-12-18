# from CoreFunctions import *
import datetime as dt
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader
import scipy.optimize
import scipy.stats

plt.style.use('seaborn')


class RiskReturn(object):
    """RiskReturn is an object that needs a return series and the
    periodicity of the return series to be initialized.

    The object will be initialized with the following attributes.
    Attributes with any available getter and setter methods are mentioned in square braces.
    1. return_series [get]
    2. return_periodicity [get]
    3. # todo risk-free rates how to deal with excess returns ?
    4. dollar_index (value of $1 compounded) [get]
    5. dollar_index_startvalue [get, set]
    6. drawdown [get]
    7. max_drawdown [get]
    8. annualized_returns [get]
    9. annualized_volatility [get]
    10. semi_deviation (annualized) [get]
    11. sharpe_ratio [get]
    12. sortino_ratio [get]
    13. var_level [get, set]
    14. gaussian_var [get]
    15. historic_var [get]
    16. conditional_var (pd.DataFrame with historic and gaussian conditional VaR) [get]
    17. conditional_gauassian_var [get]
    18. conditional_historic_var [get]
    """

    def __init__(self, return_series, periodicity, risk_free_rates=None):

        self.return_series = return_series

        self.periodicity = periodicity

        self.RiskFreeRates = risk_free_rates

    def get_return_series(self):
        """returns the pd.Series or pd.DataFrame that was used to
        initialize the RiskReturn object.

        Returns
        -------
        pd.DataFrame or pd.Series
            periodic return series for the asset(s)
        """
        return self.return_series.copy(deep=True)

    def get_return_periodicity(self):
        """Periodicity of the return series.
        Supported periodicity: {"D": Daily, "W": Weekly,
                                 "M": Monthly, "Y": Yearly}.

        Returns
        -------
        str
            periodicity of the return series of the asset(s) or porfolio.
        """
        return self.periodicity

    def sortino_ratio(self):
        """Computes the annulized Sortino Ratio of asset periodic return series.
        Sortino ratio is computed as the ratio of the excess returns over a risk-free
        rate to the semi-devition

        Returns
        -------
        float, pd.DataFrame
            Annulized Sortino Ratio of the return series
        """

        if self.RiskFreeRates == None:
            self.RiskFreeRates = 0

        excess_returns = self.return_series - self.RiskFreeRates
        annualized_excess_returns = annualized_return(return_series=excess_returns,
                                                      periodicity=periodicity)
        semi_dev = semi_deviation(return_series=return_series,
                                  periodicity=periodicity,
                                  direction='down')

        return annualized_excess_returns/semi_dev

    @staticmethod
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
        if return_periodicity.upper() == 'D':
            scale_factor = math.sqrt(252)
        elif return_periodicity.upper() == 'W':
            scale_factor = math.sqrt(52)
        elif return_periodicity.upper() == 'M':
            scale_factor = math.sqrt(12)
        elif return_periodicity.upper() == 'Y':
            scale_factor = 1
        else:
            raise ValueError('Invalid periodicity')

        return scale_factor

    @staticmethod
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

        if return_periodicity.upper() == 'D':
            scale_factor = 252/num_periods
        elif return_periodicity.upper() == 'W':
            scale_factor = 52/num_periods
        elif return_periodicity.upper() == 'M':
            scale_factor = 12/num_periods
        elif return_periodicity.upper() == 'Y':
            scale_factor = 1/num_periods
        else:
            raise ValueError('Invalid periodicity')

        return scale_factor

    def annualized_volatility(self):
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
        periodic_vol = self.return_series.std()
        scale_factor = volatilty_scaling_helper(
            return_periodicity=self.periodicity)
        return periodic_vol * scale_factor

    def annualized_return(self):
        """computes annulized returns for a periodic return series.
        See documentation for return_annualizing_helper() for details on
        time period scaling

        Returns
        -------
        float
            annualized returns of the periodic return series
        """
        compunded_ret = (1+self.return_series).prod()
        annualizing_exponent = return_annualizing_helper(return_periodicity=self.periodicity,
                                                         num_periods=self.return_series.shape[0])
        return (compunded_ret ** annualizing_exponent) - 1

    def sharpe_ratio(self):
        """Computes annualized sharpe ratio for a given periodic return series.
        Sharpe ratio is computed as the ratio of the excess returns generated by an
        asset or a portfolio over the volatility for the asset or portfolio

        Parameters
        ----------
        self : RiskReturn

        Returns
        -------
        float
            Annualized Sharpe Ratio of the return series
        """

        risk_free_rates = self.RiskFreeRates

        if risk_free_rates == None:
            risk_free_rates = 0

        excess_returns = self.return_series - risk_free_rates
        annualized_excess_returns = self.annualized_return()
        annual_vol = self.annualized_volatility()

        return annualized_excess_returns/annual_vol

    def semi_deviation(self, direction=None):
        """Computes the anualized semi-deviation for a given return series.
        Semi-deviation is calculated as the standard deviation of returns
        that are less tha 0.

        Parameters
        ----------
        self : RiskReturn
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

        # invoke helper func to get scaling factor
        scale_factor = RiskReturn.volatilty_scaling_helper(
            return_periodicity=self.periodicity)

        if direction == 'up':
            return_mask = self.return_series > 0
            return self.return_series[return_mask].std() * scale_factor

        elif direction == 'down':
            return_mask = self.return_series < 0
            return self.return_series[return_mask].std() * scale_factor

        else:
            up_return_mask = self.return_series > 0
            up = self.return_series[up_return_mask].std() * scale_factor

            down_return_mask = self.return_series < 0
            down = self.return_series[down_return_mask].std() * scale_factor

            return up/down

    def sortino_ratio(self):
        """Computes the annulized Sortino Ratio for a given periodic return series.
        Sortino ratio is computed as the ratio of the excess returns over a risk-free
        rate to the semi-devition

        Returns
        -------
        float
            Annulized Sortino Ratio for the return series
        """

        risk_free_rates = self.RiskFreeRates

        if risk_free_rates == None:
            risk_free_rates = 0

        excess_returns = RiskReturn(return_series=(self.return_series - risk_free_rates),
                                    periodicity=self.periodicity,
                                    risk_free_rates=self.RiskFreeRates)
        annualized_excess_returns = excess_returns.annualized_return()
        semi_dev = self.semi_deviation(direction='down')
        return annualized_excess_returns/semi_dev

    def semi_deviation_ratio(self):
        """computes the ratio of positive semi-deviation to negative semi-deviation
        for a given asset(s), portfolio.

        Returns
        -------
        float, pd.DataFrame
            returns the semi-deviation ratio see above for definition of the
            ratio for the assets or the portfolio.
        """
        return self.semi_deviation(direction='up')/self.semi_deviation(direction='down')

    def dollar_index(self, start_value=1):
        """Returns the value of a dollar if it was
        invested in the asset(s) or portfolio.

        Parameters
        ----------
        self : RiskReturn
        start_value : int, optional
            Starting value of the dollar that will be invested in the
            asset(s) or portfolio, by default 1.

        Returns
        -------
        pd.Series or pd.DataFrame
        """
        return start_value * (1+self.return_series).cumprod()

    def Drawdown(self, start_value=1):
        """Computes the maximum drawdown for a given return series. Maximum drawdown for a
        time period t is the difference between the value of dollar index and cumulative
        maximum of the dollar index untill time t expressed as a percentage of the
        cumulative maximum of the dollar index untill time t

        for more: https://en.wikipedia.org/wiki/Drawdown_(economics)

        Parameters
        ----------
        start_value : int, optional
            Starting value for computing the dollar index, by default 1

        Returns
        -------
        pd.DataFrame
            Dataframe with the dollar index, cumulative max in dollar index, and the max drawdown
        """

        wealth_index = self.dollar_index(start_value=start_value)
        prev_peak = wealth_index.cummax()
        return (wealth_index - prev_peak)/prev_peak

    def worst_drawdown(self):
        """Returns the maximum drawdown for the asset(s) or portfolio.

        Returns
        -------
        float, pd.DataFrame
        """
        dates = self.Drawdown().idxmin()
        values = -1 * self.Drawdown().min()
        return pd.DataFrame(data={'Dates': dates,
                                  "Max Drawdown": values},
                            index=self.return_series.columns)

    def skew(self):
        return self.return_series.skew()

    def excess_kurtosis(self):
        return self.return_series.kurtosis()

    def historic_VaR(self, level=0.1):
        """Computes the historic Value at Risk for the time period of the return series at the
        given level. Historic VaR is the minimum expected loss over the periodicity of the return
        series with a probability of 'level'. Returns the VaR as a positive value

        Parameters
        ----------
        self : RiskReturn
        level : float
            Probability level to compute the historic VaR, 0.1 by default.

        Returns
        -------
        float, pd.Series
            Minimun expected loss in the periodicity of the return series
            with a probability of 'level' (VaR)
        """

        return -1 * self.return_series.quantile(level)

    def Gaussian_VaR(self, level=0.1):
        """Computes the Parametric Value at Risk for the time period of the return series at the
        given level assuming returns follow Gaussian Distribution. Guassian VaR is the minimum
        expected loss (assuming Gaussian Distribution) over the periodicity of the return series
        with a probability of 'level'. Returns the VaR as a positive value

        Parameters
        ----------
        self : RiskReturn
        level : float
            Probability level to compute the historic VaR, 0.1 by default.

        Returns
        -------
        float, pd.Series
            Minimun expected loss in the periodicity of the return series
            with a probability of 'level' (VaR) assuming returns are Gaussian
        """
        return -1 * (self.return_series.mean() + scipy.stats.norm.ppf(q=level) * self.return_series.std())

        # todo consider adding Cornish-Fisher Var
        '''
        Using the below method outputs the VaR in an unlabeled array
        while the above method outputs the VaR as a pd.Series

        return -1 * scipy.stats.norm.ppf(q=level,
                                        loc=return_series.mean(),
                                        scale=return_series.std()) '''

    def conditional_VaR(self, level=0.1, VaR_method=None):
        """Computes the conditional VaR of the return series for the periodicity
        with a probability of 'level' using the passed in VaR_method

        Parameters
        ----------
        return_series : pd.Series, pd.DataFrame
            Periodic returns for an asset or a portfolio
        level : float
            Probability level to compute the conditional VaR, by default 0.1
        VaR_method : str
            Method to estimate VaR of the return series
            Valid options:
            - 'Guassian'
            - 'Historic'
            - None : Returns a dataframe with conditional
                    VaR estimated using the above 2 methods. Default option.


        Returns
        -------
        float, pd.Dataframe
            Average of the expected loss in the periodicity of the return series
            with a probability greatear than or equal to 'level' (VaR)
        """

        if VaR_method == None:
            print('VaR Method not provided or not valid; returning pd.DataFrame with cVaR using Guassian & Historic method')

            # make a recursive call to conditional_VaR function with direction arg
            conditional_var_normal = self.conditional_VaR(level=level,
                                                          VaR_method='Gaussian')
            conditional_hist_var = self.conditional_VaR(level=level,
                                                        VaR_method='Historic')

            return pd.DataFrame({'Gaussian Conditional VaR': conditional_var_normal,
                                 'Historic Conditional VaR': conditional_hist_var})

        elif VaR_method.lower() == 'gaussian':
            var_normal = self.Gaussian_VaR(level=level)
            bool_mask = self.return_series < -var_normal
            return -self.return_series[bool_mask].mean()

        elif VaR_method.lower() == 'historic':
            hist_var = self.historic_VaR(level=level)
            bool_mask = self.return_series < -hist_var
            return -self.return_series[bool_mask].mean()

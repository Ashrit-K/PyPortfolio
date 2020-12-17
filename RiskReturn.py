from CoreFunctions import *


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

        self.dollar_index_startvalue = 1

        # todo option to compute the dollar_index for a slice of time
        self.dollar_index = dollar_index(return_series=return_series,
                                         start_value=self.dollar_index_startvalue)

        self.Drawdown = (
            self.dollar_index - (self.dollar_index).cummax())/(self.dollar_index).cummax()

        self.MaxDrawdown = min(self.Drawdown)

        self.AnnualizedReturns = annualized_return(return_series=self.return_series,
                                                   periodicity=self.periodicity)

        self.AnnaulizedVolatility = annualized_volatility(return_series=self.return_series,
                                                          periodicity=self.periodicity)

        self.SharpeRatio = sharpe_ratio(return_series=self.return_series,
                                        periodicity=self.periodicity,
                                        risk_free_rates=0)

        self.SemiDeviation = semi_deviation(return_series=self.return_series,
                                            periodicity=self.periodicity,
                                            direction=None)

        self.PositiveSemiDeviation = self.SemiDeviation['Positive semi-deviation']

        self.NegativeSemiDeviation = self.SemiDeviation['Negative semi-deviation']

        self.semideviationratio = self.PositiveSemiDeviation/self.NegativeSemiDeviation

        if isinstance(self.return_series, pd.Series):
            self.skew = pd.Series(data=scipy.stats.skew(self.return_series),
                                  index=[self.return_series.name],
                                  name="Skew")

            self.excesskurtosis = pd.Series(data=scipy.stats.kurtosis(self.return_series),
                                            index=[self.return_series.name],
                                            name="Kurtosis")

        if isinstance(self.return_series, pd.DataFrame):
            self.skew = pd.DataFrame(data=scipy.stats.skew(self.return_series),
                                     columns=['Skew'],
                                     index=self.return_series.columns)

            self.excesskurtosis = pd.DataFrame(data=scipy.stats.kurtosis(self.return_series),
                                               columns=['Excess Kurtosis'],
                                               index=self.return_series.columns)

        self.SortinoRatio = sortino_ratio(return_series=self.return_series,
                                          periodicity=self.periodicity,
                                          risk_free_rates=risk_free_rates)

        self.var_level = 0.1

        self.GaussianVaR = Gaussian_VaR(return_series=return_series,
                                        level=self.var_level)

        self.HistoricVaR = historic_VaR(return_series=return_series,
                                        level=self.var_level)

        # todo index is set to 0 when passing scalars
        self.ConditionalVaR = pd.DataFrame(data={'Historic Conditional VaR': [self.HistoricVaR],
                                                 'Guassian Conditional VaR': [self.GaussianVaR]},
                                           index=None)

        self.ConditionalHistoricVaR = self.return_series[self.return_series < -
                                                         self.HistoricVaR].mean()

        self.ConditionalGuassianVaR = self.return_series[self.return_series < -
                                                         self.GaussianVaR].mean()

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

    def get_annual_return(self):
        """computes the annaulized return from the periodic returns
        for the assets(s) or portfolio

        Returns
        -------
        pd.Dataframe or float
        """
        return self.AnnualizedReturns

    def get_riskfree_rates(self):
        return self.RiskFreeRates

    def get_annual_volatility(self):
        """computes the annaulized volatility from the periodic return
        volatility for the assets(s) or portfolio

        Returns
        -------
        pd.Dataframe or float
        """
        return self.AnnaulizedVolatility

    def get_sharpe_ratio(self):
        """returns the annaulized Sharpe ratio for the asset(s) or portfolio.

        Returns
        -------
        float or pd.DataFrame
        """
        return self.SharpeRatio

    # ? think about implementation of class methods like this.
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

    def get_sortino_ratio(self):
        """returns the annaulized Sharpe ratio for the asset(s) or portfolio.

        Returns
        -------
        float or pd.DataFrame
        """
        return self.SortinoRatio

    def get_drawdown(self):
        """returns the time-series of the drawdown for the asset(s) or portfolio.

        Returns
        -------
        float or pd.DataFrame
        """
        return self.Drawdown.copy(deep=True)

    def get_max_drawdown(self):
        """return the worst drawdown for the asset(s) or portfolio.

        Returns
        -------
       float or pd.DataFrame
        """
        return self.MaxDrawdown

    def get_semi_deviation(self):
        """Returns the dataframe with the positive and negative semi-deviation
        of the asset(s) or portfolio.

        Returns
        -------
        pd.DataFrame
        """
        return self.SemiDeviation.copy(deep=True)

    def get_positive_semi_deviation(self):
        """Returns the positive semi-deviation of the asset(s) or portfolio.

        Returns
        -------
        float, pd.DataFrame
        """
        return self.PositiveSemiDeviation

    def get_negative_semi_deviation(self):
        """Returns the negative semi-deviation of the asset(s) or portfolio.

        Returns
        -------
        float, pd.DataFrame
        """
        return self.NegativeSemiDeviation

    def get_semideviation_ratio(self):
        return self.semideviationratio

    def get_var_level(self):
        return self.var_level

    def get_gaussian_var(self):
        return self.GaussianVaR

    def get_historic_var(self):
        return self.HistoricVaR

    def get_conditional_var(self):
        return self.ConditionalVaR

    def get_conditional_historic_var(self):
        return self.ConditionalHistoricVaR

    def get_conditional_gaussian_var(self):
        return self.ConditionalGuassianVaR

    def get_dollar_index(self):
        """Returns the value of a dollar compounded at the
        rate of return of the asset(s) or portfolio.

        Returns
        -------
        pd.Series
        """
        return self.dollar_index

    def get_dollar_index_startvalue(self):
        return self.dollar_index_startvalue

    def set_dollar_index_startvalue(self, startvalue):
        temp = self.dollar_index_startvalue
        self.dollar_index_startvalue = startvalue
        print('Start Value changed from to {} to {}'.format(temp, startvalue))
        del temp

    def set_VaR_level(self, var_level):
        temp = self.var_level
        self.var_level = var_level
        print('VaR level changed from to {} to {}'.format(temp, var_level))
        del temp

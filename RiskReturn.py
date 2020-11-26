from CoreFunctions import *


class RiskReturn(object):
    """RiskReturn is an object that needs a return series and the 
    periodicity of the return series to be initialized. 

    The object will be initialized with all the following attributes.
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
                                            periodicity=self.periodicity)

        self.SortinoRatio = sortino_ratio(return_series=self.return_series,
                                          periodicity=self.periodicity,
                                          risk_free_rates=risk_free_rates)

        self.var_level = 0.1

        self.GaussianVaR = Gaussian_VaR(return_series=return_series,
                                        level=self.var_level)

        self.HistoricVaR = historic_VaR(return_series=return_series,
                                        level=self.var_level)

        # todo index is set to 0 when passing scalars
        self.ConditionalVaR = pd.DataFrame({'Historic Conditional VaR': [self.HistoricVaR],
                                            'Guassian Conditional VaR': [self.GaussianVaR]},
                                           index=None)

        self.ConditionalHistoricVaR = self.return_series[self.return_series < -
                                                         self.HistoricVaR].mean()

        self.ConditionalGuassianVaR = self.return_series[self.return_series < -
                                                         self.GaussianVaR].mean()

    def get_return_series(self):
        return self.return_series

    def get_return_periodicity(self):
        return self.periodicity

    def get_annual_return(self):
        return self.AnnualizedReturns

    def get_annual_volatility(self):
        return self.AnnaulizedVolatility

    def get_sharpe_ratio(self):
        return self.SharpeRatio

    def get_sortino_ratio(self):
        return self.SortinoRatio

    def get_drawdown(self):
        return self.Drawdown

    def get_max_drawdown(self):
        return self.MaxDrawdown

    def get_semi_deviation(self):
        return self.SemiDeviation

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

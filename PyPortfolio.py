from RiskReturn import *


class PyPortfolio(RiskReturn):

    def __init__(self, return_series, periodicity, weights=None):
        RiskReturn.__init__(self,
                            return_series=return_series,
                            periodicity=periodicity)

        self.weights = weights

        if self.weights == None:
            n = self.return_series.shape[1]
            self.weights = np.repeat(1/n, n)

        # returns the portfolio returns as a RiskReturn object
        self.portfolio_return = RiskReturn(return_series=portfolio_returns(return_series=self.return_series,
                                                                           weights=self.weights),
                                           periodicity=self.periodicity,
                                           risk_free_rates=self.RiskFreeRates)

        self.portfolio_volatility = portfolio_volatility(return_series=self.return_series,
                                                         weights=self.weights)

    def get_Portfolio_RiskReturn(self):
        return self.portfolio_return

    @classmethod
    def Minimum_vol_weights(cls):

        pass

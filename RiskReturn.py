from PyPortfolio import *


class RiskReturn(object):

    def __init__(self, return_series, periodicity, risk_free_rates=None):

        self.return_series = return_series

        self.periodicity = periodicity

        self.RiskFreeRates = risk_free_rates

        self.DrawdownDf = Drawdown(return_series=self.return_series)

        self.MaxDrawdown = min(self.DrawdownDf['drawdown'])

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

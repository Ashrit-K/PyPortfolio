from RiskReturn import *


class PyPortfolio(RiskReturn):

    def __init__(self, return_series, periodicity, weights):
        RiskReturn.__init__(self,
                            return_series=return_series,
                            periodicity=periodicity)

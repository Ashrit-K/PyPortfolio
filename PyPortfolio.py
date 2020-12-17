from RiskReturn import *
from CoreFunctions import *
plt.style.use('ggplot')


class PyPortfolio(RiskReturn):

    def __init__(self, return_series, periodicity, default_weights=None):

        RiskReturn.__init__(self,
                            return_series=return_series,
                            periodicity=periodicity)

        self.default_weights = default_weights

        if self.default_weights == None:
            n = self.return_series.shape[1]
            self.default_weights = np.repeat(1/n, n)

        # returns the portfolio returns as a RiskReturn object
        self.portfolio_RiskReturn = RiskReturn(return_series=portfolio_returns(return_series=self.return_series,
                                                                               weights=self.default_weights),
                                               periodicity=self.periodicity,
                                               risk_free_rates=self.RiskFreeRates)

        # ? is this necessary
        # self.portfolio_volatility = portfolio_volatility(return_series=self.return_series,
        #                                                  weights=self.default_weights)

    def get_Portfolio_RiskReturn(self):
        return self.portfolio_return

    def minimum_volatility_weights(self,
                                   annualized_target_return: float,
                                   allow_shorts=False):
        """Returns the portfolio weights that minimimzes the portfolio volatility for a
        given level of annualized return

        Parameters
        ----------
        self : PyPortfolio
        annualized_target_return : float
            the level of return for which portfolio volatility will be minimized
        allow_shorts : boolean
            True will set the weight bounds to (-1,1)
            False will set the weight bounds to (0,1). Defualt mode for the optimization

        Returns
        -------
        (str, np.array)
            a tuple with the status of the optimization and the optimal weights
        """
        return_series = self.get_return_series()
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
                    abs(helper_pret(weights, return_series) -
                        annualized_target_return)
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

    def EfficientFrontier(self):
        """Returns a tuple containing two pd.DataFrames:
        - the first datframe consists of portfolio returns and portfolio vol along 
        efficient frontier
        - the second dataframe consists of the optimal weights for each asset for 
        a given level of portfolio returns

        Parameters
        ----------
        self : PyPortfolio


        Returns
        -------
        tuple
            tuple with 2 df; see above description
        """
        return_series = self.get_return_series()
        periodicity = self.get_return_periodicity()

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
                                           num=30)

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

    def maximum_sharpe_weights(self,
                               allow_shorts=False):
        """Computes the weights for the maximum sharpe ratio portfolio

        Parameters
        ----------
        self : PyPortfolio
        allow_shorts : bool, optional, by default False.

        Returns
        -------
        (str, np.array)
            tuple containing the status of the optimization and 
            and the optimal weights that maximizes the portfolio
            sharpe ratio.
        """
        return_series = self.get_return_series()
        periodicity = self.get_return_periodicity()
        risk_free_rates = self.get_riskfree_rates()

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

        def helper_neg_sharpe(weights, return_series, periodicity, risk_free_rates):
            """returns negative sharpe ratio for the return series"""

            pret = portfolio_returns(
                weights=weights, return_series=return_series)
            return -1*sharpe_ratio(return_series=pret,
                                   periodicity=periodicity,
                                   risk_free_rates=risk_free_rates)

        # * invoke the optimizer
        optimal_weights = scipy.optimize.minimize(fun=helper_neg_sharpe,
                                                  x0=initial_weights,
                                                  args=(return_series,
                                                        periodicity,
                                                        risk_free_rates),
                                                  constraints=[
                                                      weights_add_one],
                                                  bounds=weight_bounds,
                                                  method='SLSQP')

        return (optimal_weights.success, optimal_weights.x)

    def maximum_sortino_weights(self,
                                allow_shorts=False):
        """Computes the weights for the maximum sortino ratio portfolio

        Parameters
        ----------
        self : PyPortfolio
        allow_shorts : bool, optional, by default False.

        Returns
        -------
        (str, np.array)
            tuple containing the status of the optimization and 
            and the optimal weights that maximizes the portfolio
            sortino ratio.
        """
        return_series = self.get_return_series()
        periodicity = self.get_return_periodicity()
        risk_free_rates = self.get_riskfree_rates()

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

        def helper_neg_sortino(weights, return_series, periodicity, risk_free_rates):
            """returns negative sharpe ratio for the return series"""

            pret = portfolio_returns(
                weights=weights, return_series=return_series)
            return -1*sortino_ratio(return_series=pret,
                                    periodicity=periodicity,
                                    risk_free_rates=risk_free_rates)

        # * invoke the optimizer
        optimal_weights = scipy.optimize.minimize(fun=helper_neg_sortino,
                                                  x0=initial_weights,
                                                  args=(return_series,
                                                        periodicity,
                                                        risk_free_rates),
                                                  constraints=[
                                                      weights_add_one],
                                                  bounds=weight_bounds,
                                                  method='SLSQP')

        return (optimal_weights.success, optimal_weights.x)

    def global_minimum_variance_weights(self,
                                        allow_shorts=False):
        """Computes the weights for the minimum portfolio volatility

        Parameters
        ----------
        self : PyPortfolio
        allow_shorts : bool, optional, by default False.

        Returns
        -------
        (str, np.array)
            tuple containing the status of the optimization and 
            and the optimal weights that minimizes the
            annualized portfolio volatility.
        """

        return_series = self.get_return_series()
        periodicity = self.get_return_periodicity()

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

        def pvol_helper(weights, return_series, periodicity):
            return portfolio_volatility(weights, return_series) * \
                volatilty_scaling_helper(return_periodicity=periodicity)

        # * invoke the optimizer
        optimal_weights = scipy.optimize.minimize(fun=pvol_helper,
                                                  x0=initial_weights,
                                                  args=(return_series,
                                                        periodicity,),
                                                  constraints=[
                                                      weights_add_one],
                                                  bounds=weight_bounds,
                                                  method='SLSQP')

        return (optimal_weights.success, optimal_weights.x)

    def maximum_semideviation_ratio_weights(self,
                                            allow_shorts=False):
        """Computes the weights that maximzed portfolio semi-deviation.
        Semi-deviation ratio is the ratio of positive semi-deviation to 
        negative semi-deviation.

        Parameters
        ----------
        self : PyPortfolio
        allow_shorts : bool, optional, by default False.

        Returns
        -------
        (str, np.array)
            tuple containing the status of the optimization 
            and the optimal weights that maximizes the
            semi-deviation ratio of portfolio.
        """

        return_series = self.get_return_series()
        periodicity = self.get_return_periodicity()

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

        def semidevratio_helper(weights, return_series, periodicity):
            pret = portfolio_returns(weights, return_series)

            return -1 * semi_deviation_ratio(return_series=pret,
                                             periodicity=periodicity)

        # * invoke the optimizer
        optimal_weights = scipy.optimize.minimize(fun=semidevratio_helper,
                                                  x0=initial_weights,
                                                  args=(return_series,
                                                        periodicity,),
                                                  constraints=[
                                                      weights_add_one],
                                                  bounds=weight_bounds,
                                                  method='SLSQP')

        return (optimal_weights.success, optimal_weights.x)

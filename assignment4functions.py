import pandas as pd
import numpy as np
from scipy.stats import t


def AnalyticalNormalMeasures(alpha, weights, portfolioValue, riskMeasureTimeIntervalInDay, returns):
    """
    This function determines VaR and ES of a portfolio using the parametric method
    mean/covariance via t-student with nu=4.

    :param alpha:                           significance value
    :param weights:                         portfolio weights
    :param portfolioValue:                  notional
    :param riskMeasureTimeIntervalInDay:    estimation interval in day
    :param returns:                         returns' matrix

    :return: VaR:                           Value at Risk
    :return: ES:                            Expected Shortfall
    """

    # find estimation first date
    end_date = returns.iloc[-1, 0]  # final date
    end_date = pd.to_datetime(end_date)  # convert it to datetime format
    start_date = end_date - pd.Timedelta(days=riskMeasureTimeIntervalInDay - 1)  # to go back 5 years in time
    start_date = start_date.strftime('%Y-%m-%d')

    # reduced dataset at the estimation interval
    returns = returns[(returns['Date'] >= start_date)]
    returns = returns.reset_index(drop=True)

    # substitute NaN with previous data (NaN = missing share price)
    returns = returns.ffill()

    # log returns' mean for each analyzed company
    mu_vector = returns.iloc[:, 1:].mean()

    # log returns' standard deviation for each analyzed company
    std_vector = returns.iloc[:, 1:].std()
    # var_matrix = returns.iloc[:, 1:].var()

    # portfolio mean
    mu_portfolio = np.dot(weights, mu_vector)

    # portfolio std

    std_portfolio = np.dot(weights, std_vector)

    # degrees of freedom
    nu = 4

    # standard VaR
    VaR_std = t.ppf(alpha, df=nu)

    # VaR with variance/covariance method
    VaR = mu_portfolio + std_portfolio * VaR_std

    # portfolio VaR
    var_value = portfolioValue * VaR

    # t-student probability density function
    phi_t = t.pdf(VaR_std, df=nu)

    # standard ES
    ES_std = ((nu + VaR_std ** 2) / (nu - 1)) * (phi_t / (1 - alpha))

    # ES with variance/covariance method
    ES = mu_portfolio + std_portfolio * ES_std

    # portfolio ES
    es_value = portfolioValue * ES

    return var_value, es_value

def price_to_return(Dataset):
    """
    This function computes log returns from share prices.

    :param Dataset:                           share prices

    :return returns:                          log returns
    """

    # compute the log_returns
    Dataset.iloc[:, 1:] = Dataset.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    log_returns = np.log(Dataset.iloc[:, 1:] / Dataset.iloc[:, 1:].shift(1))
    Dataset.iloc[:, 1:] = log_returns
    returns = Dataset.drop(Dataset.index[0])
    return returns
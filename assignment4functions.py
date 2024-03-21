import pandas as pd
import numpy as np
from scipy.stats import t
from scipy.stats import norm
import math
from FE_Library import yearfrac


def SliceDataFromStartDate(data, endDate, duration):
    """
    SliceDataFromStartDate takes a dataframe and cuts it above start date of computation of risk measures
    and after the end date
    :param data:
    :param endDate:
    :param duration:
    :return:
    """

    # reduce dataset up until the end date
    # endDate = pd.to_datetime(endDate)  # convert it to datetime format
    data = data[(data['Date'] <= endDate)]

    # find estimation first date
    endDate = pd.to_datetime(endDate)
    startDate = endDate - pd.Timedelta(days=duration)  # to go back 5 years in time
    startDate = startDate.strftime('%Y-%m-%d')

    # reduced dataset at the estimation interval
    data = data[(data['Date'] >= startDate)]

    # Remove the date from the returns dataFrame and transform it to a numpy array
    # data = data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    # data = data.reset_index(drop=True)

    return data


def AnalyticalNormalMeasures(alpha, weights, portfolioValue, riskMeasureTimeIntervalInDay, returns):
    """
    This function determines VaR and ES of a portfolio using the parametric method
    mean/covariance via t-student with nu=4.

    :param alpha:                           significance value
    :param weights:                         portfolio weights
    :param portfolioValue:                  notional
    :param riskMeasureTimeIntervalInDay:    estimation interval in day
    :param returns:                         returns' matrix

    :return: ES:                            Expected Shortfall
    :return: VaR:                           Value at Risk
    """
    # Remove the date from the returns dataFrame and transform it to a numpy array
    # returns = returns.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    # returns = returns.reset_index(drop=True)

    # substitute NaN with previous data (NaN = missing share price)
    returns = returns.ffill()

    # log returns' mean for each analyzed company
    mu_vector = returns.iloc[:, 1:].mean()
    cov_matrix = pd.DataFrame.cov(returns.iloc[:, 1:])

    # assume loss distribution is gaussian
    mean_loss = weights.dot(mu_vector)
    cov_loss = (weights.dot(cov_matrix)).dot(weights)

    # degrees of freedom
    nu = 4

    # standard VaR
    VaR_std = t.ppf(alpha, df=nu)

    # VaR with variance/covariance method
    VaR = riskMeasureTimeIntervalInDay*mean_loss + math.sqrt(riskMeasureTimeIntervalInDay)*cov_loss * VaR_std

    # portfolio VaR
    var_value = portfolioValue * VaR
    print('VaR:', var_value)

    # t-student probability density function
    phi_t = t.pdf(VaR_std, df=nu)

    # standard ES
    ES_std = ((nu + VaR_std ** 2) / (nu - 1)) * (phi_t / (1 - alpha))

    # ES with variance/covariance method
    ES = riskMeasureTimeIntervalInDay*mean_loss + math.sqrt(riskMeasureTimeIntervalInDay)*cov_loss * ES_std

    # portfolio ES
    es_value = portfolioValue * ES
    print('ES:', es_value)

    return es_value, var_value


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
    returns = returns.reset_index(drop=True)
    return returns


def HSMeasurements(returns, alpha, weights, portfolioValue, riskMeasureTimeIntervalInDay):
    """
    This function performs Historical Simulation measurements of Value at Risk and Expected Shortfall

    :param returns:
    :param alpha:
    :param weights:
    :param portfolioValue:
    :param riskMeasureTimeIntervalInDay:
    :return:
    """

    # "Simulate" loss distribution
    loss = - portfolioValue * returns.iloc[:, 1:].to_numpy().dot(weights)
    # simulated loss distribution using historical log-returns values
    loss.sort(kind='stable')  # sort loss distribution in increasing order
    loss = loss[::-1]  # loss distribution changed to decreasing order (worst loss first value)

    n = len(loss)  # number of historical returns used --> "size" of loss distribution
    index = math.floor(n*(1 - alpha))  # index of the loss corresponding to VaR

    VaR = math.sqrt(riskMeasureTimeIntervalInDay) * loss[index]
    print('VaR:', VaR)
    ES = math.sqrt(riskMeasureTimeIntervalInDay)*loss[1:index].mean()
    # mean value of the worst losses up to the VaR one
    print('ES:', ES)

    return ES, VaR


def bootstrapStatistical(numberOfSamplesToBootstrap, returns):
    """
    This function samples M numbers from 1 to n.

    :param numberOfSamplesToBootstrap:      number of samples = M
    :param returns:                         time series of returns of assets in the portfolio
                                            number of time observations = n
    :return: samples:                       indexes of returns that were sampled
    """
    n = len(returns)
    samples = np.array([np.random.randint(0, n-1) for _ in range(numberOfSamplesToBootstrap)])
    return samples


def WHSMeasurements(returns, alpha, lambda_P, weights, portfolioValue, riskMeasureTimeIntervalInDay):
    """
    This function determines VaR and ES of a portfolio using Weighted Historical Simulation

    :param returns:                         returns' matrix
    :param alpha:                           significance value
    :param lambda_P:                           historical index
    :param weights:                         portfolio weights
    :param portfolioValue:                  notional
    :param riskMeasureTimeIntervalInDay:    estimation interval in day
    :param returns:                         returns' matrix

    :return: ES:                            Expected Shortfall with WHS
    :return: VaR:                           Value at Risk with WHS
    """

    # observations number
    n = len(returns)

    # normalization factor
    C = (1 - lambda_P) / (1 - lambda_P ** n)

    # historical losses
    L = - portfolioValue * returns.iloc[:, 1:].to_numpy().dot(weights)

    # simulation weights
    # date = pd.to_datetime(returns.iloc[:, 0])

    # last data
    # last_date = date.iloc[-1]

    # determine the yearfractions corresponding to the (business) dates of the returns
    # yearfrac_vector = [yearfrac(data, last_date, 3) for data in date]

    # Compute the exponents for the lambda coefficients: decreasing in time
    lambdaExponent = np.arange(n-1, -1, -1)

    # compute simulation weights
    weights_sim = C * np.power(lambda_P, lambdaExponent)

    # order losses in decreasing way and the respective simulation weights
    L, weights_sim = zip(*sorted(zip(L, weights_sim), reverse=True))

    # find the index that satisfy the constraints
    # initialize index counter
    i_temp = 0
    # initialize sum simulation weights
    sum_weights_sim = 0

    # while loop to find the index of the weights sum corresponding to 1-alpha
    while sum_weights_sim <= (1 - alpha):
        sum_weights_sim += weights_sim[i_temp]
        i_temp += 1
    i_star = i_temp - 1

    # compute VaR with WHS
    VaR_WHS = math.sqrt(riskMeasureTimeIntervalInDay) * L[i_star]
    print('VaR:', VaR_WHS)

    # compute ES with WHS
    ES_WHS = (math.sqrt(riskMeasureTimeIntervalInDay) * (np.dot(weights_sim[:i_star], L[:i_star])) /
              (np.sum(weights_sim[:i_star])))
    print('ES:', ES_WHS)

    return ES_WHS, VaR_WHS


def plausibilityCheck(returns, portfolioWeights, alpha, portfolioValue, riskMeasureTimeIntervalInDay):
    """
    This function performs a plausibility check on the VaR (rule of thumb)

    :param returns:                         returns' matrix
    :param portfolioWeights:                weights of portfolio = sensitivities
    :param alpha:                           significance value
    :param portfolioValue:                  notional
    :param riskMeasureTimeIntervalInDay:    estimation interval in day
    :return: VaR:                           Value at Risk
    """
    # Remove the date from the returns dataFrame and transform it to a numpy array
    returns = returns.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    returns = returns.reset_index(drop=True)

    # Compute upper and lower quantiles of the risk factors distributions (asset returns), one for each asset
    upperQuantiles = np.quantile(returns, alpha, axis=0)  # axis = 0 to compute one quantile for each column
    lowerQuantiles = np.quantile(returns, 1 - alpha, axis=0)

    # Compute signedVaR for each risk factor
    signedVaR = portfolioWeights*(abs(upperQuantiles) + abs(lowerQuantiles))/2

    # Compute Correlation Matrix with command corrcoef
    CorrelationMatrix = np.corrcoef(returns, rowvar=False)  # rowvar = False means that each column is a variable

    # Compute VaR according to plausibility check rule of thumb
    VaR = (math.sqrt(signedVaR.transpose().dot(CorrelationMatrix).dot(signedVaR)) *
           math.sqrt(riskMeasureTimeIntervalInDay))
    VaR_value = VaR*portfolioValue  # multiply for the portfolio value in time t to get reasonable measure
    print('Plausibility check on VaR:', VaR_value)

    return VaR_value


def PrincCompAnalysis(yearlyCovariance, yearlyMeanReturns, weights, H, alpha, numberOfPrincipalComponents,
                      portfolioValue):
    """
    this function calculates the VaR and the ES via a Gaussian parametric PCA approach

    :param yearlyCovariance:
    :param yearlyMeanReturns:
    :param weights:
    :param H:
    :param alpha:
    :param numberOfPrincipalComponents:
    :param portfolioValue:
    :return: ES:
    :return: VaR:
    """
    # assume loss distribution is gaussian
    mean_loss = weights.dot(yearlyMeanReturns)
    cov_loss = (weights.dot(yearlyCovariance)).dot(weights)
    VaR_std = norm.ppf(alpha, loc=mean_loss, scale=cov_loss)
    ES_std = norm.ppf(alpha, loc=mean_loss, scale=cov_loss)/(1-alpha)

    # compute eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(yearlyCovariance)
    # order for decrescent eigenvalues
    eigenvalues, eigenvectors = zip(*sorted(zip(eigenvalues, eigenvectors.T), reverse=True))
    eigenvalues = np.diag(eigenvalues)  # make eig into a diag matrix
    eigenvectors = np.array(eigenvectors).T  # make eigenvectors be  matrix again, it was messed up cause of zip

    # new definitions
    mu_hat = np.dot(eigenvectors.T, yearlyMeanReturns)
    weights_hat = np.dot(eigenvectors.T, weights)

    # mean and covariance for reduced form portfolio
    mu_rfp = sum((weights_hat*mu_hat)[:numberOfPrincipalComponents])
    sigma_rfp = sum(((weights_hat ** 2).dot(eigenvalues))[:numberOfPrincipalComponents])
    VaR = H*mu_rfp + math.sqrt(H)*sigma_rfp * VaR_std
    VaR_value = portfolioValue * VaR
    print('VaR with PCA:', VaR_value)

    ES = H*mu_rfp + math.sqrt(H)*sigma_rfp * ES_std
    ES_value = ES * portfolioValue
    print('ES with PCA:', ES_value)

    return ES_value, VaR_value




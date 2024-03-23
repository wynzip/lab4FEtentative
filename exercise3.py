# stock price is equal to 1
# for r i use bootstrap on interest rates
# for exercise 3 consider as starting date the 15 of February 2008
# (the date in which you have the CDS and the discounts bootstrap).

import pandas as pd
import numpy as np
from FE_Library import yearfrac
from assignment4functions import blsCall

# load EUROSTOXX_Dataset
file_csv = "EUROSTOXX50_Dataset.csv"
Dataset = pd.read_csv(file_csv)

# name of columns useful for Portfolio A
columns_name = ["Date", "ISP.MI"]

# reduced dataset for Portfolio A
Dataset_new = Dataset[columns_name]

# input data
partCoeff = 0.99  # participation coefficient
notional = 3e7  # notional
settlementDate = '2008-02-19'  # today's date
number_years = 7  # expiration years for option
sigma = 0.20  # volatility
S0 = 1

# load boostrap DF values
nameFile1 = 'Bootstrap.csv'
bootstrapData = pd.read_csv(nameFile1, sep=';')
bootstrapData['Date'] = pd.to_datetime(bootstrapData['Date'])
bootstrapData.rename(columns={'Discount Factor':'DF'}, inplace=True)

# load boostrap CDS values
nameFile2 = 'CDS_bootstrap.csv'
survProbsData = pd.read_csv(nameFile2, sep=';')
survProbsData['Date'] = pd.to_datetime(survProbsData['Date'])

# Create vector of payment dates
paymentDates = np.array(['2008-02-19', '2009-02-19', '2010-02-19', '2011-02-21', '2012-02-20', '2013-02-19', '2014-02-19', '2015-02-19'])
paymentDates = pd.to_datetime(paymentDates)

# Do we have the discount value for each of these dates? If not, we should perform a linear interpolation on zero rates
# Let's obtain the zero rates
bootStartDate = bootstrapData.iloc[0, 0]  # value date of bootstrap
bootSize = len(bootstrapData['Date'])-1  # number of bootstrapped dates (excluding the first one)
yearFracBoot = np.zeros(bootSize)  # initialize yearfrac vec

# Calculating needed yearfractions
for i in range(bootSize):
    yearFracBoot[i] = yearfrac(bootStartDate, bootstrapData.iloc[i+1, 0], 3)

# Zero Rates definition
zeroRates = - np.log(bootstrapData.iloc[1:, 1])/yearFracBoot

# Year fractions for payment dates
numPayments = int(len(paymentDates)-1)
yearFracPayment = np.zeros(numPayments)
for i in range(numPayments):
    yearFracPayment[i] = yearfrac(bootStartDate, paymentDates[i+1], 3)

# Interpolation to obtain the zero rates for all the payments dates
zeroRatesPayment = np.interp(yearFracPayment, yearFracBoot, zeroRates)

# Find the discount factors for payment dates
discFactorsPayment = np.exp(-zeroRatesPayment*yearFracPayment)

##########################################################

# Monte Carlo simulation to price the Cliquet option
# Find the forward discount rates
forwardDiscFact = discFactorsPayment/np.array([1] + discFactorsPayment[:-1].tolist())  # fwd discount factors

deltaPayments = np.zeros(numPayments)
for i in range(numPayments):
    deltaPayments[i] = yearfrac(paymentDates[i], paymentDates[i+1], 3)  # yearfracs between payment dates

forwardDiscRates = -np.log(forwardDiscFact)/deltaPayments  # forward discount rates

# MC parameters
nSim = int(1e5)
#random.seed.set(3)

# Random realizations of std Gaussian r.v. to simulate underlying's GBM dynamics
gaussRV = np.random.normal(loc=0, scale=1, size=((nSim, numPayments)))  # numPayments is 7

# Initializing matrix to store simulated underlying value in time
underlyingDynamics = np.ones((nSim, numPayments + 1))
underlyingDynamics[:, 0] = S0 * underlyingDynamics[:, 0]

# Evaluating the underlying simulated dynamics: Geometric Brownian Motion
for i in range(numPayments):
    underlyingDynamics[:, i+1] = (underlyingDynamics[:, i] * np.exp((forwardDiscRates[i] - sigma**2/2)*deltaPayments[i])
                                  + sigma * np.sqrt(deltaPayments[i]) * gaussRV[:, i])

# Now that we have the stock dynamics in time, let's evaluate the option's payoff, year by year, one payoff for each sim
payoffMC = np.maximum(0, np.add(partCoeff * underlyingDynamics[:, 1:], - underlyingDynamics[:, :-1]))

# Compute discounted payoff, one for each year (one for each "call" constituting our Cliquet option), including notional
payoffDiscMC = np.mean(payoffMC, axis=0) * discFactorsPayment * notional
# compute the mean payoff of each column (so one for each year) and then discount them by the corresponding DF

# Evaluate final MC price of Cliquet option in case of no default
priceNoDefaultMC = np.sum(payoffDiscMC)
print('MC price NON-defaultable: ', priceNoDefaultMC)

# Compute default probabilites (year by year)
survProbs = survProbsData.iloc[:, 1].to_numpy()
defaultProbs = np.add(np.array([1] + survProbs[:-1].tolist()), - survProbs)

# Now we can evaluate the price of the option in case of possible defaults
priceDefaultMC = np.dot(survProbs, payoffDiscMC)
# NB: for now we're considering recovery 0
print('MC price defaultable: ', priceDefaultMC)

################################################

# Analytical price for the Cliquet option
# We consider the Cliquet option as a sum of individual Call options, each one lasting one year and starting when the
# previous one expires

# Let's price each individual call
callPricesVec = np.zeros(numPayments)
discFactorsPayment = np.insert(discFactorsPayment, 0, 1)
for i in range(numPayments):
    callPricesVec[i] = blsCall(S0/discFactorsPayment[i], S0/(partCoeff*discFactorsPayment[i]), forwardDiscRates[i], 0, sigma, deltaPayments[i])

for i in range(numPayments):
    callPricesVec[i] = blsCall(S0, S0/partCoeff, forwardDiscRates[i], 0, sigma, deltaPayments[i])

# Now let's discount the back to today's date, sum the values, multiply by the notional, and we obtain the analytical price with NO default
priceNoDefaultAN = np.sum(discFactorsPayment[1:] * callPricesVec) * notional * partCoeff
print('Analytical price NON-defaultable: ', priceNoDefaultAN)

# Now, price in defaultable case
priceDefaultAN = np.sum(discFactorsPayment[1:] * callPricesVec * survProbs) * notional * partCoeff
print('Analytical price defaultable: ', priceDefaultAN)





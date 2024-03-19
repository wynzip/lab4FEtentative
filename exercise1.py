import pandas as pd
import numpy as np
from assignment4functions import price_to_return
from assignment4functions import HSMeasurements
from assignment4functions import bootstrapStatistical
from assignment4functions import WHSMeasurements
from assignment4functions import plausibilityCheck


# load EUROSTOXX_Dataset
file_csv = "EUROSTOXX50_Dataset.csv"
Dataset = pd.read_csv(file_csv)

# name of columns useful for Portfolio A
columns_name_A = ["Date", "AXAF.PA", "SASY.PA", "TTEF.PA", "VOWG_p.DE"]

# reduced dataset for Portfolio A
Dataset_A = Dataset[columns_name_A]

# last date
end_date = '2019-03-20'

# reduced Dataset_A until 20-02-2020
Dataset_A = Dataset_A[(Dataset_A['Date'] <= end_date)]
Dataset_A = Dataset_A.reset_index(drop=True)

# substitute NaN with previous data
Dataset_A = Dataset_A.ffill()

# compute returns for Portfolio A
returns_A = price_to_return(Dataset_A.copy())

# working on portfolio A: performing HS and Bootstrap
sharesNumber = np.array([20e3, 20e3, 25e3, 10e3])  # number of shares bought of each stock - alphabetical order
todayPrices = Dataset_A[Dataset_A['Date'] == end_date]  # take the prices S(t) at value date ( = end_date)
todayPrices = todayPrices.iloc[:, 1:].to_numpy().reshape(len(sharesNumber))
# convert it to numpy array and reshape as sharesNumber (4,)

portfolioValue = (sharesNumber*todayPrices).sum()  # initial value of Portfolio
weights = sharesNumber*todayPrices/portfolioValue  # FROZEN PORTFOLIO assumption

alpha = 0.95
years = 5
days_for_year = 365
riskMeasureTimeIntervalInDay = years*days_for_year + 1

print('Var and ES with historical simulation')
VaR_HS_ptf1, ES_HS_ptf1 = HSMeasurements(returns_A, alpha, weights, portfolioValue, riskMeasureTimeIntervalInDay)

# Statistical bootstrap
numberOfSamplesToBootstrap = 200
samples = bootstrapStatistical(numberOfSamplesToBootstrap, returns_A)

returns_A_bootstrap = returns_A.iloc[samples, :]
print('Var and ES with statistical bootstrap')
VaR_bootstrap_ptf1, ES_bootstrap_ptf1 = HSMeasurements(returns_A_bootstrap, alpha, weights, portfolioValue, riskMeasureTimeIntervalInDay)

VaR_HS_ptf1_check = plausibilityCheck(returns_A, weights, alpha, portfolioValue, riskMeasureTimeIntervalInDay)
print('Plausibility check:', VaR_HS_ptf1_check)

# name of columns useful for Portfolio B
columns_name_B = ["Date", "ADSGn.DE", "AIR.PA", "BBVA.MC", "BMWG.DE", "DTEGn.DE"]

# reduced dataset for Portfolio B
Dataset_B = Dataset[columns_name_B]

# last date
end_date = '2019-03-20'

# reduced Dataset_B until 20-02-2020
Dataset_B = Dataset_B[(Dataset_B['Date'] <= end_date)]
Dataset_B = Dataset_B.reset_index(drop=True)

# substitute NaN with previous data
Dataset_B = Dataset_B.ffill()

# compute returns for Portfolio B
returns_B = price_to_return(Dataset_B.copy())

# Portfolio B weights (since we are investing 20 cents in each stock)
sharesNumber_B = np.full(5, 0.20)

todayPrices_B = Dataset_B[Dataset_B['Date'] == end_date]  # take the prices S(t) at value date ( = end_date)
todayPrices_B = todayPrices_B.iloc[:, 1:].to_numpy().reshape(len(sharesNumber_B))  # convert it to numpy array and reshape as sharesNumber (4,)

portfolioValue_B = (sharesNumber_B*todayPrices_B).sum()  # initial value of Portfolio
weights_B = sharesNumber_B*todayPrices_B/portfolioValue_B  # FROZEN PORTFOLIO assumption

# significance value
alpha_B = 0.95
# historical index
lambda_B = 0.95

# portfolio value WRONG
#portfolioValue_B = np.dot(weights_B, Dataset_B.iloc[-1, 1:])

# estimation interval
years = 5
days_for_year = 365
riskMeasureTimeIntervalInDay_B = years*days_for_year+1

# evaluate VaR and ES for Portfolio B with WHS
print('Var and ES with weighted historical simulation')
VaR_B_WHS, ES_B_WHS = WHSMeasurements(returns_B, alpha_B, lambda_B, weights_B, portfolioValue_B, riskMeasureTimeIntervalInDay_B)
print(VaR_B_WHS)
print(ES_B_WHS)

VaR_WHS_ptf2_check = plausibilityCheck(returns_B, weights_B, alpha_B, portfolioValue_B, riskMeasureTimeIntervalInDay_B)
print('Plausibility check:', VaR_WHS_ptf2_check)

# load indexes.csv
indexes = pd.read_csv('_indexes.csv')

# name of columns useful for Portfolio C from indexes.csv
columns_name_C = indexes['Ticker'].tolist()[:18]

# add "Date"
columns_name_C.insert(0, "Date")

# reduced dataset for Portfolio C
Dataset_C = Dataset[columns_name_C]

# substitute NaN with previous data
Dataset_C = Dataset_C.ffill()

# compute returns for Portfolio
return_C = price_to_return(Dataset_C.copy())


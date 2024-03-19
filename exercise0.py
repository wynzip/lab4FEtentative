import pandas as pd
import numpy as np
from assignment4functions import AnalyticalNormalMeasures
from assignment4functions import price_to_return

# load EUROSTOXX_Dataset
file_csv = "EUROSTOXX50_Dataset.csv"
Dataset = pd.read_csv(file_csv)

# name of columns useful for exercise 0
columns_name = ["Date", "ADSGn.DE", "ALVG.DE", "MUVGn.DE", "OREP.PA"]

# reduced dataset with companies to analyze
Dataset_0 = Dataset[columns_name]

# last date
end_date = '2020-02-20'

# reduced dataset until 20-02-2020
Dataset_0 = Dataset_0[(Dataset_0['Date'] <= end_date)]
Dataset_0 = Dataset_0.reset_index(drop=True)

# fill NA with previous values
Dataset_0 = Dataset_0.ffill()

# create "matrix" of log-returns
log_returns = price_to_return(Dataset_0.copy())

# portfolio weights
weights = np.full(4, 0.25)

# significance value
alpha = 0.99

# value of portfolio (€)
portfolioValue = 1.5e7  # 15000000

# estimation interval
years = 5
days_for_year = 365
riskMeasureTimeIntervalInDay = years*days_for_year+1

VaR_portfolio, ES_portfolio = AnalyticalNormalMeasures(alpha, weights, portfolioValue, riskMeasureTimeIntervalInDay,
                                                       log_returns)

print(VaR_portfolio)
print(ES_portfolio)





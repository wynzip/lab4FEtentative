import pandas as pd
import numpy as np
from assignment4functions import price_to_return

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
return_A = price_to_return(Dataset_A)

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
return_B = price_to_return(Dataset_B)

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
return_C = price_to_return(Dataset_C)

print(return_C)

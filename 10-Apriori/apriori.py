# Apriori
# Importing the libraries
import numpy as np
import pandas as pd

from apyori import apriori

dataset = pd.read_csv('sepet.csv', header = None)

print(dataset)

#%% Apriori Algorithm(We find Apyori from GitHub)
# t = []

# for i in range (7501):
#     t.append([str(dataset.values[i, j]) for j in range (20)])

# t2 = []
    
# for i in range (7501):
#     t2.append([str(dataset.values[i, j]) for j in range (len(dataset.iloc[i].dropna(axis=0)))]) 
    
txns = []  
  
for i in range (dataset.shape[0]):
    txns.append([item for item in dataset.iloc[i].dropna(axis=0)]) 


rules = apriori(txns,
                min_support=0.01,
                min_confidence=0.2,
                min_lift=3,
                min_length=2)

print(list(rules))
# df = pd.DataFrame(rules)

# print(df)


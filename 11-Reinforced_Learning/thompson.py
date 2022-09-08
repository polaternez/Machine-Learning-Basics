# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


dataset = pd.read_csv('Ads_CTR_Optimization.csv')

#%% Random Selection

# import random

# N = 10000
# n_ad = 10 
# total_reward = 0
# chosen_ads = []

# for n in range(N):
#     ad = random.randrange(n_ad)
#     chosen_ads.append(ad)
#     reward = dataset.values[n,ad] 
#     total_reward +=reward
    
    
# #plt.hist(chosen_ads)
# sns.countplot(chosen_ads)
# plt.show()

#%% Thompson Sampling
import random

N = 10000 
n_ad = 10  
n_ones = [0] * n_ad 
n_zeros = [0] * n_ad
total_reward = 0 
chosen_ads = []

for n in range(N):
   ad = 0
   max_thmpsn = 0
   for i in range(n_ad):
       rand_beta = random.betavariate(n_ones[i]+1, n_zeros[i]+1)
       if max_thmpsn < rand_beta:
           max_thmpsn = rand_beta
           ad = i

   chosen_ads.append(ad)
   reward = dataset.values[n, ad]
   if reward == 1:
       n_ones[ad] += 1
   else:
       n_zeros[ad] += 1
       
   total_reward += reward

print("Total Reward :",total_reward)

#plt.hist(chosen_ads)
sns.countplot(chosen_ads)
plt.show()


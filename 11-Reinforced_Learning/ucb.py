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

#%% Upper Confidence Bound(UCB)
# import math
# #UCB
# N = 10000 # 10.000 tıklama
# n_ad= 10  # toplam 10 ilan var
# #Ri(n)
# rewards = [0] * n_ad #ilk basta butun ilanların odulu 0
# #Ni(n)
# n_selections = [0] * n_ad #o ana kadarki tıklamalar
# total_reward = 0 # toplam odul
# chosen_ads = []

# for n in range(1,N):
#     ad = 0 #seçilen ilan
#     max_ucb = 0
#     for i in range(n_ad):
#         if(n_selections[i] > 0):
#             avg_reward = rewards[i] / n_selections[i]
#             delta = math.sqrt(3/2* math.log(n)/n_selections[i])
#             ucb = avg_reward + delta
#         else:
#             ucb = N*10
#         if max_ucb < ucb: #max'tan büyük bir ucb çıktı
#             max_ucb = ucb
#             ad = i          
    
#     chosen_ads.append(ad)
#     n_selections[ad] = n_selections[ad]+ 1
#     reward = dataset.values[n,ad] # verilerdeki n. satır = 1 ise odul 1
#     rewards[ad] = rewards[ad]+ reward
#     total_reward +=reward
    
# print('Toplam Odul:')   
# print(total_reward)

# #plt.hist(chosen_ads)
# sns.countplot(chosen_ads)
# plt.show()

#%% My practice(ucb)
import math
import random

N = 10000
n_ad = 10  
chosen_ads = []
n_selections = [0] * n_ad 
rewards = [0] * n_ad 
total_reward = 0

for n in range(N):
    ad = 0 
    max_ucb = 0
    for i in range(n_ad):
        if n_selections[i] > 0:
            avg_reward = rewards[i] / n_selections[i]
            delta = math.sqrt(3/2 * math.log(n) / n_selections[i])
            ucb = avg_reward + delta
        else:
            # ad = random.randrange(n_ad)   
            ad = i
            break
        
        if max_ucb < ucb: 
            max_ucb = ucb
            ad = i          
    
    chosen_ads.append(ad)
    n_selections[ad] += 1
    reward = dataset.values[n,ad]
    rewards[ad] += reward
    total_reward += reward
    
print('Total Reward :', total_reward)   

#plt.hist(chosen_ads)
sns.countplot(chosen_ads)
plt.show()










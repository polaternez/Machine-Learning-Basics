# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


dataset = pd.read_csv('Ads_CTR_Optimization.csv')

#%%
import random

N = 10000
n_ad = 10 
totel_reward = 0
chosen_ads = []

for n in range(N):
    ad = random.randrange(n_ad)
    chosen_ads.append(ad)
    reward = dataset.values[n, ad] 
    totel_reward += reward
    
    
#plt.hist(chosen_ads)
sns.countplot(chosen_ads)
plt.show()











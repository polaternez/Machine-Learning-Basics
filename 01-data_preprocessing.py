import  numpy as np
import  pandas as pd
import  matplotlib.pyplot as plt


dataset = pd.read_csv("datasets/eksikveriler.csv")

print(dataset)
print("Null values:\n", dataset.isnull().sum())

# boy = dataset["boy"]
# print(boy)

# boykilo = dataset[["boy", "kilo"]]
# print(boykilo)

#%%1_Missing Values
miss_data = dataset.iloc[:, 1:4].values

print(miss_data)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(miss_data)
miss_data = imputer.transform(miss_data)

print(miss_data)

dataset.iloc[:,1:4] = miss_data

#%%2_Categorical Data Encoding

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

dataset2 = dataset.apply(LabelEncoder().fit_transform)
ulke = dataset2.iloc[:, :1]

ohe = OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()

print(ulke)

ulke = pd.DataFrame(ulke, columns=["fr", "tr", "us"])
result = pd.concat([ulke, dataset.iloc[:, 1:4], dataset2.iloc[:,4:]], axis=1)


#%%4_Train/Test Split
X = result.iloc[:, :6].values
y = result.iloc[:, 6].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33,
                                                    random_state=0)

#5_Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



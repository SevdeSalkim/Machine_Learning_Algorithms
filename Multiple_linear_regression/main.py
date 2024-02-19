##Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[: , :-1].values
y = dataset.iloc[:,-1].values

#Encoding categorical data

state = dataset.iloc[:,-2].values.reshape(-1,1)
profit = dataset.iloc[:,-1].values
first_three_columns = dataset.iloc[:, :3].values

from sklearn.preprocessing import OneHotEncoder
#Object
ohe = OneHotEncoder()
state_encoded = ohe.fit_transform(state).toarray()

#Convert dataframe
state_df = pd.DataFrame(data=state_encoded , columns=["New York","California","Florida"])
first_three_columns_df = pd.DataFrame(data=first_three_columns , columns=["R&D Spend","Administration","Marketing Spend"])
profit_df = pd.DataFrame(data=profit , columns=["Profit"])

# concat  merge dataframe
df = pd.concat([first_three_columns_df, state_df,profit_df], axis=1)

# Split train and test data

from sklearn.model_selection import train_test_split
#features
X = df.iloc[:,:-1].values
#targets
y = df.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=0)

#Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# metrics
from sklearn.metrics import  mean_squared_error, r2_score

print(f"P : {mean_squared_error(y_test, y_pred)}")

#R2 hesaplayalım 
r2 = r2_score(y_test, y_pred)
print("R2 Skoru ",r2)

















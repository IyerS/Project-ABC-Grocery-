# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 17:26:32 2024

@author: Administrator
"""
# Import packages
import pandas as pd
import pickle 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.utils import shuffle 
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold 
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFECV 


# Import the pickle file worked on
data_for_model = pickle.load(open("C:/Users/Administrator/Documents/DS-Infinity/abc_regression_modelling.p", "rb"))

# drop unnecessary column
data_for_model.drop("customer_id", axis=1, inplace=True)

# shuffle data
data_for_model = shuffle(data_for_model, random_state=42)

## Data Prep
# remove missing values
data_for_model.isna().sum()
data_for_model.dropna(how = "any", inplace=True)
data_for_model.shape

# investigage outliers and drop them
data_summary = data_for_model.describe() 
outlier_columns = ["distance_from_store","total_sales","total_items"] #seems like we have a few variables to investigate
for column in outlier_columns:
    upper_quart = data_for_model[column].quantile(0.75)
    lower_quart = data_for_model[column].quantile(0.25)
    iqr = upper_quart - lower_quart
    max_border = upper_quart + iqr*2
    min_border = lower_quart - iqr*2
    outliers = data_for_model[(data_for_model[column] > max_border) | (data_for_model[column] < min_border)]
    print(f"{len(outliers)} outliers detected in column {column}")
    data_for_model.drop(outliers.index, inplace=True)

# split input and output variables
X = data_for_model.drop("customer_loyalty_score", axis=1)
y = data_for_model["customer_loyalty_score"]

# split into train test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encoding the categorical variable from dataset - gender
categorical_var = ["gender"]
one_hot_encoder = OneHotEncoder(sparse_output=False, drop = "first")
X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_var])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_var])
encoder_feature_name = one_hot_encoder.get_feature_names_out(categorical_var)

X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoder_feature_name)
X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis=1)
X_train.drop(categorical_var, axis=1, inplace=True)

X_test_encoded = pd.DataFrame(X_test_encoded, columns=encoder_feature_name)
X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis=1)
X_test.drop(categorical_var, axis=1, inplace=True)

# Feature selection 
regressor = LinearRegression()
feature_selector = RFECV(regressor)
fit = feature_selector.fit(X_train,y_train)
optimal_feature_count = feature_selector.n_features_
feature_selection = feature_selector.get_support()
X_train = X_train.loc[:,feature_selection]
X_test = X_test.loc[:,feature_selection]

df1 = pd.DataFrame(fit.cv_results_)
df2 = df1.index + 1#tolist()
df1["number_of_features"] = df2

plt.plot(df1["number_of_features"], df1["mean_test_score"] ,marker = "o")
plt.ylabel("model score")
plt.xlabel("number of features")
plt.tight_layout()
plt.show()

# model training
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# model assesment
y_pred = regressor.predict(X_test)
r_2 = r2_score(y_test, y_pred)
print(r_2)

# cross validation
cv = KFold(n_splits = 4, shuffle=True, random_state=42)
cv_score = cross_val_score(regressor, X_train, y_train, cv=cv, scoring="r2")
cv_score.mean()

# Calculate the adjusted r2 
num_data_points, num_input_vars = X_test.shape
adjusted_r_2 = 1 - (1-r_2)*(num_data_points-1)/(num_data_points-num_input_vars-1)
print(adjusted_r_2)

# Extract Model Coefficients
coef = pd.DataFrame(regressor.coef_)
input_vars = pd.DataFrame(X_train.columns)
summary_stats_coef = pd.concat([input_vars, coef], axis=1)


# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 14:35:54 2024

@author: Administrator
"""

# Import packages
import pandas as pd
import pickle 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle 
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold 
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance


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

# for random forrest, just like decision trees, we dont investigage for outliers as they do not impact the tree logic

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

# Feature selection only helps decision tree from a computational stand point. Not needed from a modeling stand point

# model training
regressor = RandomForestRegressor(random_state = 42)
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

# feature importance
feature_importances = pd.DataFrame(regressor.feature_importances_)
feature_names = pd.DataFrame(X.columns)
feature_importances = pd.concat([feature_names, feature_importances], axis=1)
feature_importances.columns = ["input variable","feature importance"]
feature_importances.sort_values(by = "feature importance", inplace=True)

plt.barh(feature_importances["input variable"], feature_importances["feature importance"])
plt.title("Feature Importance of Random Forest")
plt.xlabel("Feature Importance")
plt.ylabel("Input Variable")
plt.show()

# permutation importance (its the decrease of model performance when particular features are randomly shuffled in and out)
result = permutation_importance(regressor, X_test, y_test, n_repeats = 10, random_state=42)
permutation_importances = pd.DataFrame(result["importances_mean"])
feature_names = pd.DataFrame(X.columns)
permutation_importances = pd.concat([feature_names, permutation_importances], axis=1)
permutation_importances.columns = ["input variable","permutation importance"]
permutation_importances.sort_values(by = "permutation importance", inplace=True)

plt.barh(permutation_importances["input variable"], permutation_importances["permutation importance"])
plt.title("permutation Importance of Random Forest")
plt.xlabel("permutation Importance")
plt.ylabel("Input Variable")
plt.show()

# Since this model turned out to be the best in accuracy to predict the customer loyalty score, lets save this model and the encoder logic to be applied to the rest of the data

import pickle

pickle.dump(regressor, open("random_forrest_regression_model_abc_loyaltyScorePredictor.p", "wb"))
pickle.dump(one_hot_encoder, open("random_forrest_regression_one_hot_encoder_abc_loyaltyScorePredictor.p", "wb"))














































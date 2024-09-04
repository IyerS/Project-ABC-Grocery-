# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 14:35:54 2024

@author: Administrator
"""

# Import packages
import pandas as pd
import pickle 
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree 
from sklearn.utils import shuffle 
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold 
from sklearn.preprocessing import OneHotEncoder


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

# for decision trees, we dont investigage outliers as they do not impact the tree logic

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
regressor = DecisionTreeRegressor(random_state = 42)
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

# is there too much fitting in the model? Run the predict on the training set and see if the accuracy is 1.0 (over fitted model)
y_pred_trained = regressor.predict(X_train)
r_2 = r2_score(y_train, y_pred_trained)
print(r_2)

# it is, let us find out the max depth that the tree must go to not over fit the training set

max_depth_list = list(range(1,10))
accuracy_score = []
for d in max_depth_list:
    regressor = DecisionTreeRegressor(max_depth=d, random_state=42)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    accuracy = r2_score(y_test, y_pred)
    accuracy_score.append(accuracy)  
score_sheet = pd.DataFrame(accuracy_score, max_depth_list).reset_index()
score_sheet.columns = ["depth","accuracy_score"]


max_accuracy = max(accuracy_score)
max_accuracy_idx = accuracy_score.index(max_accuracy)
max_accuracy_depth = max_depth_list[max_accuracy_idx]

plt.plot(max_depth_list, accuracy_score)
plt.scatter(max_accuracy_depth, max_accuracy, marker="x", color = "Red")
plt.title(f"Accuracy by Max Depth \n Optimal Tree Depth: {max_accuracy_depth} with Accuracy: {round(max_accuracy,3)}")
plt.xlabel("Max Depth of decision tree")
plt.ylabel("Accuracy scores")
plt.tight_layout()
plt.show()

# it looks like we have a max depth of 7 but after 4 the graph tapers off - so we will maybe use 4 to generalize the model a bit more
# doing model fitting onwards again with a max_depth value now = 4

# model training
regressor = DecisionTreeRegressor(max_depth=4, random_state = 42)
regressor.fit(X_train, y_train)

# model assesment
y_pred = regressor.predict(X_test)
r_2 = r2_score(y_test, y_pred)
print(r_2)

# ploting the model
plt.figure(figsize=(25,15))
tree = plot_tree(regressor,
                 feature_names = X_train.columns.tolist(),
                 filled = True, rounded = True,
                 fontsize = 24)
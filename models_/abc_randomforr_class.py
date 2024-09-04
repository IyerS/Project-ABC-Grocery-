# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 20:40:05 2024

@author: Administrator
"""

# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle 
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance


# Import the pickle file worked on
data_for_model = pd.read_pickle("C:/Users/Administrator/Documents/DS-Infinity/abc_classification_modelling.p")

# drop unnecessary column
data_for_model.drop("customer_id", axis=1, inplace=True)

# shuffle data
data_for_model = shuffle(data_for_model, random_state=42)

# investigating class balance
data_for_model["signup_flag"].value_counts(normalize=True)

## Data Prep
# remove missing values
data_for_model.isna().sum()
data_for_model.dropna(how = "any", inplace=True)

# split input and output variables
X = data_for_model.drop("signup_flag", axis=1)
y = data_for_model["signup_flag"]

# split into train test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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


# model training
clf = RandomForestClassifier(random_state=42, n_estimators=500, max_features=5) #play with max_iter
clf.fit(X_train, y_train)

# model assesment
y_pred_class = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)[:,1]
#accuracy_score(y_test, y_pred_class)
#confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_class)
print(conf_matrix)
plt.matshow(conf_matrix, cmap="coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("confusion matrix")
plt.ylabel("Actual class")
plt.xlabel("Predicted class")
for (i,j), corr_val in np.ndenumerate(conf_matrix):
    plt.text(j,i,corr_val,ha="center",va="center",fontsize=20)
plt.show()

# Accuracy (no of correct classification of total)
accuracy_score(y_test, y_pred_class)
# Precision (of all predictions made positive, how many were actually positive)
precision_score(y_test, y_pred_class)
# Recall (of all positive observations, how many were predicted as positive)
recall_score(y_test, y_pred_class)
# f1-score (harmonic mean of precision and recall)
f1_score(y_test, y_pred_class)

# feature importance (mean decrease in gini impurity score)
feature_importances = pd.DataFrame(clf.feature_importances_)
feature_names = pd.DataFrame(X.columns)
feature_importances = pd.concat([feature_names, feature_importances], axis=1)
feature_importances.columns = ["input variable","feature importance"]
feature_importances.sort_values(by = "feature importance", inplace=True)

plt.barh(feature_importances["input variable"], feature_importances["feature importance"])
plt.title("Feature Importance of Random Forest")
plt.xlabel("Feature Importance")
plt.ylabel("Input Variable")
plt.show()

# permutation importance (its the decrease of model performance when we randomize the values of each input variable)
result = permutation_importance(clf, X_test, y_test, n_repeats = 10, random_state=42)
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



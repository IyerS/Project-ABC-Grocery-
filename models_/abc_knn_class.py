# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 20:09:24 2024

@author: Administrator
"""

# Import packages
import numpy as np
import pandas as pd
import pickle 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.utils import shuffle 
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold 
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import RFECV 


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

# Feature scaling (normalizing so that data will be comparable with each other over the euclidean distances for the k neighbours)
scale_norm = MinMaxScaler() 
X_train = pd.DataFrame(scale_norm.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scale_norm.transform(X_test), columns=X_test.columns)

# Feature selection 
from sklearn.ensemble import RandomForestClassifier #using this because it has a better feature selection accuracy
clf = RandomForestClassifier(random_state=42) #play with max_iter
feature_selector = RFECV(clf)
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
clf = KNeighborsClassifier() #play with max_iter
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

# Finding the optimal value for "k"
k_list = list(range(2,25))
accuracy_score = []
for d in k_list:
    clf = KNeighborsClassifier(n_neighbors=d)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = f1_score(y_test, y_pred)
    accuracy_score.append(accuracy)  
score_sheet = pd.DataFrame(accuracy_score, k_list).reset_index()
score_sheet.columns = ["k","accuracy_score"]

max_accuracy = max(accuracy_score)
max_accuracy_idx = accuracy_score.index(max_accuracy)
max_accuracy_k = k_list[max_accuracy_idx]

plt.plot(k_list, accuracy_score)
plt.scatter(max_accuracy_k, max_accuracy, marker="x", color = "Red")
plt.title(f"Accuracy (F1 Score) by Optimal 'k' value \n Optimal k value: {max_accuracy_k} with Accuracy: {round(max_accuracy,3)}")
plt.xlabel("Max Depth of decision tree")
plt.ylabel("Accuracy (F1) scores")
plt.tight_layout()
plt.show()

    
    
    
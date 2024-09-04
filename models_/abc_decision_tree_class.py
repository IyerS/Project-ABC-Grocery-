# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 12:16:35 2024

@author: Administrator
"""

# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.utils import shuffle 
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


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
clf = DecisionTreeClassifier(random_state=42, max_depth=5) #play with max_iter
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


# Finding the best Max Depth (Max depth can be extracted as 9)

max_depth_list = list(range(1,15))
accuracy_score = []
for d in max_depth_list:
    clf = DecisionTreeClassifier(max_depth=d, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = f1_score(y_test, y_pred)
    accuracy_score.append(accuracy)  
score_sheet = pd.DataFrame(accuracy_score, max_depth_list).reset_index()
score_sheet.columns = ["depth","accuracy_score"]

max_accuracy = max(accuracy_score)
max_accuracy_idx = accuracy_score.index(max_accuracy)
max_accuracy_depth = max_depth_list[max_accuracy_idx]

plt.plot(max_depth_list, accuracy_score)
plt.scatter(max_accuracy_depth, max_accuracy, marker="x", color = "Red")
plt.title(f"Accuracy (F1 Score) by Max Depth \n Optimal Tree Depth: {max_accuracy_depth} with Accuracy: {round(max_accuracy,3)}")
plt.xlabel("Max Depth of decision tree")
plt.ylabel("Accuracy (F1) scores")
plt.tight_layout()
plt.show()

# printing our tree
plt.figure(figsize=(25,15))
tree = plot_tree(clf,
                 feature_names = X_train.columns.tolist(),
                 filled = True, rounded = True,
                 fontsize = 24)

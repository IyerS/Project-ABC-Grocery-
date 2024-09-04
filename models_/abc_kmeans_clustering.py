# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 13:52:49 2024

@author: Administrator
"""

from sklearn.cluster import KMeans 
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 

#######################################################
# Create data
#######################################################

transactions = pd.read_excel("C:/Users/Administrator/Documents/DS-Infinity/grocery_database.xlsx", sheet_name="transactions")
product_areas = pd.read_excel("C:/Users/Administrator/Documents/DS-Infinity/grocery_database.xlsx", sheet_name="product_areas")

transactions = pd.merge(transactions, product_areas, how = "inner", on= "product_area_id")

# drop non-food category

transactions.drop(transactions[transactions["product_area_name"] == "Non-Food"].index, inplace = True)

# aggregate sales data ay customer level
transaction_summary = transactions.groupby(["customer_id","product_area_name"])["sales_cost"].sum().reset_index()

# put the product areas as columns using some pivot
transaction_summary_pivot = transactions.pivot_table(index = "customer_id",
                                                     columns = "product_area_name",
                                                     values = "sales_cost",
                                                     aggfunc = "sum",
                                                     fill_value = 0,
                                                     margins = True,
                                                     margins_name = "Total").rename_axis(None, axis=1)

# Convert to percentage of total sales
transaction_summary_pivot = transaction_summary_pivot.div(transaction_summary_pivot["Total"], axis = 0)

# data for clustering
data_for_clustering = transaction_summary_pivot.drop(["Total"], axis=1)

#######################################################
# Data prep and cleaning
#######################################################

data_for_clustering.isna().sum()

# Normalize data
scale_norm = MinMaxScaler()
data_for_clustering_scaled = pd.DataFrame(scale_norm.fit_transform(data_for_clustering), columns = data_for_clustering.columns)

#######################################################
# Using WCSS to find the value of k to use
#######################################################

k_values = list(range(1,10))
wcss = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_for_clustering_scaled)
    wcss.append(kmeans.inertia_)
    
plt.plot(k_values,wcss)
plt.title("within Cluster sum of squares")
plt.show()

# using the k from above to model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data_for_clustering_scaled)

# adding the cluster label to our data
data_for_clustering["Cluster"] = kmeans.labels_
data_for_clustering["Cluster"].value_counts()

#######################################################
# Profiling these clusters
#######################################################
cluster_summary = data_for_clustering.groupby("Cluster")[["Dairy","Fruit","Meat","Vegetables"]].mean()

























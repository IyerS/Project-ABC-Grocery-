# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:53:47 2024

@author: Administrator
"""

import pandas as pd
import pickle 

loyalty_scores = pd.read_excel("C:/Users/Administrator/Documents/DS-Infinity/grocery_database.xlsx", sheet_name="loyalty_scores")
customer_details = pd.read_excel("C:/Users/Administrator/Documents/DS-Infinity/grocery_database.xlsx", sheet_name="customer_details")
transactions = pd.read_excel("C:/Users/Administrator/Documents/DS-Infinity/grocery_database.xlsx", sheet_name="transactions")

# create the customer level dataset

data_for_regression = pd.merge(customer_details, loyalty_scores, how="left", on="customer_id")

trans1 = transactions.groupby(["customer_id"]).sum("sales_cost")["sales_cost"]
trans2 = transactions.groupby(["customer_id"]).sum("num_items")["num_items"]
trans3 = transactions.groupby(["customer_id"]).nunique("transaction_id")["transaction_id"]
test1 = transactions[transactions["customer_id"] == 2]["transaction_id"].nunique()

sales_summary = transactions.groupby("customer_id").agg({"sales_cost":"sum",
                                                         "num_items":"sum",
                                                         "transaction_id":"count",
                                                         "product_area_id":"nunique"}).reset_index()

sales_summary.columns = ["customer_id","total_sales","total_items","transaction_count","product_area_count"]

sales_summary["average_basket_value"] = sales_summary["total_sales"] / sales_summary["transaction_count"]

data_for_regression = pd.merge(data_for_regression, sales_summary, how="inner", on="customer_id")

regression_modelling = data_for_regression[data_for_regression["customer_loyalty_score"].notna()]
regression_scoring = data_for_regression[data_for_regression["customer_loyalty_score"].isna()]
regression_scoring.drop("customer_loyalty_score", axis=1, inplace=True)


## saving our file using pickle

pickle.dump(regression_modelling, open("abc_regression_modelling.p","wb"))
pickle.dump(regression_scoring, open("abc_regression_scoring.p","wb"))

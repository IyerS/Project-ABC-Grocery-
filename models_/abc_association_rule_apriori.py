# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 13:23:44 2024

@author: Administrator
"""

from apyori import apriori
import pandas as pd

alcohol_transaction = pd.read_csv("C:/Users/Administrator/Documents/DS-Infinity/PY ML/sample_data_apriori.csv")
alcohol_transaction.drop("transaction_id", axis=1, inplace=True)

# modifying the data for apriori algo that takes a list of lists with each transaction being one list of products
transactions = []
for index, row in alcohol_transaction.iterrows():
    transaction = list(row.dropna())
    transactions.append(transaction)
    
# Applying the aprori algorithm
apriori_rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)
apriori_rules = list(apriori_rules)
apriori_rules[2][0][0]

# Convert this output into a dataframe from list comprehension
product1 = [list(rule[2][0][0])[0] for rule in apriori_rules]
product2 = [list(rule[2][0][1])[0] for rule in apriori_rules]
support = [rule[1] for rule in apriori_rules]
confidence = [rule[2][0][2] for rule in apriori_rules]
lift = [rule[2][0][3] for rule in apriori_rules]

apriori_rules = pd.DataFrame({"product 1": product1,
                              "product 2": product2,
                              "support": support,
                              "confidence": confidence,
                              "lift": lift})
# sort based on lift
apriori_rules.sort_values(by="lift", ascending=False, inplace=True)

# Search Rules

apriori_rules[apriori_rules["product 1"].str.contains("New Zealand")]

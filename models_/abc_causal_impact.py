# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 19:21:17 2024

@author: Administrator
"""
#############################################################################################
# what is the impact of user joining the mailer club on their purchase pattern
#####################################################################################################

from causalimpact import CausalImpact
import pandas as pd

transactions = pd.read_excel("C:/Users/Administrator/Documents/DS-Infinity/grocery_database.xlsx", sheet_name="transactions")
campaign_data = pd.read_excel("C:/Users/Administrator/Documents/DS-Infinity/grocery_database.xlsx", sheet_name="campaign_data")

customer_daily_sales = transactions.groupby(["customer_id", "transaction_date"])["sales_cost"].sum().reset_index()
customer_daily_sales = customer_daily_sales.merge(campaign_data, how = "left", on = "customer_id")

causal_impact_df = customer_daily_sales.pivot_table(index = "transaction_date",
                                                    columns = "signup_flag",
                                                    values = "sales_cost",
                                                    aggfunc = "mean")

causal_impact_df.index
# provide a frequency for our datetime index generated above
causal_impact_df.index.freq = "D"

# new data frame in the order of 1 and 0 instead of 0 and 1
causal_impact_df = causal_impact_df[[1,0]]

# renaming the columns
causal_impact_df.columns = ["member","non-member"]

# apply causal impact
pre_period = ["2020-04-01","2020-06-30"]
post_period = ["2020-07-01","2020-09-30"]

ci = CausalImpact(causal_impact_df, pre_period, post_period)

# plot impact
ci.plot()

print(ci.summary())
print(ci.summary(output = "report"))

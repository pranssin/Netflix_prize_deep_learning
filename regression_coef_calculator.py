"""
This file is responsible for creating regressions for each user in our dataset and then saving
the coefficients so that we may use them later.

@author: David Rocker
@author: Pranshu Sinha
@author: Sameer Poudwal
"""

from sklearn.linear_model import LinearRegression
import numpy as np
import database
import pandas as pd
import pickle as pkl
import datetime as dt
from copy import deepcopy
import sys

datab = database.Database()
count_total = 0
count=0
max_count=2
users = []

# Get list of users
with open('userid_list.pkl', 'rb') as f:
    users = pkl.load(f)

coef_data = {}


command = "SELECT rating, date_watched FROM netflix.data WHERE user_id={};"
col_list = ['rating', 'date_watched']

insert_command = "INSERT INTO reg_coef VALUES ({});"

date_norm_factor = dt.datetime.strptime('1999-11-11', "%Y-%m-%d")
date_norm_factor = dt.datetime.toordinal(date_norm_factor)

'''
Get date bucket values
Get first and last dates, convert to ordinal, and divide by 30 or use linspace
'''
last_date = dt.datetime.strptime('2005-12-31', "%Y-%m-%d")
last_date = dt.datetime.toordinal(last_date)
timestep = (last_date - date_norm_factor) / 29
buckets = [int(date_norm_factor + i * timestep) for i in range(0, 30)]

col_names = ", ".join(["col{} double precision".format(i) for i in range(0, 62)])
create_table = "CREATE TABLE IF NOT EXISTS reg_coef (" \
                      "user_id integer NOT NULL," \
                      "{} " \
                      ")".format(col_names)

datab.execute_command(create_table)

command2 = []

# Create a regression for each user in our user list
count = 5000
for user in users[5000:]:
    user_data = datab.get_data(command.format(user), col_list)
    user_data['date_watched'] = user_data['date_watched'].map(dt.datetime.toordinal)
    user_data['date_watched'] = user_data['date_watched'] - date_norm_factor
    count_total = count_total + 1
    '''
    Don't overwrite the user data now, store as two seperate DataFrames.  One
    DataFrame for overall regression, and one for use in buckets
    
    Perhaps one main list as well that we append coefficients to, then set the
    value for that user to be the list after we are done with all linear models
    '''
    results = [user]
    bucket_data = deepcopy(user_data)
    user_data = user_data.as_matrix()
    
    # Create and fit the linear regression
    lm = LinearRegression()
    lm.fit(user_data[:, 1].reshape(-1, 1), user_data[:, 0].reshape(-1, 1))
    
    # Get the coefficients
    intercept_1 = lm.intercept_[0]
    b_1 = lm.coef_[0][0]
    intercept = lm.intercept_[0]
    b = lm.coef_[0][0]
    
    # Put the coefficients into a list
    results.extend([intercept, b])
        
    # Store the results in a hash table and then save them after each iteration in case we lose power
    # or something happens to our computer
    coef_data[user] = results
    with open('user_coef_data_3.pkl', 'wb') as f:
        pkl.dump(coef_data[user], f)

    
    print("Done with user {}".format(count))
    count = count + 1

    
with open('user_coef_data_3.pkl', 'wb') as f:
    pkl.dump(coef_data, f)
    

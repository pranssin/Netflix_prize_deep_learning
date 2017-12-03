#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 12:58:10 2017

@author: david
"""

from sklearn.linear_model import LinearRegression
import numpy as np
import database
import pandas as pd
import pickle as pkl
import datetime as dt
from copy import deepcopy

datab = database.Database()

col_names = ", ".join(["col{} double precision".format(i) for i in range(0, 62)])
delete_table = "DROP TABLE IF EXISTS netflix.reg_coef;"
create_table = "CREATE TABLE IF NOT EXISTS netflix.reg_coef (" \
                      "user_id integer NOT NULL," \
                      "{} " \
                      ")" \
                      "WITH (" \
                      "     OIDS = FALSE" \
                      ");" \
                      "ALTER TABLE netflix.data OWNER to postgres;".format(col_names)
                      
datab.execute_command(delete_table)
datab.execute_command(create_table)


command = 'SELECT DISTINCT user_id FROM netflix.data limit 1;'
users = datab.get_data(command=command, col_list=['user_id'])
users = users.user_id.values
coef_data = {}


command = "SELECT rating, date_watched FROM netflix.data WHERE user_id={};"
col_list = ['rating', 'date_watched']

insert_command = "INSERT INTO netflix.reg_coef VALUES ({})"

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

for user in users:
    user_data = datab.get_data(command.format(user), col_list)
    user_data['date_watched'] = user_data['date_watched'].map(dt.datetime.toordinal)
    user_data['date_watched'] = user_data['date_watched'] - date_norm_factor
    
    '''
    Don't overwrite the user data now, store as two seperate DataFrames.  One
    DataFrame for overall regression, and one for use in buckets
    
    Perhaps one main list as well that we append coefficients to, then set the
    value for that user to be the list after we are done with all linear models
    '''
    results = [user]
    bucket_data = deepcopy(user_data)
    user_data = user_data.as_matrix()
    
    lm = LinearRegression()
    lm.fit(user_data[:, 1].reshape(-1, 1), user_data[:, 0].reshape(-1, 1))
    intercept = lm.intercept_[0]
    b = lm.coef_[0][0]
    
    results.extend([intercept, b])
    
    
    '''
    Create for loop to train model and get coefficients for each bucket
    '''
    for i in range(0, 29):
        lm = LinearRegression()
        temp = bucket_data.loc[buckets[i] <= bucket_data[1] < buckets[i+1]]
        x = temp[:, 1].reshape(-1, 1)
        y = temp[:, 0].reshape(-1, 1)
        lm.fit(x, y)
        intercept = lm.intercept_[0]
        b = lm.coef_[0][0]
        
        results.append(intercept)
        results.append(b)
    
    coef_data[user] = results
    print(results)
    values = ", ".join([str(i) for i in results])
    #datab.execute_command(insert_command.format(values))
    
    '''
    Add delete statements to free up space so we dont get a memory dump
    '''
    
with open('user_coef_data.pkl', 'wb') as f:
    pkl.dump(coef_data, f)



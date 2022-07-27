#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json 
import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import joblib
from fastapi import FastAPI, Depends, HTTPException, status
import sys
from fastapi import FastAPI

app = FastAPI( title="scoring crédit",
    description="""Obtenir des informations relatives à la probabilité qu'un client ne rembourse pas son prêt""")


@app.get("/")
def read_root():
    return {"200": "Welcome To Heroku"}


path = os.path.join('data', 'y_train.csv')
df_current_clients = pd.read_csv(path, index_col='SK_ID_CURR')

df_current_clients_by_target_repaid = df_current_clients[df_current_clients["TARGET"] == 0]
df_current_clients_by_target_not_repaid = df_current_clients[df_current_clients["TARGET"] == 1]

path = os.path.join('data', 'X_train.csv')
X_train = pd.read_csv(path, index_col='SK_ID_CURR')
df_clients_to_predict=X_train.reset_index()
clients_id = df_clients_to_predict["SK_ID_CURR"].tolist()


df_prediction_by_id = df_clients_to_predict[df_clients_to_predict["SK_ID_CURR"] == id]
df_prediction_by_id = df_clients_to_predict[df_clients_to_predict["SK_ID_CURR"] == id]



@app.get("/clients/{sk_ids}")
async def client_details(sk_ids: int):
    """ 
    EndPoint to get client's detail 
    """ 

    clients_id = df_clients_to_predict["SK_ID_CURR"].tolist()

    if sk_ids not in clients_id:
        raise HTTPException(status_code=404, detail="client's id not found")
    else:
        # Filtering by client's id
        df_by_id = df_clients_to_predict[COLUMNS][df_clients_to_predict["SK_ID_CURR"] == sk_ids]
        idx = df_clients_to_predict[df_clients_to_predict["SK_ID_CURR"]==sk_ids].index[0]

        for col in df_by_id.columns:
            globals()[col] = df_by_id.iloc[0, df_by_id.columns.get_loc(col)]
        
        client = {
            "clientId" : int(SK_ID_CURR),
            "gender" : "Man" if int(CODE_GENDER) == 0 else "Woman",
            "age" : calculate_years(int(DAYS_BIRTH)),
            "antiquity" : calculate_years(int(DAYS_REGISTRATION)),
            "credit" : float(AMT_CREDIT),
            "anualIncome" : float(AMT_INCOME_TOTAL),
            "source2" : float(EXT_SOURCE_2),
            "source3" : float(EXT_SOURCE_3),
        }

    return client


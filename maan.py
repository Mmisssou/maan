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

path = os.path.join('model', 'bestmodel_joblib.pkl')
with open(path, 'rb') as file:
    model = joblib.load(file)


df_prediction_by_id = df_clients_to_predict[df_clients_to_predict["SK_ID_CURR"] == id]
df_prediction_by_id = df_clients_to_predict[df_clients_to_predict["SK_ID_CURR"] == id]





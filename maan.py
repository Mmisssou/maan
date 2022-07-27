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
    return {"200": "Bienvenue sur API du projet N°:07 Encadré par :Monsieur : El hadji Abdoulaye Thiam "}


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

COLUMNS = [
    "SK_ID_CURR", "AMT_INCOME_TOTAL", "CODE_GENDER", 
    "DAYS_BIRTH", "DAYS_REGISTRATION", 
    "AMT_CREDIT", "EXT_SOURCE_2",
    "EXT_SOURCE_3", 
]
path = os.path.join('model', 'bestmodel_joblib.pkl')
with open(path, 'rb') as file:
    model = joblib.load(file)



@app.post('/api/sk_ids/')
def sk_ids(sk_ids : int):
    # Extract list of all the 'SK_ID_CURR' ids in the X_test dataframe
    sk_ids = pd.Series(list(X_test.index.sort_values()))
    # Convert pd.Series to JSON
    sk_ids_json = json.loads(sk_ids.to_json())
    # Returning the processed data
    return sk_ids


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


@app.get("/api/predictions/clients/{sk_ids}")
async def predict(id: int):
    """ 
    EndPoint to get the probability honor/compliance of a client
    """ 

    # Loading the model
    

    # Filtering by client id
    df_prediction_by_id = df_clients_to_predict[df_clients_to_predict["SK_ID_CURR"] == id]
    df_prediction_by_id = df_prediction_by_id.drop(columns=["SK_ID_CURR"])

    # Predicting
    result = model.predict(df_prediction_by_id)
    result_proba = model.predict_proba(df_prediction_by_id)

    if (int(result[0]) == 0):
         result = "Yes"
    else:
         result = "No"    

    return {"repay" : result, "probability" : result_proba}




@app.get('/api/feat_imp/')
def send_feat_imp(feat_imp):
    feat_imp = pd.Series(clf_step.feature_importances_,
                         index=X_te_featsel.columns).sort_values(ascending=False)
    # Convert pd.Series to JSON
    feat_imp_json = json.loads(feat_imp.to_json())
    # Return the processed data as a json object
    return feat_imp_json

@app.get('/api/data_cust/')
def data_cust(X_cust:int):
    # Parse the http request to get arguments (sk_id_cust)
    sk_id_cust = int(request.args.get('SK_ID_CURR'))
    # Get the personal data for the customer (pd.Series)
    X_cust_ser = X_test.loc[sk_id_cust, :]
    X_cust_proc_ser = X_te_featsel.loc[sk_id_cust, :]
    # Convert the pd.Series (df row) of customer's data to JSON
    X_cust_json = json.loads(X_cust_ser.to_json())
    X_cust_proc_json = json.loads(X_cust_proc_ser.to_json())
    # Return the cleaned data
    return  X_cust_json, X_cust_proc_json


# find 20 nearest neighbors among the training set
def get_df_neigh(sk_id_cust):
    # get data of 20 nearest neigh in the X_tr_featsel dataframe (pd.DataFrame)
    neigh = NearestNeighbors(n_neighbors=20)
    neigh.fit(X_tr_featsel)
    X_cust = X_te_featsel.loc[sk_id_cust: sk_id_cust]
    idx = neigh.kneighbors(X=X_cust,
                           n_neighbors=20,
                           return_distance=False).ravel()
    nearest_cust_idx = list(X_tr_featsel.iloc[idx].index)
    X_neigh_df = X_tr_featsel.loc[nearest_cust_idx, :]
    y_neigh = y_train.loc[nearest_cust_idx]
    return X_neigh_df, y_neigh

@app.get('/api/neigh_cust/')
def neigh_cust(X_neigh, y_neigh):
    # Parse the http request to get arguments (sk_id_cust)
    sk_id_cust = int(request.args.get('SK_ID_CURR'))
    # return the nearest neighbors
    X_neigh_df, y_neigh = get_df_neigh(sk_id_cust)
    # Convert the customer's data to JSON
    X_neigh_json = json.loads(X_neigh_df.to_json())
    y_neigh_json = json.loads(y_neigh.to_json())
    # Return the cleaned data jsonified
    return X_neigh_json, y_neigh_json

@app.get('/api/all_proc_data_tr/')
def all_proc_data_tr(X_tr_featsel_json, y_train_json):
    # get all data from X_tr_featsel, X_te_featsel and y_train data
    # and convert the data to JSON
    X_tr_featsel_json = json.loads(X_tr_featsel.to_json())
    y_train_json = json.loads(y_train.to_json())
    # Return the cleaned data jsonified
    return X_tr_featsel_json, y_train_json


@app.get('/api/scoring_cust/')
def scoring_cust(sk_id_cust, score_cust, thresh):
    # Parse http request to get arguments (sk_id_cust)
    sk_id_cust = int(request.args.get('SK_ID_CURR'))
    # Get the data for the customer (pd.DataFrame)
    X_cust = X_test.loc[sk_id_cust:sk_id_cust]
	# Compute the score of the customer (using the whole pipeline)   
    score_cust = bestmodel.predict_proba(X_cust)[:,1][0]
    # Return score
    return  sk_id_cust, score_cust, thresh

#Importing the logit function for the base value transformation
from scipy.special import expit 
# Conversion of shap values from log odds to probabilities
def shap_transform_scale(shap_values, expected_value, model_prediction):
    #Compute the transformed base value, which consists in applying the logit function to the base value    
    expected_value_transformed = expit(expected_value)
    #Computing the original_explanation_distance to construct the distance_coefficient later on
    original_explanation_distance = sum(shap_values)
    #Computing the distance between the model_prediction and the transformed base_value
    distance_to_explain = model_prediction - expected_value_transformed
    #The distance_coefficient is the ratio between both distances which will be used later on
    distance_coefficient = original_explanation_distance / distance_to_explain
    #Transforming the original shapley values to the new scale
    shap_values_transformed = shap_values / distance_coefficient
    return shap_values_transformed, expected_value_transformed

@app.get('/api/shap_values/')
def shap_values(shap_val,shap_val_cust_trans,exp_val,exp_val_trans,X_neigh_):
    # Parse http request to get arguments (sk_id_cust)
    sk_id_cust = int(request.args.get('SK_ID_CURR'))
    # return the nearest neighbors
    X_neigh, y_neigh = get_df_neigh(sk_id_cust)
    X_cust = X_te_featsel.loc[sk_id_cust].to_frame(sk_id_cust).T
    X_neigh_ = pd.concat([X_neigh, X_cust], axis=0)
    # prepare the shap values of nearest neighbors + customer
    shap_val_neigh_ =  shap_val_X_tr_te.loc[X_neigh_.index]
    # Conversion of shap values from log odds to probabilities of the customer's shap values
    shap_t, exp_t = shap_transform_scale(shap_val_X_tr_te.loc[sk_id_cust],
                                         expected_val,
                                         clf_step.predict_proba(X_neigh_)[:,1][-1])
    shap_val_cust_trans = pd.Series(shap_t,
                                    index=X_neigh_.columns)
    # Converting the pd.Series to JSON
    X_neigh__json = json.loads(X_neigh_.to_json())
    shap_val_neigh_json = json.loads(shap_val_neigh_.to_json())
    shap_val_cust_trans_json = json.loads(shap_val_cust_trans.to_json())
    # Returning the processed data
    return  shap_val_neigh_json, shap_val_cust_trans_json, expected_val, exp_t, X_neigh__json


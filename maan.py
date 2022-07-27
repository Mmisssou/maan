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









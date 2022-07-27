#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"200": "Welcome To Heroku"}









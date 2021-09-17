# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 19:35:30 2021

@author: Akshay
"""

import pickle
import pandas as pd
import numpy as np

def prediction(loaded_model,inp):
    X_test=inp[:-1]
    Y_test=inp[-1]

    ynew = loaded_model.predict(X_test)
    ynew_prob = loaded_model.predict_proba(X_test)
    #print('inp',ynew, ynew_prob)
    
    prob1 = round(ynew_prob[0][0]*100,1)
    prob2 = round(ynew_prob[0][1]*100,1)
    
    return ynew[0], prob1, prob2

filename = 'final_model'#.sav'
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
#print(loaded_model)
inp=[0.20055,0.39641,2.0472,32.351,1.3305,1.1389,0.6598,0.1666,497.42,0.73378,0.14942,43.37,1.2479,0.21402,0.47706,1.4582,1.7615,0.11788,94.14,1.741,593.27,0.051402,49.394,0.37854,2.2437,0.001924,0.643620821,4.216690321,0]
print(len(inp))
print(scaler.transform(inp))
cl, p1, p2=prediction(loaded_model,scaler.transform(inp))
print('predicted class is', cl)
print('prob for class 0 is', p1)
print('prob for class 1 is', p2)

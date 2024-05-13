# Import Libraries
import tensorflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras
from datetime import datetime
from keras import backend as K


def r_square(y_true, y_pred):
    """
    compute the coefficient of determination (R^2) for regression
    ref:ref: https://github.com/keras-team/keras/issues/7947
    """
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))


def rmse(y_true, y_pred):
    """
    compute the root mean squared error (rmse) for regression
      ref: https://github.com/keras-team/keras/issues/7947"""
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

# Objective Function
def obj_func(model, scaler, X): 
    """
    Objective function based on the model
    """
    # scaler_transform, shape = (n_samples, n_features)
    inputs =  scaler.transform(X) 
    # model_predict = (n_samples, 1)
    output = model.predict(inputs) 
    return output.flatten()

def salt_adsorption_ML():
    # {'num_layers': 5,c'units_0': 50, 'learning_rate': 0.01, 'units_1': 50,
    #'units_2': 50, 'units_3': 50, 'units_4': 50}
    model = Sequential()
    model.add(Dense(50, input_dim=10, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(keras.layers.Dense(1, activation='linear'))

    model.compile(optimizer=tensorflow.keras.optimizers.legacy.Adam(learning_rate=0.001),
                          loss="mean_squared_error") # metrics=["mean_squared_error", rmse, r_square])
    
    return model


def specific_capacitance_ML():
    # {'num_layers': 10, 'units_0': 21, 'units_1': 38, 'units_2': 9, 'units_3': 35, 'units_4': 21,
    # 'units_5': 45,'units_6': 50, 'units_7': 14, 'units_8': 50, 'units_9': 10, 'learning_rate': 0.01,}
    model = Sequential()
    model.add(Dense(21, input_dim=7, activation='relu'))
    model.add(Dense(38, activation='relu'))
    model.add(Dense(9, activation='relu'))
    model.add(Dense(35, activation='relu'))
    model.add(Dense(21, activation='relu'))
    model.add(Dense(45, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(14, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(keras.layers.Dense(1, activation='linear'))


    model.compile(optimizer=tensorflow.keras.optimizers.legacy.Adam(learning_rate=0.01),
                          loss="mean_squared_error") # metrics=["mean_squared_error", rmse, r_square])
    
    return model


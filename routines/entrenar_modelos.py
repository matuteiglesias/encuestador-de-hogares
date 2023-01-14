#!/usr/bin/env python
# coding: utf-8
# 


### PARSEO DE ARGUMENTOS

import argparse

parser = argparse.ArgumentParser(description='A script to process data for a range of years')

parser.add_argument('-y','--years', nargs='+', help='Set the range of years to process data for. Default is the current year and the next year', required=False, type=int, default=[2022, 2023])
parser.add_argument('-ow','--overwrite', nargs=1, required=False, default= True, help='Flag to specify if previous data should be overwritten. Default is True')

args = parser.parse_args()

overwrite = args.overwrite
startyr = args.years[0]
endyr = args.years[1]


# Column names
y_cols = ['CAT_OCUP', 'P47T', 'PP10E', 'PP10D', 'PP07K', 'PP07I', 'V3_M', 'PP07G4', 'CH16', 'T_VI', 
          'V12_M', 'TOT_P12', 'PP07G3', 'V5_M', 'PP07H', 'V2_M', 'PP10C', 
          'PP08D1', 'PP07J', 'CAT_INAC', 'CH07', 'CH08', 'P21', 'PP07G1', 'PP07G_59', 'PP07G2']

x_cols1 = ['IX_TOT', 'P02', 'P03', 'AGLO_rk', 'Reg_rk', 'V01', 'H05', 'H06',
       'H07', 'H08', 'H09', 'H10', 'H11', 'H12', 'H16', 'H15', 'PROP', 'H14',
       'H13', 'P07', 'P08', 'P09', 'P10', 'P05', 'CONDACT']

predecir1 = ['CAT_OCUP', 'CAT_INAC', 'CH07']

x_cols2 = x_cols1 + predecir1
predecir2 = ['INGRESO', 'INGRESO_NLB', 'INGRESO_JUB', 'INGRESO_SBS']

x_cols3 = x_cols2 + predecir2
predecir3 = ['PP07G1','PP07G_59', 'PP07I', 'PP07J', 'PP07K']

# Columnas de ingresos. Necesitan una regresion...
columnas_pesos = [u'P21', u'P47T', u'PP08D1', u'TOT_P12', u'T_VI', u'V12_M', u'V2_M', u'V3_M', u'V5_M']

x_cols4 = x_cols3 + predecir3

predecir4 = columnas_pesos
y_cols4 = predecir4


#####  TRAINING  #####

#####  Funciones clasificador y regresor  #####

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import os
import joblib

def fit_model(train_data, x_cols, y_cols, out_filename,
             model):
    X = train_data[x_cols]
    y = train_data[y_cols]
    
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.1) # less memory used
    
    clf = model.fit(X.values, y.values) # fit model

    # save the model to disk
    if not os.path.exists('./../fitted_RF/'):
        os.makedirs('./../fitted_RF/')
    joblib.dump(model, out_filename, compress=3)
    print('saved model at: ' + out_filename)

#     print(sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)[:5])
    del clf
    del X; del y # liberar memoria eliminando los dataframes mas pesados
    
##########################################################################################    
################ Loop principal. Entrenar y guardar modelos  #############################
##########################################################################################

import pandas as pd
from numpy import log10

for yr in [str(s) for s in range(startyr, endyr)]:
    print(yr)
    train_data = pd.read_csv('./../data/training/EPHARG_train_'+yr[2:]+'.csv')
    
    ## ETAPA 1:
    out = './../fitted_RF/clf1_'+yr+'_ARG'
    if (not os.path.exists(out)) or (overwrite):
        fit_model(train_data, x_cols = x_cols1, y_cols = predecir1, out_filename = out,
                 model = RandomForestClassifier(n_estimators=100, max_depth = 20, n_jobs = -1))
    
    ## ETAPA 2:
    out = './../fitted_RF/clf2_'+yr+'_ARG'
    if (not os.path.exists(out)) or (overwrite):
        fit_model(train_data, x_cols = x_cols2, y_cols = predecir2, out_filename = out,
                 model = RandomForestClassifier(n_estimators=100, max_depth = 20, n_jobs = -1))
    
    ## ETAPA 3:
    out = './../fitted_RF/clf3_'+yr+'_ARG'
    if (not os.path.exists(out)) or (overwrite):
        fit_model(train_data, x_cols = x_cols3, y_cols = predecir3, out_filename = out,
                 model = RandomForestClassifier(n_estimators=100, max_depth = 20, n_jobs = -1))
    
    ## ETAPA 4 (Regresion)
    ## Tomar log de las columnas en pesos.
    train_data[columnas_pesos] = log10(train_data[columnas_pesos].clip(-.9) + 1)

    ## Entrenar modelo, para cada trimestre
    for q in train_data.Q.unique():
        print(q)
        out = './../fitted_RF/clf4_'+q+'_ARG'
        if (not os.path.exists(out)) or (overwrite):
            train_q = train_data.loc[train_data.Q == q]
            fit_model(train_q, x_cols = x_cols4, y_cols = y_cols4, out_filename = out,
                     model = RandomForestRegressor(n_estimators=1, max_depth = 40, n_jobs = -1))
            del train_q;

    del train_data; 

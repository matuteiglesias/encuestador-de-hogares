# %% [markdown]
# 

# %%



x_cols1 = ['IX_TOT', 'P02', 'P03', 'AGLO_rk', 'Reg_rk', 'V01', 'H05', 'H06',
       'H07', 'H08', 'H09', 'H10', 'H11', 'H12', 'H16', 'H15', 'PROP', 'H14',
       'H13', 'P07', 'P08', 'P09', 'P10', 'P05', 'CONDACT']

predecir1 = ['CAT_OCUP', 'CAT_INAC', 'CH07']

x_cols2 = x_cols1 + predecir1
predecir2 = ['INGRESO', 'INGRESO_NLB', 'INGRESO_JUB', 'INGRESO_SBS']

x_cols3 = x_cols2 + predecir2
# La seccion PP07G pregunta si el trabajo es en blanco y que beneficios tiene. Puede ayudar a la regresion para ingresos.
# predecir3 = ['PP07G1', 'PP07G2', 'PP07G3', 'PP07G4', 'PP07G_59', 'PP07H', 'PP07I', 'PP07J', 'PP07K']
predecir3 = ['PP07G1','PP07G_59', 'PP07I', 'PP07J', 'PP07K']

# Columnas de ingresos. Necesitan una regresion...
columnas_pesos = [u'P21', u'P47T', u'PP08D1', u'TOT_P12', u'T_VI', u'V12_M', u'V2_M', u'V3_M', u'V5_M']

x_cols4 = x_cols3 + predecir3
# Columnas de ingresos. Necesitan una regresion...



# %%
import pandas as pd
from numpy import log10

models_path = './..'
overwrite = True
startyr = 2023
endyr = 2024

yr = str(startyr)


# %%


from sklearn.model_selection import train_test_split # importamos las funciones para dividir los datos
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor # importamos los modelos de random forest
import os # para trabajar con el sistema operativo
import joblib # para guardar el modelo entrenado

def fit_model(train_data, x_cols, y_cols, out_filename, model):
    """
    Entrena un modelo de random forest.
    train_data: dataframe con los datos de entrenamiento
    x_cols: lista con los nombres de las columnas de entrada del modelo
    y_cols: lista con los nombres de las columnas de salida del modelo
    out_filename: string con la ruta y nombre del archivo donde se guardará el modelo
    model: instancia del modelo a entrenar
    """
    X = train_data[x_cols] # separamos las columnas de entrada
    y = train_data[y_cols] # separamos las columnas de salida
    
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.1) # dividimos los datos en entrenamiento y test
    
    fitted_model = model.fit(X.values, y.values) # entrenamos el modelo
    
    # guardamos el modelo en disco
    if not os.path.exists(models_path + '/fitted_RF/'):
        os.makedirs(models_path + '/fitted_RF/')
    joblib.dump(model, out_filename, compress=3)
    print('saved model at: ' + out_filename)


    return fitted_model

    # print(sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)[:5])
    # del fitted_model
    # del X; del y # liberar memoria eliminando los dataframes mas pesados


# %%

print(yr)
train_data = pd.read_csv('./../../data/training/EPHARG_train_'+yr[2:]+'.csv')

## ETAPA 4 (Regresion)
## Tomar log de las columnas en pesos.
train_data[columnas_pesos] = log10(train_data[columnas_pesos].clip(-.9) + 1)

out = models_path + '/fitted_RF/reg_ARG'

fitted_model = fit_model(train_data, x_cols = x_cols4, y_cols = columnas_pesos, out_filename = out,
            model = RandomForestRegressor(n_estimators=1, max_depth = 20, n_jobs = -1))


# %%
fitted_model

# %%


# import pandas as pd
# from numpy import log10
# models_path = './..'
# overwrite = True
# startyr = 2023
# endyr = 2024

# for yr in [str(s) for s in range(startyr, endyr)]:
#     print(yr)
#     train_data = pd.read_csv('./../../data/training/EPHARG_train_'+yr[2:]+'.csv')
    
#     ## ETAPA 1:
#     out = models_path + '/fitted_RF/clf1_'+yr+'_ARG'
#     if (not os.path.exists(out)) or (overwrite):
#         fit_model(train_data, x_cols = x_cols1, y_cols = predecir1, out_filename = out,
#                  model = RandomForestClassifier(n_estimators=100, max_depth = 15, n_jobs = -1))
    
#     ## ETAPA 2:
#     out = models_path + '/fitted_RF/clf2_'+yr+'_ARG'
#     if (not os.path.exists(out)) or (overwrite):
#         fit_model(train_data, x_cols = x_cols2, y_cols = predecir2, out_filename = out,
#                  model = RandomForestClassifier(n_estimators=100, max_depth = 15, n_jobs = -1))
    
#     ## ETAPA 3:
#     out = models_path + '/fitted_RF/clf3_'+yr+'_ARG'
#     if (not os.path.exists(out)) or (overwrite):
#         fit_model(train_data, x_cols = x_cols3, y_cols = predecir3, out_filename = out,
#                  model = RandomForestClassifier(n_estimators=100, max_depth = 15, n_jobs = -1))
    
#     ## ETAPA 4 (Regresion)
#     ## Tomar log de las columnas en pesos.
#     train_data[columnas_pesos] = log10(train_data[columnas_pesos].clip(-.9) + 1)

#     ## Entrenar modelo, para cada trimestre
#     for q in train_data.Q.unique():
#         print(q)
#         out = models_path + '/fitted_RF/clf4_'+q+'_ARG'
#         if (not os.path.exists(out)) or (overwrite):
#             train_q = train_data.loc[train_data.Q == q]
#             fit_model(train_q, x_cols = x_cols4, y_cols = columnas_pesos, out_filename = out,
#                      model = RandomForestRegressor(n_estimators=1, max_depth = 20, n_jobs = -1))
#             del train_q;

#     del train_data; 



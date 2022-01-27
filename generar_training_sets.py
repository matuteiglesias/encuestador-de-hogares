
# Importar modulos
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import glob


## Anios que se computan. Hay que hacerlo pasar a ser variable input del script.
startyr = 2018
endyr = 2022



radio_ref = pd.read_csv('./data/info/radio_ref.csv')
dpto_region = pd.read_csv('./data/info/DPTO_PROV_Region.csv')
radio_ref = radio_ref.merge(dpto_region)
AGLO_Region = radio_ref[['AGLOMERADO', 'Region']].drop_duplicates()

# Decision sobre cual es la region de un aglomerado. GBA tiene que ir a Gran Buenos Aires, aunque algunos de sus radios en partidos como Rodriguez, Escobar, etc sean region pampeana.
# Viedma Patagones, se tendria que tirar de un lado, y la mayoria de sus radios, son Patagonia.
# Se tiene que corregir a mano, porque el AGLO 0 SI tiene varias regiones.

AGLO_Region = AGLO_Region.loc[~((AGLO_Region.AGLOMERADO == 33) & (AGLO_Region.Region == 'Pampeana'))]
AGLO_Region = AGLO_Region.loc[~((AGLO_Region.AGLOMERADO == 93) & (AGLO_Region.Region == 'Pampeana'))]

### Match column names

names_censo = ['IX_TOT', 'P02', 'P03', 'CONDACT', 'AGLOMERADO',
    'V01', 'H05', 'H06', 'H07', 'H08', 'H09', 'H10', 'H11', 'H12', 'H16', 'H15', 'PROP', 'H14', 'H13',
      'P07', 'P08', 'P09', 'P10', 'P05']


names_EPH = ['IX_TOT','CH04','CH06','CONDACT', 'AGLOMERADO',
    'IV1', 'IV3', 'IV4','IV5','IV6','IV7','IV8','IV10','IV11','II1','II2','II7','II8','II9',
    'CH09','CH10','CH12','CH13','CH15']

col_mon = [u'P21', u'P47T', u'PP08D1', u'TOT_P12', u'T_VI', u'V12_M', u'V2_M', u'V3_M', u'V5_M']


## IPC
### Trimestral
url = 'https://raw.githubusercontent.com/matuteiglesias/IPC-Argentina/main/data/info/indice_precios_Q.csv'

cpi = pd.read_csv(url, index_col = 0)
cpi.index = pd.to_datetime(cpi.index)
cpi = cpi['2003':]


### Anual
url = 'https://raw.githubusercontent.com/matuteiglesias/IPC-Argentina/main/data/info/indice_precios_M.csv'

cpi_M = pd.read_csv(url, index_col = 0)[['index']]
cpi_M.index = pd.to_datetime(cpi_M.index)
cpi_M = cpi_M['2003':]


### Referencia de nivel de precios
### 2016 - 01 - 01

url = 'https://raw.githubusercontent.com/matuteiglesias/IPC-Argentina/main/data/info/indice_precios_d.csv'

cpi_d = pd.read_csv(url, index_col=0)[['index']]
cpi_d.index = pd.to_datetime(cpi_d.index)
cpi_d = cpi_d['2003':]

ix = cpi_d.loc['2016-01-01'].values[0]

from datetime import datetime
mes_actual = datetime.today().replace(day=1).strftime("%Y-%m-%d")


## When running the first time we may not have the folder where training data is saved
import os

if not os.path.exists('./data/training/'):
    os.makedirs('./data/training/')
    
    
AGLO_rk = pd.read_csv('./data/info/AGLO_rk')
Reg_rk = pd.read_csv('./data/info/Reg_rk')


### LOOP PRINCIPAL

# from pandas.tseries.offsets import MonthEnd

path ='./../microdatos-EPH-INDEC/microdatos/' # depende de donde hayamos descargado los microdatos
# path ='./../EPH/microdatos/' # depende de donde hayamos descargado los microdatos

for y in range(startyr, endyr):
    print(y)
    yr = str(y)[2:]
    training_file = './data/training/EPHARG_train_'+str(yr)+'.csv'
    
    if not os.path.exists(training_file): # Si todavia no existe la training data de ese anio.

        allFiles = glob.glob(path + 'hogar/*'+str(yr)+'.txt')
        frame = pd.DataFrame()
        list_ = []
        for file_ in allFiles:
            df = pd.read_csv(file_,index_col=None, header=0, delimiter = ';',
                            usecols = ['CODUSU','ANO4','TRIMESTRE','IX_TOT', 'AGLOMERADO',
            'IV1', 'IV3', 'IV4','IV5','IV6','IV7','IV8','IV10','IV11','II1','II2','II7','II8','II9']) 
            ['II2', 'IV5', 'IX_TOT', 'II7', 'IV4', 'II1', 'IV7', 'IV6', 'IV11', 'IV8', 'IV3', 'II8', 'IV1', 'IV10', 'II9']

            print(len(df))
            list_ += [df]
        df = pd.concat(list_)

        # Correcciones Respuestas. Para que matchee censo
        df = df.loc[df.IV1 != 9]
        df['IV10'] = df['IV10'].map({1: 1, 2: 2, 3: 2, 0: 0, 9: 9})
        df['II9'] = df['II9'].map({1: 1, 2: 2, 3: 2, 4: 4, 0: 0})
        df['II7'] = df['II7'].map({1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 6, 8: 6, 9: 6, 0: 0})
        df['II9'] = df['II9'].map({1: 1, 2: 2, 3: 2, 4: 4, 0: 0})
        df['IX_TOT'] = df['IX_TOT'].clip(0, 8)

        hogar = df
        hogar = hogar.drop_duplicates()
        print(hogar.shape)

        allFiles = glob.glob(path + 'individual/usu_individual*'+str(yr)+'.txt')
        frame = pd.DataFrame()
        list_ = []
        for file_ in allFiles:
            print(file_)
        #     print(file_)
            df = pd.read_csv(file_,index_col=None, header=0, delimiter = ';',
                             usecols = ['CODUSU','ANO4','TRIMESTRE','CH04','CH06', 'AGLOMERADO', 'CH09','CH10','CH12','CH13','CH15'] +\
                             ['CH07', 'ESTADO','CAT_INAC','CAT_OCUP','PP07G1', 'PP07G2', 'PP07G3', 'PP07G4', 'PP07G_59', 'PP07H', 'PP07I', 'PP07J', 'PP07K',
                             'P47T', 'V3_M', 'T_VI', 'V12_M', 'TOT_P12', 'V5_M','V2_M', 'PP08D1', 'P21'])
            df = df.rename(columns = {'ESTADO': 'CONDACT'})


            list_ += [df]
        df = pd.concat(list_)

        # Correcciones Respuestas. Para que matchee censo
        df['CH15'] = df['CH15'].map({1:1, 2:1, 3:1, 4:2, 5:2, 9:0})
        df['CH06'] = df['CH06'].clip(0)
        df['CH09'] = df['CH09'].map({1:1, 2:2, 0:2, 3:2})
        df.loc[df['CH06'] < 14, 'CONDACT'] = 0 # Menores de 14 van con CONDACT 0, como en el Censo

        ## En Censo, Jardin y educacion especial no preguntan terminado si/no.
        df['CH12'] = df.CH12.replace(99, 0)
        df.loc[df.CH12.isin([0, 1, 9]), 'CH13'] = 0

    #     df['MAYOR'] = df['CH06'] >= 14 
    #     df['MAYOR'] = df['CH06'] // 7
    #     df['CONDACT'] = df['CAT_OCUP'].fillna(-1)

        indiv = df
        indiv = indiv.dropna(subset = ['P47T'])
        print(indiv.shape)

        indiv_table = indiv[list(indiv.columns.difference(hogar.columns)) + ['CODUSU', 'ANO4', 'TRIMESTRE', 'AGLOMERADO']]
        EPH = hogar.merge(indiv_table, on = ['CODUSU', 'ANO4', 'TRIMESTRE', 'AGLOMERADO'], indicator = True)

        print('Hogar - Indiv merged:')
        print(EPH.shape)


    #     EPH = EPH.loc[EPH.P47T != -9]

        EPH = EPH.merge(AGLO_Region)

        EPH_no_aglo = EPH.copy(); 
        EPH_no_aglo['AGLOMERADO'] = 0

        EPH = pd.concat([EPH, EPH_no_aglo]).reset_index(drop = True)

        print('No aglo agregado:')
        print(EPH.shape)

    #     # Quarters / deflation
    #     EPH['Q'] = EPH.ANO4.astype(str) + ':' + (3*EPH.TRIMESTRE).astype(str)
    #     EPH['Q'] = pd.to_datetime(EPH['Q'], format='%Y:%m') + MonthEnd(1)
    # #     cpi_ultimo_Q = indice_precios['index'].values[-1]

        # Quarters / deflation
        EPH['Q'] = EPH.ANO4.astype(str) + ':' + (3*EPH.TRIMESTRE).astype(str)
        EPH['Q'] = pd.to_datetime(EPH['Q'], format='%Y:%m') - pd.DateOffset(months=1) + pd.DateOffset(days=14)
        print(EPH['Q'].unique())


    #     EPH[col_mon] = cpi_mes_actual*EPH[col_mon].div(EPH[['Q'] + col_mon].merge(cpi, on = 'Q', how = 'left')['index'].values, 0)
        EPH[col_mon] = ix*EPH[col_mon].div(EPH[['Q'] + col_mon].merge(cpi, on = 'Q', how = 'left')['index'].values, 0)

        # 2018Q3 -> Mar19 1.3156
        # 2018Q3 -> Abr19 1.361
    #     EPH[col_mon] = 1.361*EPH[col_mon]

        EPH[col_mon] = EPH[col_mon].round()

        print('deflactado:')
        print(EPH.shape)

        training = EPH.rename(columns = dict(zip(names_EPH, names_censo)))
        
        training = training.loc[training.P47T >= -0.001]
        training[col_mon] = training[col_mon].fillna(0)
        training = training.merge(AGLO_rk[['AGLOMERADO', 'AGLO_rk']]).merge(Reg_rk[['Region', 'Reg_rk']])
        
        training.to_csv(training_file, index = False)




### PARSEO DE ARGUMENTOS

import argparse
import pandas as pd
import glob
import os

#os.chdir(os.path.dirname(__file__))

parser = argparse.ArgumentParser()
# parser.add_argument('-i', action='append', nargs='+')


### ARGUMENTO YEARS PARA PEDIR LA VENTANA DE TIEMPO DESEADA
parser.add_argument('-y','--years', nargs='+', help='<Required> Set flag', required=False, type=int, default=[2021, 2022])
# parser.add_argument('-ow','--overwrite', nargs=1, required=False, default= True)
parser.add_argument('-ow', '--overwrite', action='store_true', help='Overwrite existing files', required=False)

args = parser.parse_args()

overwrite = args.overwrite
startyr = args.years[0]
# endyr = args.years[1]
endyr = args.years[1] + 1  # Include the end year in the range

# Decision sobre cual es la region de un aglomerado. GBA tiene que ir a Gran Buenos Aires, aunque algunos de sus radios en partidos como Rodriguez, Escobar, etc sean region pampeana.
# Viedma Patagones, se tendria que tirar de un lado, y la mayoria de sus radios, son Patagonia.
# Se tiene que corregir a mano, porque el AGLO 0 SI tiene varias regiones.
AGLO_Region = pd.read_csv('./data/info/radio_ref.csv', usecols = ['AGLOMERADO', 'Region']).drop_duplicates()

AGLO_Region = AGLO_Region.loc[~((AGLO_Region.AGLOMERADO == 33) & (AGLO_Region.Region == 'Pampeana'))]
AGLO_Region = AGLO_Region.loc[~((AGLO_Region.AGLOMERADO == 93) & (AGLO_Region.Region == 'Pampeana'))]


####  NOMBRES DE VARIABLES: CENSO - EPH
names_censo = ['IX_TOT', 'P02', 'P03', 'CONDACT', 'AGLOMERADO',
    'V01', 'H05', 'H06', 'H07', 'H08', 'H09', 'H10', 'H11', 'H12', 'H16', 'H15', 'PROP', 'H14', 'H13',
      'P07', 'P08', 'P09', 'P10', 'P05']

names_EPH = ['IX_TOT','CH04','CH06','CONDACT', 'AGLOMERADO',
    'IV1', 'IV3', 'IV4','IV5','IV6','IV7','IV8','IV10','IV11','II1','II2','II7','II8','II9',
    'CH09','CH10','CH12','CH13','CH15']

####  NOMBRES DE VARIABLES MONETARIAS
col_mon = [u'P21', u'P47T', u'PP08D1', u'TOT_P12', u'T_VI', u'V12_M', u'V2_M', u'V3_M', u'V5_M']

# Load CPI Data
cpi_url = 'https://raw.githubusercontent.com/matuteiglesias/IPC-Argentina/main/data/info/indice_precios_Q.csv'
cpi = pd.read_csv(cpi_url, index_col=0)
cpi.index = pd.to_datetime(cpi.index)
cpi = cpi['2003':]
## Forzar dia 15 del mes
cpi.index = cpi.index - pd.offsets.MonthBegin(1) + pd.offsets.Day(14)

## MENSUAL

cpi_M_url = 'https://raw.githubusercontent.com/matuteiglesias/IPC-Argentina/main/data/info/indice_precios_M.csv'
cpi_M = pd.read_csv(cpi_M_url, index_col=0)[['index']]
cpi_M.index = pd.to_datetime(cpi_M.index)
cpi_M = cpi_M['2003':]

## Indice
ix = cpi_M.loc['2016-01'].values[0][0]

if not os.path.exists('./data/training/'):
    os.makedirs('./data/training/')
    
######################################################################
###########   LOOP PRINCIPAL. CREACION DE TRAINING SETS    ###########
######################################################################
# Main Loop: Create Training Sets
path ='./../microdatos-EPH-INDEC/microdatos/' # depende de donde hayamos descargado los microdatos
# path ='./EPH/microdatos/' # depende de donde hayamos descargado los microdatos

# Temporary directory to store downloaded data
temp_dir = './temp_data/'

if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)
    
for y in range(startyr, endyr):
    print(y)
    yr = str(y)[2:]
    training_file = './data/training/EPHARG_train_'+str(yr)+'.csv'
    
    # Si todavia no existe la training data de ese anio, o si la opcion overwrite esta activada:
    if not os.path.exists(training_file) or overwrite:
        allFiles = glob.glob(path + 'hogar/*'+str(yr)+'.txt')
        frame = pd.DataFrame()
        list_ = []

        
        for file_ in allFiles:
            df = pd.read_csv(file_,index_col=None, header=0, delimiter = ';',
                            usecols = ['CODUSU','ANO4','TRIMESTRE','IX_TOT', 'AGLOMERADO',
            'IV1', 'IV3', 'IV4','IV5','IV6','IV7','IV8','IV10','IV11','II1','II2','II7','II8','II9']) 

            print(len(df))
            list_ += [df]
        df = pd.concat(list_)

        # Correcciones Respuestas. Para que matchee censo
        df = df.loc[df.IV1 != 9]
        df['IV10'] = df['IV10'].map({1: 1, 2: 2, 3: 2, 0: 0, 9: 9})
        df['II9'] = df['II9'].map({1: 1, 2: 2, 3: 2, 4: 4, 0: 0})
        df['II7'] = df['II7'].map({1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 6, 8: 6, 9: 6, 0: 0})
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

        indiv = df
        indiv = indiv.dropna(subset = ['P47T'])
        print(indiv.shape)

        indiv_table = indiv[list(indiv.columns.difference(hogar.columns)) + ['CODUSU', 'ANO4', 'TRIMESTRE', 'AGLOMERADO']]
        EPH = hogar.merge(indiv_table, on = ['CODUSU', 'ANO4', 'TRIMESTRE', 'AGLOMERADO'])#, indicator = True)

        print('Hogar - Indiv merged:')
        print(EPH.shape)

        EPH = EPH.merge(AGLO_Region)

        EPH_no_aglo = EPH.copy(); 
        EPH_no_aglo['AGLOMERADO'] = 0

        EPH = pd.concat([EPH, EPH_no_aglo]).reset_index(drop = True)

        print('No aglo agregado:')
        print(EPH.shape)

    #     # Quarters / deflation

        # Quarters / deflation
        EPH['Q'] = EPH.ANO4.astype(str) + ':' + (3*EPH.TRIMESTRE).astype(str)
        EPH['Q'] = pd.to_datetime(EPH['Q'], format='%Y:%m') - pd.DateOffset(months=1) + pd.DateOffset(days=14)
        print(EPH['Q'].unique())


    #     EPH[col_mon] = cpi_mes_actual*EPH[col_mon].div(EPH[['Q'] + col_mon].merge(cpi, on = 'Q', how = 'left')['index'].values, 0)
        EPH[col_mon] = ix*EPH[col_mon].div(EPH[['Q'] + col_mon].merge(cpi, on = 'Q', how = 'left')['index'].values, 0)
        EPH[col_mon] = EPH[col_mon].round()

        print('deflactado:')
        print(EPH.shape)
        training = EPH.rename(columns = dict(zip(names_EPH, names_censo)))
        
        # remove bad observations
        training = training.loc[training.P47T >= -0.001].fillna(0)
        
        for col in ['CAT_OCUP', 'CH07', 'PP07G1', 'PP07G_59', 'PP07I', 'PP07J', 'PP07K']:
            training = training.loc[training[col] != 9]

        ### RANKING AGLOMERADO
        AGLO_rk = training.loc[(training.CAT_OCUP == 3) & (training.P47T >= 100)].groupby(['ANO4', 'AGLOMERADO'])[['P47T']].mean()
        AGLO_rk['AGLO_rk'] = AGLO_rk.rank(pct = True).round(3)
        AGLO_rk = AGLO_rk.sort_values('P47T').reset_index()

        ### RANKING REGION
        Reg_rk = training.loc[(training.CAT_OCUP == 3) & (training.P47T >= 100)].groupby(['ANO4', 'Region'])[['P47T']].mean()
        Reg_rk['Reg_rk'] = Reg_rk.rank(pct = True).round(3)
        Reg_rk = Reg_rk.sort_values('P47T').reset_index()
            
        training = training.merge(AGLO_rk[['ANO4', 'AGLOMERADO', 'AGLO_rk']]).merge(Reg_rk[['ANO4', 'Region', 'Reg_rk']])
        
        ## Crear columnas binarias para ingreso.
        training['INGRESO'] = (training.P47T > 100).astype(int)
        training['INGRESO_NLB'] = (training.T_VI > 100).astype(int)
        training['INGRESO_JUB'] = (training.V2_M > 100).astype(int)
        training['INGRESO_SBS'] = (training.V5_M > 100).astype(int)
        
        ## Ordenar por id de hogar.
        training = training.sort_values('CODUSU')
        
        training.to_csv(training_file, index = False)

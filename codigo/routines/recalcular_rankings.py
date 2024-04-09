# %% [markdown]
# ### Ranking de AGLOS y Regiones
# 
# A continuacion se extrae el ranking de aglomerados y regiones para cada uno de los anios.

# %%
import pandas as pd
import os

# %%
aglo_list = []
regs_list = []

startyr = 2023; endyr = 2025;
for y in range(startyr, endyr):
    print(y)
    yr = str(y)[2:]
    training_file = './../../data/training/EPHARG_train_'+str(yr)+'.csv'

    if not os.path.exists(training_file):
        continue
    
    aglo_table = pd.read_csv(training_file, usecols = ['ANO4', 'AGLOMERADO', 'AGLO_rk']).drop_duplicates()
    aglo_list += [aglo_table]
    
    regs_table = pd.read_csv(training_file, usecols = ['ANO4', 'Region', 'Reg_rk']).drop_duplicates()
    regs_list += [regs_table]
    
aglo_rk = pd.concat(aglo_list)
regs_rk = pd.concat(regs_list)

aglo_rk.to_csv('./../../data/info/AGLO_rk', index = False)



# # Define the mapping dictionary
regiones = {
    'Gran Buenos Aires': 'gran_buenos_aires',
    'Pampeana': 'pampeana',
    'Noroeste': 'noroeste',
    'Noreste': 'noreste',
    'Patagónica': 'patagonia',
    'Cuyo': 'cuyo'
}

# Update region names using the mapping dictionary
regs_rk['region_'] = regs_rk['Region'].map(regiones)
regs_rk.to_csv('./../../data/info/Reg_rk', index = False)

# %% [markdown]
# ## Listo. Salvado el training set.

# %% [markdown]
# 



# %% [markdown]
# ### Ranking de AGLOS y Regiones
# 
# A continuacion se extrae el ranking de aglomerados y regiones para cada uno de los anios.

# %%
import pandas as pd
import os

import pandas as pd
import os

# Initialize lists to store data
aglo_list = []
regs_list = []

startyr = 2023
endyr = 2025

# Loop through the specified years
for y in range(startyr, endyr):
    print(f"Processing year: {y}")
    yr = str(y)[2:]
    training_file = f'./data/training/EPHARG_train_{yr}.csv'

    # Check if the training file exists
    if not os.path.exists(training_file):
        print(f"Training file not found for year {y}: {training_file}")
        continue

    try:
        # Read agglomerate data
        aglo_table = pd.read_csv(training_file, usecols=['ANO4', 'AGLOMERADO', 'AGLO_rk']).drop_duplicates()
        aglo_list.append(aglo_table)
        print(f"Read agglomerate data for year {y}, {len(aglo_table)} records")

        # Read region data
        regs_table = pd.read_csv(training_file, usecols=['ANO4', 'Region', 'Reg_rk']).drop_duplicates()
        regs_list.append(regs_table)
        print(f"Read region data for year {y}, {len(regs_table)} records")
    
    except Exception as e:
        print(f"Error processing file for year {y}: {e}")

# Concatenate agglomerate data
if aglo_list:
    aglo_rk = pd.concat(aglo_list)
    aglo_rk.to_csv('./data/info/AGLO_rk.csv', index=False)
    print(f"Saved agglomerate rankings with {len(aglo_rk)} records")
else:
    print("No agglomerate data to concatenate")

# Concatenate region data
if regs_list:
    regs_rk = pd.concat(regs_list)
    print(f"Saved region rankings with {len(regs_rk)} records")

    # Define the mapping dictionary for region names
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
    regs_rk.to_csv('./data/info/Reg_rk.csv', index=False)
    print(f"Region names mapped and saved to './data/info/Reg_rk.csv'")
else:
    print("No region data to concatenate")

print("Processing complete.")

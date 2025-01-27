import nbformat as nbf

# Codice da includere nel notebook
code = """import pandas as pd
import os
# Caricamento dei file CSV con gestione del separatore di migliaia
df_2021 = pd.read_csv('dataset/dataset2021.csv', sep=';', thousands='.')
df_2022 = pd.read_csv('dataset/dataset2022.csv', sep=';', thousands='.')
df_2023 = pd.read_csv('dataset/dataset2023.csv', sep=';', thousands='.')

# Aggiunta della colonna "Anno" per ciascun dataset
df_2021['Anno'] = 2021
df_2022['Anno'] = 2022
df_2023['Anno'] = 2023

# Concatenazione dei dataset in un unico dataframe
df_unico = pd.concat([df_2021, df_2022, df_2023], ignore_index=True)

# Salvataggio del dataset combinato in un nuovo file CSV
df_unico.to_csv('dataset/dataset_completo.csv', index=False, sep=';', float_format='%.3f')

# Ricaricamento del dataset per verificare il contenuto
dataset_completo = pd.read_csv('dataset/dataset_completo.csv', sep=';', thousands='.')

# Visualizzazione del dataset finale
dataset_completo
"""

# Crea un nuovo notebook
nb = nbf.v4.new_notebook()

# Aggiungi una cella di codice al notebook
nb.cells.append(nbf.v4.new_code_cell(code))

# Salva il notebook come file .ipynb
with open("wastegone.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook salvato correttamente come 'wastegone.ipynb'")
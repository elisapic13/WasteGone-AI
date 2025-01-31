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

# Trova i valori nulli (NaN)
nan_mask = dataset_completo.isna()

# Trova i valori uguali a "-"
dash_mask = dataset_completo.applymap(lambda x: x == " -   " or x == " N.C. " or x == "NC")

# Somma i valori nulli (NaN) e i valori uguali a "-"
nan_count = nan_mask.sum()
dash_count = dash_mask.sum()

# Combina i risultati
total_count = nan_count + dash_count

total_count

# Colonne da verificare
columns_to_check = [
    "Kg di rifiuti differenziati (RDi)",
    "Kg di compostaggio domestico",
    "Kg di rifiuti non differenziati (RUind)",
    "Totale Kg di rifiuti prodotti (RDi+comp+RUind)",
    "Produzione R.U. pro capite annua in Kg",
    "%RD",
    "Tasso di riciclaggio"
]

# Filtra le righe che contengono "N.C." in almeno una delle colonne specificate

filtered_rows = dataset_completo[dataset_completo[columns_to_check].applymap(lambda x: x == " N.C. " or x =="NC" or x == " NC ").any(axis=1)]

# Visualizza le righe filtrate
filtered_rows

# Elimina le righe trovate dal dataset completo
dataset_completo = dataset_completo[~dataset_completo.index.isin(filtered_rows.index)]

# Salva il dataset aggiornato
dataset_completo.to_csv('dataset/dataset_completo_pulito.csv', index=False, sep=';')

# Visualizza il dataset aggiornato
dataset_completo

# Eliminazione delle colonne specificate
dataset_completo = dataset_completo.drop(columns=['Kg di compostaggio domestico', 'Provincia','%RD', 'Tasso di riciclaggio'])

# Visualizzazione del dataset finale senza le colonne eliminate
dataset_completo

# Conta le occorrenze di ciascun comune
comune_counts = dataset_completo['Comune'].value_counts()

# Filtra i comuni che appaiono almeno 3 volte
comuni_da_tenere = comune_counts[comune_counts >= 3].index

# Crea un nuovo dataset mantenendo solo i comuni desiderati
dataset_filtrato = dataset_completo[dataset_completo['Comune'].isin(comuni_da_tenere)]

# Salva il nuovo dataset su file (opzionale)
dataset_filtrato.to_csv('dataset/dataset_filtrato.csv', index=False, sep=';', encoding='utf-8')

# Mostra il dataset filtrato
dataset_filtrato

"""

# Crea un nuovo notebook
nb = nbf.v4.new_notebook()

# Aggiungi una cella di codice al notebook
nb.cells.append(nbf.v4.new_code_cell(code))

# Salva il notebook come file .ipynb
with open("wastegone.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook salvato correttamente come 'wastegone.ipynb'")
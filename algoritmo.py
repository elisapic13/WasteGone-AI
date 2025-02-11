import nbformat as nbf

# Codice da includere nel notebook
code = r"""import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Carica il dataset specificando il separatore come punto e virgola
df = pd.read_csv('dataset/dataset_filtrato.csv', delimiter=';')

# Funzione per pulire le colonne con percentuali
def clean_percentage_column(col):
    return col.str.replace('%', '').str.replace(',', '.').astype(float)

# Funzione per pulire i numeri da separatori di migliaia e spazi
def clean_numeric_column(col):
    col = col.replace('-', np.nan)
    col = col.str.replace(' ', '').str.replace('.', '')
    return pd.to_numeric(col, errors='coerce')

# Funzione per rimuovere gli outlier utilizzando l'IQR
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.3 * IQR
    upper_bound = Q3 + 1.3 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Puliamo le colonne che contengono percentuali
df['%RD'] = clean_percentage_column(df['%RD'])
df['Tasso di riciclaggio'] = clean_percentage_column(df['Tasso di riciclaggio'])

# Puliamo le colonne target che contengono numeri con separatori di migliaia
df['Kg di rifiuti differenziati (RDi)'] = clean_numeric_column(df['Kg di rifiuti differenziati (RDi)'])
df['Kg di rifiuti non differenziati (RUind)'] = clean_numeric_column(df['Kg di rifiuti non differenziati (RUind)'])

# Separiamo il training set (2021-2022) e il test set (2023)
train_df = df[df['Anno'].isin([2021, 2022])]  # Usa solo i dati del 2021 e 2022 per l'allenamento
test_df = df[df['Anno'] == 2023]  # I dati del 2023 sono utilizzati per la valutazione

# Variabili indipendenti (features)
features = ['Anno', 'Abitanti', '%RD', 'Tasso di riciclaggio', 'Produzione R.U. pro capite annua in Kg']

# Variabili target
target_diff = 'Kg di rifiuti differenziati (RDi)'
target_non_diff = 'Kg di rifiuti non differenziati (RUind)'

# Creiamo i set di training e test
X_train = train_df[features]
y_train_diff = train_df[target_diff]
y_train_non_diff = train_df[target_non_diff]

X_test = test_df[features]
y_test_diff = test_df[target_diff]
y_test_non_diff = test_df[target_non_diff]


"""

# Crea un nuovo notebook
nb = nbf.v4.new_notebook()

# Aggiungi una cella di codice al notebook
nb.cells.append(nbf.v4.new_code_cell(code))

# Salva il notebook come file .ipynb
with open("algoritmo.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook salvato correttamente come 'algoritmo.ipynb'")

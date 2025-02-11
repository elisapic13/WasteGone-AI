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

# Rimuoviamo le righe con NaN nei target e nelle predizioni
train_df_clean = train_df.dropna(subset=[target_diff, target_non_diff])
test_df_clean = test_df.dropna(subset=[target_diff, target_non_diff])

# Rimuoviamo gli outlier per i target
train_df_clean = remove_outliers(train_df_clean, target_diff)
train_df_clean = remove_outliers(train_df_clean, target_non_diff)
test_df_clean = remove_outliers(test_df_clean, target_diff)
test_df_clean = remove_outliers(test_df_clean, target_non_diff)

# Creiamo i set puliti di training e test dopo aver rimosso gli outlier
X_train_clean = train_df_clean[features]
y_train_diff_clean = train_df_clean[target_diff]
y_train_non_diff_clean = train_df_clean[target_non_diff]

X_test_clean = test_df_clean[features]
y_test_diff_clean = test_df_clean[target_diff]
y_test_non_diff_clean = test_df_clean[target_non_diff]

# Crea e allena il modello Random Forest per i rifiuti differenziati
model_diff = RandomForestRegressor(n_estimators=100, random_state=42)
model_diff.fit(X_train_clean, y_train_diff_clean)

# Predizioni sul test set per i rifiuti differenziati (per il 2023)
y_pred_diff_clean = model_diff.predict(X_test_clean)

# Crea e allena il modello Random Forest per i rifiuti non differenziati
model_non_diff = RandomForestRegressor(n_estimators=100, random_state=42)
model_non_diff.fit(X_train_clean, y_train_non_diff_clean)

# Predizioni sul test set per i rifiuti non differenziati (per il 2023)
y_pred_non_diff_clean = model_non_diff.predict(X_test_clean)

# Creiamo una copia di test_df_clean per evitare il SettingWithCopyWarning
test_df_clean = test_df_clean.copy()

# Aggiungiamo le predizioni al DataFrame di test pulito per il 2023
test_df_clean['Predizione Rifiuti Differenziati'] = y_pred_diff_clean
test_df_clean['Predizione Rifiuti Non Differenziati'] = y_pred_non_diff_clean

# Calcoliamo l'errore quadratico medio per i rifiuti differenziati
mse_diff_clean = mean_squared_error(y_test_diff_clean, y_pred_diff_clean)
print(f'MSE per i rifiuti differenziati per il 2023: {mse_diff_clean}')

# Calcoliamo l'errore quadratico medio per i rifiuti non differenziati
mse_non_diff_clean = mean_squared_error(y_test_non_diff_clean, y_pred_non_diff_clean)
print(f'MSE per i rifiuti non differenziati per il 2023: {mse_non_diff_clean}')

# Calcoliamo il MAE per i rifiuti differenziati
mae_diff_clean = mean_absolute_error(y_test_diff_clean, y_pred_diff_clean)
print(f'MAE per i rifiuti differenziati per il 2023: {mae_diff_clean}')

# Calcoliamo il MAE per i rifiuti non differenziati
mae_non_diff_clean = mean_absolute_error(y_test_non_diff_clean, y_pred_non_diff_clean)
print(f'MAE per i rifiuti non differenziati per il 2023: {mae_non_diff_clean}')

# Calcoliamo il R^2 per i rifiuti differenziati
r2_diff_clean = r2_score(y_test_diff_clean, y_pred_diff_clean)
print(f'R^2 per i rifiuti differenziati per il 2023: {r2_diff_clean}')

# Calcoliamo il R^2 per i rifiuti non differenziati
r2_non_diff_clean = r2_score(y_test_non_diff_clean, y_pred_non_diff_clean)
print(f'R^2 per i rifiuti non differenziati per il 2023: {r2_non_diff_clean}')

"""

# Crea un nuovo notebook
nb = nbf.v4.new_notebook()

# Aggiungi una cella di codice al notebook
nb.cells.append(nbf.v4.new_code_cell(code))

# Salva il notebook come file .ipynb
with open("algoritmo.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook salvato correttamente come 'algoritmo.ipynb'")

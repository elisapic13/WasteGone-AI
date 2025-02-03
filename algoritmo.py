import nbformat as nbf

# Codice da includere nel notebook
code = r"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Caricare il dataset con rilevamento automatico del separatore
df = pd.read_csv("dataset/dataset_filtrato.csv", sep=None, engine='python')

# Stampiamo i nomi delle colonne per verificare che siano corretti
print("Nomi colonne nel dataset:", df.columns.tolist())

# Rimuoviamo spazi extra e standardizziamo i nomi delle colonne
df.columns = df.columns.str.strip().str.lower()

# Stampiamo i nuovi nomi delle colonne
print("Nomi colonne puliti:", df.columns.tolist())

# Selezioniamo le colonne numeriche corrette
cols_to_convert = [
    'kg di rifiuti differenziati (rdi)',
    'kg di rifiuti non differenziati (ruind)',
    'totale kg di rifiuti prodotti (rdi+ruind)'
]

# Normalizziamo i nomi delle colonne per sicurezza
cols_to_convert = [col.strip().lower() for col in cols_to_convert]

# Pulizia dei dati (rimozione caratteri non numerici e conversione)
for col in cols_to_convert:
    df[col] = df[col].astype(str).str.replace(r'[^\d.,]', '', regex=True).str.replace(',', '.')

# Conversione in numerico
df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')

# Rimuoviamo righe con valori NaN dopo la conversione
df = df.dropna()

# Controlliamo se ci sono outlier evidenti
print("Valori massimi:\n", df[cols_to_convert].max())

# Definiamo le variabili indipendenti (X) e la variabile dipendente (y)
X = df[['kg di rifiuti differenziati (rdi)', 'kg di rifiuti non differenziati (ruind)']]
y = df['totale kg di rifiuti prodotti (rdi+ruind)']

# Standardizzazione delle feature (Z-score)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Suddivisione in training e test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

"""

# Crea un nuovo notebook
nb = nbf.v4.new_notebook()

# Aggiungi una cella di codice al notebook
nb.cells.append(nbf.v4.new_code_cell(code))

# Salva il notebook come file .ipynb
with open("algoritmo.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook salvato correttamente come 'algoritmo.ipynb'")

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

# Creazione e addestramento del modello
model = LinearRegression()
model.fit(X_train, y_train)

# Predizioni per il test set
y_pred_test = model.predict(X_test)

# Predizioni su tutto il dataset
df['y_pred'] = model.predict(X_scaled)

# Calcolare le metriche di valutazione
mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_test)

# Stampare i risultati
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")

# Coefficienti del modello
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficiente'])
print(coefficients)

# Grafico migliorato
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, alpha=0.7, label="Valori Predetti")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label="Ideale")
plt.xlabel("Valori Reali")
plt.ylabel("Valori Predetti")
plt.title("Regressione Lineare Multipla (Standardizzata)")
plt.legend()
plt.show()

# Salva il dataset con le predizioni
df.to_csv("dataset/dataset_predizioni.csv", index=False)

"""

# Crea un nuovo notebook
nb = nbf.v4.new_notebook()

# Aggiungi una cella di codice al notebook
nb.cells.append(nbf.v4.new_code_cell(code))

# Salva il notebook come file .ipynb
with open("algoritmo.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook salvato correttamente come 'algoritmo.ipynb'")

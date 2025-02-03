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


# Identificare le colonne chiave
col_anno = 'anno'
col_comune = 'comune'
col_x1 = 'kg di rifiuti differenziati (rdi)'
col_x2 = 'kg di rifiuti non differenziati (ruind)'
col_target = 'totale kg di rifiuti prodotti (rdi+ruind)'

# Funzione per convertire stringhe in numeri gestendo errori
def convert_to_float(series):
    return pd.to_numeric(series.astype(str)
                         .str.replace('.', '', regex=False)
                         .str.replace(',', '.', regex=False)
                         .str.strip(), errors='coerce')  # Sostituisce errori con NaN

# Convertiamo le colonne numeriche
df[col_x1] = convert_to_float(df[col_x1])
df[col_x2] = convert_to_float(df[col_x2])
df[col_target] = convert_to_float(df[col_target])

# Rimuoviamo eventuali righe con valori NaN dopo la conversione
df.dropna(subset=[col_x1, col_x2, col_target], inplace=True)

def predict_2024(group):
    if len(group) < 2:
        return np.nan  # Non possiamo fare una previsione con meno di 2 dati
    
    model = LinearRegression()
    X = group[[col_x1, col_x2]]
    y = group[col_target]
    model.fit(X, y)
    
    X_pred = group[[col_x1, col_x2]].iloc[-1].values.reshape(1, -1)  # Usa i dati più recenti
    return model.predict(X_pred)[0]

# Applicare il modello per ogni comune
df_pred = df.groupby(col_comune).apply(predict_2024).reset_index()
df_pred.columns = [col_comune, 'predizione_2024']

# Riporta i numeri nel formato originale con il punto come separatore delle migliaia e la virgola per i decimali
df_pred['predizione_2024'] = df_pred['predizione_2024'].apply(lambda x: f"{x:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))

# Salvare il dataset con le predizioni
df_pred.to_csv("dataset/predizioni_2024.csv", index=False)

print(df.head())


"""

# Crea un nuovo notebook
nb = nbf.v4.new_notebook()

# Aggiungi una cella di codice al notebook
nb.cells.append(nbf.v4.new_code_cell(code))

# Salva il notebook come file .ipynb
with open("algoritmo.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook salvato correttamente come 'algoritmo.ipynb'")

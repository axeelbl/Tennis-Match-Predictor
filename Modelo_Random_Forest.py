from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os
import pandas as pd
import numpy as np

# Cargar archivo
filename = "tennis_model_dataset.csv"
folder_path = '/Users/Stikets/Desktop/Axel/python/'
full_path = os.path.join(folder_path, filename)

df_modelo = pd.read_csv(full_path)
print("✅ Archivo leído")

# Copia para trabajar
df_ml = df_modelo.copy()

# Codificar columnas categóricas (p1_hand, p2_hand)
le_hand = LabelEncoder()
df_ml['p1_hand'] = le_hand.fit_transform(df_ml['p1_hand'])
df_ml['p2_hand'] = le_hand.transform(df_ml['p2_hand'])

# Seleccionar columnas explícitamente
features = [
    'p1_rank', 'p2_rank', 'p1_age', 'p2_age', 'p1_hand', 'p2_hand',
    'p1_ace', 'p2_ace', 'elo_p1', 'elo_p2',
    'h2h_p1_vs_p2', 'h2h_p2_vs_p1',
    'p1_recent_wins', 'p2_recent_wins',
    'p1_surface_wr', 'p2_surface_wr',
    'tourney_A', 'tourney_D', 'tourney_F', 'tourney_G', 'tourney_M', 'tourney_O'
]

X = df_ml[features]
y = df_ml['target']

# Dividir en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
print("🤖 Entrenando modelo Random Forest ...")
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Predecir y evaluar
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"🔍 Precisión del modelo: {accuracy:.4f}")

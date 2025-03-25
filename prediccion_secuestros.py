# prediccion_secuestros_corregido.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import folium
from folium.plugins import HeatMap

# --- 1. Dataset simulado mejorado ---
localidades_posibles = ["El Poblado", "Comuna 13", "La Candelaria", "Bello", "Robledo"]
data = {
    "localidad": np.random.choice(localidades_posibles, size=100),
    "hora_dia": np.random.choice(["diurno", "nocturno"], size=100),
    "distancia_policia": np.round(np.random.uniform(0.1, 5.0, size=100)),
    "indice_pobreza": np.random.randint(10, 90, size=100),
    "camaras_vigilancia": np.random.randint(0, 15, size=100),
    "secuestro": np.random.choice([0, 1], size=100, p=[0.7, 0.3])
}

df = pd.DataFrame(data)

# --- 2. Preprocesamiento robusto ---
# Definir columnas categóricas y numéricas
categorical_features = ["localidad", "hora_dia"]
numeric_features = ["distancia_policia", "indice_pobreza", "camaras_vigilancia"]

# Crear transformadores
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combinar en un ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# --- 3. Modelo y Pipeline ---
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        scale_pos_weight=(len(df) - sum(df["secuestro"])) / sum(df["secuestro"]),
        eval_metric='logloss',
        use_label_encoder=False
    ))
])

# Dividir datos
X = df.drop("secuestro", axis=1)
y = df["secuestro"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar modelo
model.fit(X_train, y_train)

# Evaluación
y_pred = model.predict(X_test)
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# --- 4. Predicción para nueva zona ---
nueva_zona = {
    "localidad": "Comuna 13",
    "hora_dia": "nocturno",
    "distancia_policia": 2.5,
    "indice_pobreza": 70,
    "camaras_vigilancia": 2
}

nueva_zona_df = pd.DataFrame([nueva_zona])

# Predecir probabilidad
riesgo = model.predict_proba(nueva_zona_df)[0][1]
print(f"\nRiesgo estimado para Comuna 13 (noche): {riesgo:.2%}")

# --- 5. Mapa de calor mejorado ---
coords = {
    "El Poblado": [6.2088, -75.5704],
    "Comuna 13": [6.2486, -75.5739],
    "La Candelaria": [6.2444, -75.5732],
    "Bello": [6.3357, -75.5585],
    "Robledo": [6.2598, -75.6003]
}

# Crear mapa
mapa = folium.Map(location=[6.2444, -75.5732], zoom_start=12)

# Añadir puntos de riesgo
for idx, row in df.iterrows():
    if row["localidad"] in coords:
        folium.CircleMarker(
            location=coords[row["localidad"]],
            radius=5 + row["secuestro"]*5,  # Tamaño según riesgo
            color='red' if row["secuestro"] == 1 else 'green',
            fill=True,
            fill_opacity=0.7,
            popup=f"Localidad: {row['localidad']}<br>Riesgo: {row['secuestro']}"
        ).add_to(mapa)

# Guardar mapa
mapa.save("medellin_riesgo_corregido.html")
print("\n¡Mapa generado como 'medellin_riesgo_corregido.html'!")
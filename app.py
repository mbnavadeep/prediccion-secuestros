# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import folium
from streamlit_folium import folium_static

# Configuración de la página
st.set_page_config(page_title="Predicción de Secuestros", layout="wide")
st.title("🚨 Predicción de Riesgo de Secuestro por Localidad")

# --- 1. Cargar Modelo (simulado) ---
@st.cache_resource
def load_model():
    # Datos simulados (reemplazar con tu modelo entrenado)
    localidades = ["El Poblado", "Comuna 13", "La Candelaria", "Bello", "Robledo"]
    
    # Preprocesamiento
    numeric_features = ["distancia_policia", "indice_pobreza", "camaras_vigilancia"]
    categorical_features = ["localidad", "hora_dia"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Modelo dummy (entrenar con tus datos reales)
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(eval_metric='logloss', use_label_encoder=False))
    ])
    
    # Datos de ejemplo para "entrenar"
    df = pd.DataFrame({
        "localidad": np.random.choice(localidades, 50),
        "hora_dia": np.random.choice(["diurno", "nocturno"], 50),
        "distancia_policia": np.random.uniform(0.1, 5.0, 50),
        "indice_pobreza": np.random.randint(10, 90, 50),
        "camaras_vigilancia": np.random.randint(0, 15, 50),
        "secuestro": np.random.choice([0, 1], 50)
    })
    
    X = df.drop("secuestro", axis=1)
    y = df["secuestro"]
    model.fit(X, y)
    
    return model, localidades

model, localidades = load_model()

# --- 2. Sidebar para Inputs ---
st.sidebar.header("📊 Parámetros de Entrada")
localidad = st.sidebar.selectbox("Localidad", localidades)
hora_dia = st.sidebar.radio("Hora del día", ["diurno", "nocturno"])
distancia_policia = st.sidebar.slider("Distancia a comisaría (km)", 0.1, 10.0, 2.5)
indice_pobreza = st.sidebar.slider("Índice de pobreza", 0, 100, 50)
camaras_vigilancia = st.sidebar.slider("Cámaras de vigilancia", 0, 20, 5)

# --- 3. Predicción ---
if st.sidebar.button("Predecir Riesgo"):
    input_data = pd.DataFrame([{
        "localidad": localidad,
        "hora_dia": hora_dia,
        "distancia_policia": distancia_policia,
        "indice_pobreza": indice_pobreza,
        "camaras_vigilancia": camaras_vigilancia
    }])
    
    riesgo = model.predict_proba(input_data)[0][1]
    
    # Mostrar resultado
    st.success(f"**Riesgo estimado en {localidad} ({hora_dia}): {riesgo:.2%}**")
    st.metric("Nivel de Alerta", "ALTO" if riesgo > 0.7 else "MEDIO" if riesgo > 0.4 else "BAJO")

    # --- 4. Mapa Interactivo ---
    st.subheader("🗺️ Mapa de Riesgo")
    coords = {
        "El Poblado": [6.2088, -75.5704],
        "Comuna 13": [6.2486, -75.5739],
        "La Candelaria": [6.2444, -75.5732],
        "Bello": [6.3357, -75.5585],
        "Robledo": [6.2598, -75.6003]
    }
    
    mapa = folium.Map(location=[6.2444, -75.5732], zoom_start=12)
    
    # Marcador para la localidad seleccionada
    folium.Marker(
        location=coords[localidad],
        popup=f"<b>{localidad}</b><br>Riesgo: {riesgo:.2%}",
        icon=folium.Icon(color="red" if riesgo > 0.7 else "orange" if riesgo > 0.4 else "green")
    ).add_to(mapa)
    
    folium_static(mapa, width=800)

# --- 5. Instrucciones ---
st.markdown("""
### 📌 Instrucciones:
1. Selecciona los parámetros en el panel izquierdo.
2. Haz clic en **"Predecir Riesgo"**.
3. Visualiza el resultado y el mapa.
""")
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import joblib
import os

# --- CARGA DE MODELO Y DATOS ---
@st.cache_resource
def load_all():
    df = pd.read_csv("descripciones_caf.csv", sep=';', encoding='utf-8-sig')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    classifier = joblib.load("modelos_caf/classifier.pkl")
    embeddings = joblib.load("modelos_caf/embeddings.pkl")
    return df, model, classifier, embeddings

df, model, classifier, embeddings = load_all()

# --- INTERFAZ ---
st.title("🛠️ Asistente de Diagnóstico de Frenos (CAF)")
st.write("Introduce una descripción o síntoma para obtener casos similares y sugerencias.")

query = st.text_input("Descripción o síntoma del problema:", "")

if query:
    q_emb = model.encode(query, convert_to_tensor=True)
    sims = util.pytorch_cos_sim(q_emb, embeddings)[0]
    top_idx = sims.topk(5).indices.tolist()
    st.subheader("🔍 Casos similares encontrados:")
    for i in top_idx:
        st.write(f"• {df.iloc[i]['descripcion_averia']}")

    st.divider()
    st.subheader("💡 Clasificación sugerida:")
    pred = classifier.predict([query])[0]
    st.success(f"Posible causa: **{pred}**")

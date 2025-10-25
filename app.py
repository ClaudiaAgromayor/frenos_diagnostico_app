import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# --- CARGA DE MODELO Y DATOS ---
@st.cache_resource
def load_all():
    # cargar datos
    df = pd.read_csv("descripciones_caf.csv", sep=';', encoding='utf-8-sig')
    
    # cargar modelos con extensión correcta
    classifier = joblib.load("modelos_caf/classifier_logreg.pkl")
    label_encoder = joblib.load("modelos_caf/label_encoder.pkl")
    tfidf_vectorizer = joblib.load("modelos_caf/tfidf_vectorizer.pkl")
    
    # vectorizar todas las descripciones para similitud
    X = tfidf_vectorizer.transform(df['descripcion_averia'].astype(str))
    
    return df, classifier, label_encoder, tfidf_vectorizer, X

df, classifier, label_encoder, tfidf_vectorizer, X = load_all()

# --- INTERFAZ ---
st.title("🛠️ Asistente de Diagnóstico de Frenos (CAF)")
st.write("Introduce una descripción o síntoma para obtener casos similares y sugerencias.")

query = st.text_input("Descripción o síntoma del problema:", "")

if query:
    # vectorizar query
    query_vect = tfidf_vectorizer.transform([query])
    
    # --- CASOS SIMILARES ---
    sims = cosine_similarity(query_vect, X)[0]
    top_idx = sims.argsort()[-5:][::-1]
    
    st.subheader("🔍 Casos similares encontrados:")
    for i in top_idx:
        st.write(f"• {df.iloc[i]['descripcion_averia']}")
    
    st.divider()
    
    # --- CLASIFICACIÓN ---
    pred_num = classifier.predict(query_vect)[0]       # devuelve 14
    pred_label = label_encoder.inverse_transform([pred_num])[0]  # devuelve la etiqueta real

    st.subheader("💡 Clasificación sugerida:")
    st.success(f"Posible causa: **{pred_label}**")

# casos_frenos.py
# ==========================================
# Análisis de averías CAF centrado en FRE (frenos)
# Autor: Claudia Agromayor
# ==========================================

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
import pickle

# ------------------------------------------------------------
# 1. CARGA DE DATOS
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DESC_PATH = os.path.join(BASE_DIR, "descripciones_caf.csv")
DEF_PATH = os.path.join(BASE_DIR, "Definiciones clave Codigo actuacion_V6.csv")

print("📂 Cargando ficheros...")
desc = pd.read_csv(DESC_PATH, sep=';', encoding='utf-8-sig', engine='python')
defs = pd.read_csv(DEF_PATH, sep=';', encoding='utf-8-sig', engine='python')
print(f"Descripciones: {len(desc)} filas | Definiciones: {len(defs)} filas")

# ------------------------------------------------------------
# 2. LIMPIEZA Y FILTRADO
# ------------------------------------------------------------
def clean_text(s):
    if pd.isna(s):
        return ""
    return str(s).strip()

for c in ['descripcion_ot', 'descripcion_averia', 'descripcion_reparacion', 'comentarios']:
    if c in desc.columns:
        desc[c] = desc[c].fillna("").astype(str).apply(clean_text)
    else:
        desc[c] = ""

# Combinar columnas relevantes para análisis de texto
desc["text_for_retrieval"] = (
    desc["descripcion_ot"] + " || " +
    desc["descripcion_averia"] + " || " +
    desc["descripcion_reparacion"]
).str.strip()

# Filtrar solo problemas relacionados con FRE (frenos)
mask_fre = desc.apply(
    lambda r: any("FRE" in str(r.get(col, "")).upper() for col in desc.columns),
    axis=1
)
df_fre = desc[mask_fre].reset_index(drop=True)
print(f"🛞 Filas relacionadas con FRE (frenos): {len(df_fre)}")

corpus = desc["text_for_retrieval"].tolist()

# ------------------------------------------------------------
# 3. CREACIÓN DE EMBEDDINGS O TF-IDF
# ------------------------------------------------------------
use_st = False
try:
    from sentence_transformers import SentenceTransformer
    print("🔹 Cargando modelo de embeddings (all-MiniLM-L6-v2)...")
    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = st_model.encode(corpus, show_progress_bar=True, convert_to_numpy=True)
    use_st = True
except Exception as e:
    print("⚠️ No se pudo cargar SentenceTransformer, usando TF-IDF.")
    tfidf_for_emb = TfidfVectorizer(ngram_range=(1,2), max_features=20000)
    embeddings = tfidf_for_emb.fit_transform(corpus).toarray()

# ------------------------------------------------------------
# 4. ÍNDICE DE BÚSQUEDA (CASOS SIMILARES)
# ------------------------------------------------------------
print("🧭 Construyendo índice de búsqueda...")
nn = NearestNeighbors(n_neighbors=5, metric='cosine').fit(embeddings)

def retrieve(query, k=5):
    if use_st:
        q_emb = st_model.encode([query], convert_to_numpy=True)
    else:
        q_emb = tfidf_for_emb.transform([query]).toarray()
    D, I = nn.kneighbors(q_emb, n_neighbors=k)
    return D[0], I[0]

# Ejemplo de búsqueda
example_query = "ruido al frenar y pérdida de eficacia"
D, I = retrieve(example_query)
print("\n🔍 Ejemplo de búsqueda para:", example_query)
for dist, idx in zip(D, I):
    print(f"Distancia: {dist:.3f} | Descripción OT: {desc.loc[idx, 'descripcion_ot'][:120]}")
    print("---")

# ------------------------------------------------------------
# 5. CLASIFICADOR BASE (TF-IDF + Logistic Regression)
# ------------------------------------------------------------
print("\n⚙️ Entrenando clasificador base...")
labels_raw = desc["descripcion_averia"].fillna("").astype(str).replace("", np.nan)
labels_raw = labels_raw.fillna(desc["descripcion_reparacion"]).fillna(desc["comentarios"]).fillna("UNKNOWN")

# Reducir clases a las más frecuentes
topN = 20
top_labels = labels_raw.value_counts().head(topN).index.tolist()
labels_simpl = labels_raw.apply(lambda x: x if x in top_labels else "OTHER")

le = LabelEncoder()
y = le.fit_transform(labels_simpl)

tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=20000)
X = tfidf.fit_transform(desc["descripcion_ot"].astype(str).fillna("")).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='saga')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\n📊 Resultados del clasificador:")
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

# ------------------------------------------------------------
# 6. GUARDAR MODELOS
# ------------------------------------------------------------
out_dir = os.path.join(BASE_DIR, "modelos_caf")
os.makedirs(out_dir, exist_ok=True)

with open(os.path.join(out_dir, "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(tfidf, f)
with open(os.path.join(out_dir, "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)
with open(os.path.join(out_dir, "classifier_logreg.pkl"), "wb") as f:
    pickle.dump(clf, f)

print(f"\n💾 Modelos guardados en: {out_dir}")
print("✅ Script completado correctamente.")

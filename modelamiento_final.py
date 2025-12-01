import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from gensim.models import Word2Vec

# --- PARÁMETROS ---
W2V_MODEL_FILE = "word2vec_model_300d.bin"
FILE_FULL_DATASET = "dataset_api_final_filtrado.csv"
FILE_TRAIN_SPLIT = "train_1.csv"
FILE_TEST_SPLIT = "test_1.csv"
RANDOM_STATE = 42

# Columnas No-Textuales a usar como features
NUMERIC_COLS = ['mean_api', 'num_episodes_api']
# Omitimos CATEGORICAL_COLS = ['type'] porque no estaba en el dataset final.

# --------------------------------------------------------
# PARTE 1: Funciones Auxiliares
# --------------------------------------------------------

def vector_promedio_w2v(tokens, model, num_features):
    """Calcula el vector promedio de una lista de tokens usando el modelo Word2Vec."""
    feature_vec = np.zeros((num_features,), dtype="float32")
    n_words = 0
    index2word_set = set(model.wv.index_to_key) 
    
    for word in tokens:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model.wv[word])
            
    if n_words > 0:
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec

def tokenizar_sinopsis_simple(texto):
    """Tokenización simple basada en espacio, consistente con entrenar_w2v.py."""
    if isinstance(texto, str):
        return texto.lower().split()
    return []


# --------------------------------------------------------
# PARTE 2: Preparación y Unificación de Features NO-TEXTUALES
# --------------------------------------------------------

print("Cargando y preprocesando datos...")
df_completo = pd.read_csv(FILE_FULL_DATASET)
train_df = pd.read_csv(FILE_TRAIN_SPLIT)
test_df = pd.read_csv(FILE_TEST_SPLIT)

# 🛠️ CORRECCIÓN CLAVE: Usamos el índice como ID para unir los splits con el dataset completo
ID_COL = 'merge_id_temp' 

# 1. Crear la columna de ID en todos los DataFrames a partir de su índice.
train_df = train_df.reset_index(names=[ID_COL])
test_df = test_df.reset_index(names=[ID_COL])
df_completo = df_completo.reset_index(names=[ID_COL]) 

# 2. Preprocesamiento de df_completo (Imputación y Escalado)
for col in NUMERIC_COLS:
    # Corrección de la Future Warning
    df_completo[col] = df_completo[col].fillna(df_completo[col].median())

# Escalado de Numéricas
scaler = StandardScaler()
df_completo[NUMERIC_COLS] = scaler.fit_transform(df_completo[NUMERIC_COLS])

# 3. Definición de Features No-Textuales
OTHER_FEATURES_COLS = NUMERIC_COLS 
ALL_COLS_TO_MERGE = [ID_COL] + OTHER_FEATURES_COLS

# 4. Merge: Añadimos las columnas preprocesadas (escaladas) a los splits
train_df = pd.merge(train_df, df_completo[ALL_COLS_TO_MERGE], on=ID_COL, how='left')
test_df = pd.merge(test_df, df_completo[ALL_COLS_TO_MERGE], on=ID_COL, how='left')


# --------------------------------------------------------
# PARTE 3: Embeddings de Sinopsis y Concatenación Final
# --------------------------------------------------------

print("Generando vectores de sinopsis...")
# Cargar el modelo Word2Vec entrenado
w2v_model = Word2Vec.load(W2V_MODEL_FILE)
W2V_FEATURES = w2v_model.vector_size

# Aplicar tokenización
train_df['tokens'] = train_df['text'].apply(tokenizar_sinopsis_simple)
test_df['tokens'] = test_df['text'].apply(tokenizar_sinopsis_simple)

# 1. Creación de matrices de embeddings (X_w2v)
X_train_w2v = np.array([vector_promedio_w2v(tokens, w2v_model, W2V_FEATURES) for tokens in train_df['tokens']])
X_test_w2v = np.array([vector_promedio_w2v(tokens, w2v_model, W2V_FEATURES) for tokens in test_df['tokens']])

# 2. Matriz de features NUMÉRICAS (X_otros)
X_train_other = train_df[OTHER_FEATURES_COLS].values
X_test_other = test_df[OTHER_FEATURES_COLS].values

# 3. CONCATENACIÓN FINAL: Unimos Embeddings + Features No-Textuales
X_train_final = np.hstack([X_train_w2v, X_train_other])
X_test_final = np.hstack([X_test_w2v, X_test_other])

print(f"Dimensión de X_train_final: {X_train_final.shape}")


# --------------------------------------------------------
# PARTE 4: Preparación de la Variable Objetivo (Y)
# --------------------------------------------------------

# Las columnas de géneros son todas las columnas binarias que quedan en el split
# Excluimos las columnas temporales y de texto.
cols_to_exclude = OTHER_FEATURES_COLS + [ID_COL, 'text', 'tokens']
genre_cols = [c for c in train_df.columns if c not in cols_to_exclude]

# Extraer las matrices Y
Y_train = train_df[genre_cols].values
Y_test = test_df[genre_cols].values

# --------------------------------------------------------
# PARTE 5: Modelado y Evaluación Multietiqueta
# --------------------------------------------------------

print(f"Entrenando clasificador One-vs-Rest (Regresión Logística) en {X_train_final.shape[1]} features...")
base_model = LogisticRegression(solver='liblinear', 
                                class_weight='balanced',  
                                random_state=RANDOM_STATE, 
                                max_iter=1000)

ovr_classifier = OneVsRestClassifier(base_model)
ovr_classifier.fit(X_train_final, Y_train)

# Predicciones
Y_pred = ovr_classifier.predict(X_test_final)

# Métricas de Clasificación Multietiqueta
f1_micro = f1_score(Y_test, Y_pred, average='micro')
f1_macro = f1_score(Y_test, Y_pred, average='macro')
subset_accuracy = accuracy_score(Y_test, Y_pred)
ham_loss = hamming_loss(Y_test, Y_pred)

print("\n--- Resultados del Modelo Híbrido ---")
print(f"Número de géneros predichos: {len(genre_cols)}")
print(f"F1-Score (Micro, Ponderado por frecuencia): {f1_micro:.4f}")
print(f"F1-Score (Macro, Promedio por género): {f1_macro:.4f}")
print(f"Subset Accuracy (Exact Match): {subset_accuracy:.4f}")
print(f"Hamming Loss (Error promedio por etiqueta): {ham_loss:.4f}")
print("\n✔ Modelado híbrido completado.")
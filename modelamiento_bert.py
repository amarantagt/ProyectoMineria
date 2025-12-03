import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from tqdm import tqdm  # Barra de progreso

# --- PARÁMETROS ---
FILE_FULL_DATASET = "dataset_api_final_filtrado.csv"
FILE_TRAIN_SPLIT = "train_1.csv"
FILE_TEST_SPLIT = "test_1.csv"
RANDOM_STATE = 42
BATCH_SIZE = 32  # Procesamos las sinopsis en grupos de 32 para no saturar la memoria

# Configuración del dispositivo (Usa GPU 'mps' en Mac si está disponible, si no CPU)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Usando dispositivo de procesamiento: {device}")

# Columnas Numéricas
NUMERIC_COLS = ['mean_api', 'num_episodes_api']

# --------------------------------------------------------
# PARTE 1: Carga y Preparación de BERT
# --------------------------------------------------------

print("Cargando modelo DistilBERT...")
# Usamos 'distilbert-base-uncased' (versión ligera de BERT en inglés)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)

def obtener_bert_embeddings(textos, batch_size=32):
    """Genera embeddings usando BERT en lotes (batches) para no llenar la RAM."""
    all_embeddings = []
    
    # Procesar en lotes con barra de progreso
    for i in tqdm(range(0, len(textos), batch_size), desc="Procesando BERT"):
        batch_textos = textos[i : i + batch_size]
        
        # Tokenización
        inputs = tokenizer(batch_textos, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()} # Mover a GPU/MPS
        
        with torch.no_grad(): # No necesitamos gradientes (solo inferencia)
            outputs = model(**inputs)
        
        # Obtenemos el vector del token [CLS] (representación de la frase completa)
        # last_hidden_state tiene forma (batch, seq_len, 768) -> tomamos el primero de cada seq
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_embeddings)
        
    return np.vstack(all_embeddings)

# --------------------------------------------------------
# PARTE 2: Preparación de Datos (Igual que antes)
# --------------------------------------------------------

print("Cargando CSVs...")
df_completo = pd.read_csv(FILE_FULL_DATASET)
train_df = pd.read_csv(FILE_TRAIN_SPLIT)
test_df = pd.read_csv(FILE_TEST_SPLIT)

# --- Corrección de ID para el Merge (Igual que el script anterior) ---
ID_COL = 'merge_id_temp' 
train_df = train_df.reset_index(names=[ID_COL])
test_df = test_df.reset_index(names=[ID_COL])
df_completo = df_completo.reset_index(names=[ID_COL]) 

# Preprocesamiento de Numéricas
for col in NUMERIC_COLS:
    df_completo[col] = df_completo[col].fillna(df_completo[col].median())

scaler = StandardScaler()
df_completo[NUMERIC_COLS] = scaler.fit_transform(df_completo[NUMERIC_COLS])

# Merge
OTHER_FEATURES_COLS = NUMERIC_COLS 
ALL_COLS_TO_MERGE = [ID_COL] + OTHER_FEATURES_COLS
train_df = pd.merge(train_df, df_completo[ALL_COLS_TO_MERGE], on=ID_COL, how='left')
test_df = pd.merge(test_df, df_completo[ALL_COLS_TO_MERGE], on=ID_COL, how='left')

# --------------------------------------------------------
# PARTE 3: Generación de Embeddings con BERT (EL CAMBIO CLAVE)
# --------------------------------------------------------

print("\nGenerando embeddings con BERT para TRAIN...")
# Aseguramos que sea string y limpiamos nulos
train_texts = train_df['text'].astype(str).tolist()
X_train_bert = obtener_bert_embeddings(train_texts, BATCH_SIZE)

print("\nGenerando embeddings con BERT para TEST...")
test_texts = test_df['text'].astype(str).tolist()
X_test_bert = obtener_bert_embeddings(test_texts, BATCH_SIZE)

# Unimos con las otras variables (Numéricas)
X_train_other = train_df[OTHER_FEATURES_COLS].values
X_test_other = test_df[OTHER_FEATURES_COLS].values

X_train_final = np.hstack([X_train_bert, X_train_other])
X_test_final = np.hstack([X_test_bert, X_test_other])

print(f"\nDimensión final de X_train: {X_train_final.shape}")
# (Debería ser N filas x (768 de BERT + 2 numéricas) = 770 columnas)

# --------------------------------------------------------
# PARTE 4: Modelado y Evaluación
# --------------------------------------------------------

# Columnas de Géneros
cols_to_exclude = OTHER_FEATURES_COLS + [ID_COL, 'text', 'tokens']
genre_cols = [c for c in train_df.columns if c not in cols_to_exclude]

Y_train = train_df[genre_cols].values
Y_test = test_df[genre_cols].values

print(f"Entrenando Regresión Logística con features de BERT...")
# Aumentamos max_iter porque BERT genera vectores más complejos
base_model = LogisticRegression(solver='liblinear', 
                                class_weight='balanced',  
                                random_state=RANDOM_STATE, 
                                max_iter=2000)

ovr_classifier = OneVsRestClassifier(base_model)
ovr_classifier.fit(X_train_final, Y_train)

Y_pred = ovr_classifier.predict(X_test_final)

# Métricas
f1_micro = f1_score(Y_test, Y_pred, average='micro')
f1_macro = f1_score(Y_test, Y_pred, average='macro')
subset_acc = accuracy_score(Y_test, Y_pred)
ham_loss = hamming_loss(Y_test, Y_pred)

print("\n--- Resultados con BERT (DistilBERT) ---")
print(f"F1-Score (Micro): {f1_micro:.4f}")
print(f"F1-Score (Macro): {f1_macro:.4f}")
print(f"Subset Accuracy: {subset_acc:.4f}")
print(f"Hamming Loss: {ham_loss:.4f}")

def predecir_generos(sinopsis, mean_api_val=0, num_episodes_val=0):
    # --- 1. Preparar el texto ---
    inputs = tokenizer(
        [sinopsis],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # --- 2. Obtener embedding BERT ---
    with torch.no_grad():
        outputs = model(**inputs)
    texto_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # (1, 768)
    # --- 3. Preparar features numéricas ---
    # Recuerda: scaler ya está FITTEADO con el dataset original
    numeric_vector = scaler.transform([[mean_api_val, num_episodes_val]])  # (1,2)
    # --- 4. Concatenar para crear el feature vector final ---
    X_final = np.hstack([texto_emb, numeric_vector])  # (1, 770)
    # --- 5. Hacer predicción ---
    pred = ovr_classifier.predict(X_final)[0]  # array de 0/1
    # --- 6. Convertir el vector binario a nombres de géneros ---
    generos_predichos = [genre_cols[i] for i, val in enumerate(pred) if val == 1]
    return generos_predichos

sinopsis_ejemplo = "a year after escaping sword art online, kazuto kirigaya has been settling back into the real world. however, his peace is short-lived as a new incident occurs in a game called gun gale online, where a player by the name of death gun appears to be killing people in the real world by shooting them in-game. approached by officials to assist in investigating the murders, kazuto assumes his persona of kirito once again and logs into gun gale online, intent on stopping the killer. once inside, kirito meets sinon, a highly skilled sniper afflicted by a traumatic past. she is soon dragged in his chase after death gun, and together they enter the bullet of bullets, a tournament where their target is sure to appear. uncertain of death gun's real powers, kirito and sinon race to stop him before he has the chance to claim another life. not everything goes smoothly, however, as scars from the past impede their progress. in a high-stakes game where the next victim could easily be one of them, kirito puts his life on the line in the virtual world once more."
generos = predecir_generos(
    sinopsis_ejemplo,
    mean_api_val=0,            # si no quieres usar numéricas, deja 0
    num_episodes_val=0
)
print("Géneros predichos:", generos)
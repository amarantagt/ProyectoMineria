# entrenar_w2v.py
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
# Asegúrese de que nltk.download('punkt') esté hecho

# Cargar el dataset que contiene todas las sinopsis
df = pd.read_csv("dataset_api_final_filtrado.csv")

# 1. Tokenización
df['tokens'] = df['synopsis_api'].apply(lambda x: word_tokenize(str(x).lower()))

# 2. Entrenar modelo
sentences = df['tokens'].tolist()
w2v_model = Word2Vec(
    sentences=sentences,
    vector_size=300, # La dimensión que usaremos
    window=10,
    min_count=5,
    workers=4,
    sg=1 # Skip-gram
)

# 3. Guardar el modelo entrenado
w2v_model.save("word2vec_model_300d.bin") 
print("✔ Modelo Word2Vec entrenado y guardado como 'word2vec_model_300d.bin'")
import pandas as pd
import re

# ================================
# 1. Cargar dataset original limpio
# ================================
df = pd.read_csv("dataset_completo.csv")

# ================================
# 2. Eliminar filas con sinopsis nula o vacía
# ================================
df = df.dropna(subset=["synopsis_api"])
df = df[df["synopsis_api"].str.strip() != ""]  # elimina cadenas vacías

# ================================
# 3. Función para limpiar sinopsis
# ================================
def limpiar_sinopsis(texto):
    if not isinstance(texto, str):
        return texto

    # --- 3.1 Eliminar "[Written by MAL Rewrite]"
    texto = texto.replace("[Written by MAL Rewrite]", "")
    texto = texto.replace("(Written by MAL Rewrite)", "")
    texto = texto.replace("Written by MAL Rewrite", "")

    # --- 3.2 Eliminar bloques "(Source: ...)"
    texto = re.sub(r"\(Source:.*?\)", "", texto, flags=re.IGNORECASE)

    # --- 3.3 Eliminar otros bloques que contengan la palabra "source"
    texto = re.sub(r"\(.*?source.*?\)", "", texto, flags=re.IGNORECASE)
    texto = re.sub(r"\[.*?source.*?\]", "", texto, flags=re.IGNORECASE)

    # --- 3.4 Limpiar espacios extra
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto

def normalizar_texto(texto):
    if not isinstance(texto, str):
        return texto

    # 1) Pasar a minúsculas
    texto = texto.lower()

    # 2) Eliminar caracteres raros
    # Dejamos letras, números, puntuación básica y espacios
    texto = re.sub(r"[^a-z0-9áéíóúüñ.,;:!?()\"\'\s-]", " ", texto)

    # 3) Normalizar espacios múltiples
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto
# ================================
# 4. Aplicar limpieza a la columna de sinopsis
# ================================
df["synopsis_api"] = df["synopsis_api"].apply(limpiar_sinopsis)
df["synopsis_api"] = df["synopsis_api"].apply(normalizar_texto)
# ================================
# 5. Guardar resultado final
# ================================
df.to_csv("dataset_api_limpio.csv", index=False)

print("✔ Limpieza completa realizada. Archivo guardado como 'dataset_completo_limpio_sinopsis_limpia.csv'")

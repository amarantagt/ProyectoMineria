import pandas as pd

# Lista con rutas de tus 5 CSV:
archivos = [
    "animes_parte1_resultado.csv",
    "animes_parte2_resultado.csv",
    "animes_parte3_resultado.csv",
    "animes_parte4_resultado.csv",
    "animes_parte5_resultado.csv"
]

# Leer todos los CSV en una lista de DataFrames
dfs = [pd.read_csv(archivo) for archivo in archivos]

# Juntarlos (uno debajo del otro)
df_final = pd.concat(dfs, ignore_index=True)

# Guardar el CSV final
df_final.to_csv("dataset_completo.csv", index=False)

print("Listo: dataset_completo.csv fue creado exitosamente.")

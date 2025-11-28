import pandas as pd

df = pd.read_csv("dataset_final_con_generos_limpios.csv")

# 1) Eliminar filas sin géneros
df = df[df["genres_list"].apply(lambda x: len(eval(x)) > 0)]

# 2) Eliminar géneros que quedaron con todas las filas en 0
genre_cols = [c for c in df.columns if c not in ["animeID","title","mean_api","num_episodes_api","synopsis_api","genres_api","genres_list"]]

empty_genres = [c for c in genre_cols if df[c].sum() == 0]
df = df.drop(columns=empty_genres)



df.to_csv("dataset_api_final.csv", index=False)

#Veamos si hay desbalance
original_cols = ["animeID","title","mean_api","num_episodes_api","synopsis_api","genres_api","genres_list"]

genre_cols = [c for c in df.columns if c not in original_cols]

print("Columnas de géneros detectadas:")
print(genre_cols)

desbalance = df[genre_cols].sum().sort_values()
print("\nDesbalance por género:")
print(desbalance)

#Generamos dataset filtrado
# contar frecuencia
counts = df[genre_cols].sum()
#Frecuencia minima para filtrar
threshold = 20  
rare_genres = counts[counts < threshold].index.tolist()

print("Géneros eliminados por baja frecuencia:", rare_genres)

# eliminar géneros raros
df_filtrado = df.drop(columns=rare_genres)

df_filtrado.to_csv("dataset_api_final_filtrado.csv", index=False)


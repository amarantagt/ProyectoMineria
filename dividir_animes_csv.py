import pandas as pd

# Leer el dataset original
df = pd.read_csv("animes.csv")

# Calcular tamaño de cada parte
n = len(df)
partes = 5
tamano = n // partes

for i in range(partes):
    inicio = i * tamano
    fin = (i + 1) * tamano if i < partes - 1 else n
    df_parte = df.iloc[inicio:fin]
    nombre_archivo = f"animes_parte{i+1}.csv"
    df_parte.to_csv(nombre_archivo, index=False)
    print(f"✅ Guardada parte {i+1} con {len(df_parte)} registros → {nombre_archivo}")

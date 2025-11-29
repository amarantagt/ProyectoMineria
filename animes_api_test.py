import requests
import pandas as pd
import time
from datetime import datetime

CLIENT_ID = "2d1afdbebb4f63626f1e25675e698157"

# === CONFIGURACIÓN ===
limite_prueba = 100      # <-- Cambia este número si quieres probar con más
sleep_request = 1.0       # segundos entre requests

# Leer CSV original
df = pd.read_csv("animes.csv").head(limite_prueba)

resultados = []
inicio = datetime.now()

for i, row in df.iterrows():
    nombre = row["title"]
    print(f"🔍 ({i+1}/{len(df)}) Buscando: {nombre}")
    
    url = f"https://api.myanimelist.net/v2/anime?q={nombre}&limit=1&fields=id,title,mean,num_episodes,genres,synopsis"
    headers = {"X-MAL-CLIENT-ID": CLIENT_ID}
    
    node = None
    try:
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            data = r.json()
            if "data" in data and len(data["data"]) > 0:
                node = data["data"][0]["node"]
        elif r.status_code == 429:
            print("⚠️ Límite alcanzado. Esperando 60 segundos...")
            time.sleep(60)
            continue
        else:
            print(f"❌ Error {r.status_code} para {nombre}")
    except Exception as e:
        print(f"💥 Error en {nombre}: {e}")

    if node:
        resultados.append({
            **row.to_dict(),
            "mal_id": node.get("id"),
            "mean_api": node.get("mean"),
            "synopsis_api": node.get("synopsis"),
            "genres_api": ", ".join([g["name"] for g in node.get("genres", [])]),
            "num_episodes_api": node.get("num_episodes")
        })
    else:
        resultados.append({**row.to_dict(), "mal_id": None})
        print(f"⚠️ No encontrado: {nombre}")

    time.sleep(sleep_request)

# Guardar CSV de prueba
df_final = pd.DataFrame(resultados)
df_final.to_csv("animes_actualizado_test.csv", index=False, encoding="utf-8-sig")

fin = datetime.now()
print(f"\n✅ Test completado en {fin - inicio}. Archivo: animes_actualizado_test.csv")

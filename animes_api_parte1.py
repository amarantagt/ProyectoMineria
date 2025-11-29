import requests
import pandas as pd
import time
from tqdm import tqdm
from requests.adapters import HTTPAdapter, Retry

CLIENT_ID = "e732f4be807d248e6a95a2b6d5fb01f3"

input_csv = "animes_parte1.csv"
output_csv = "animes_parte1_resultado.csv"

df = pd.read_csv(input_csv)
headers = {"X-MAL-CLIENT-ID": CLIENT_ID}
resultados = []

# --- Sesión con reintentos automáticos ---
session = requests.Session()
retries = Retry(
    total=5,                # número máximo de reintentos
    backoff_factor=1.5,     # espera exponencial entre reintentos
    status_forcelist=[500, 502, 503, 504],
    raise_on_status=False
)
session.mount("https://", HTTPAdapter(max_retries=retries))

# --- Loop con barra de progreso ---
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Consultando MAL"):
    nombre = str(row["title"])

    url = f"https://api.myanimelist.net/v2/anime?q={nombre}&limit=1&fields=id,title,main_picture,num_episodes,mean,genres,synopsis"

    try:
        respuesta = session.get(url, headers=headers, timeout=10)
    except Exception as e:
        tqdm.write(f"⚠️ Error de conexión en {nombre}: {e}. Saltando.")
        continue

    if respuesta.status_code == 200:
        datos = respuesta.json()
        if "data" in datos and len(datos["data"]) > 0:
            node = datos["data"][0]["node"]
            resultados.append({
                "animeID": row["animeID"],
                "title": row["title"],
                "mean_api": node.get("mean"),
                "num_episodes_api": node.get("num_episodes"),
                "genres_api": ", ".join([g["name"] for g in node.get("genres", [])]),
                "synopsis_api": node.get("synopsis")
            })
    else:
        tqdm.write(f"❌ Error {respuesta.status_code} en {nombre}")

    # Guardar parcial cada 50 filas
    if (idx + 1) % 50 == 0:
        pd.DataFrame(resultados).to_csv(output_csv, index=False)
        tqdm.write("💾 Guardado parcial...")

    time.sleep(2.5)

# --- Guardado final ---
pd.DataFrame(resultados).to_csv(output_csv, index=False)
tqdm.write(f"✅ Resultados guardados en {output_csv}")

import requests
import pandas

CLIENT_ID = "2d1afdbebb4f63626f1e25675e698157"

# Buscar info de Naruto y pedir más campos
url = "https://api.myanimelist.net/v2/anime?q=naruto&limit=5&fields=id,title,main_picture,num_episodes,mean,genres,synopsis"

headers = {
    "X-MAL-CLIENT-ID": CLIENT_ID
}

respuesta = requests.get(url, headers=headers)
datos = respuesta.json()

for anime in datos.get("data", []):
    node = anime["node"]
    print("Título:", node.get("title"))
    print("ID:", node.get("id"))
    print("Episodios:", node.get("num_episodes"))
    print("Sinopsis:", node.get("synopsis"))
    print("Puntuación promedio:", node.get("mean"))
    print("-----")
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import ast
df = pd.read_csv("dataset_api_limpio.csv")


def parse_genres(g):
    if pd.isna(g):
        return []

    # Sacar comillas externas
    g = g.strip().strip('"').strip("'")

    # Intentar convertir si viene como lista en forma de string
    try:
        parsed = ast.literal_eval(g)
        if isinstance(parsed, list):
            return [x.strip() for x in parsed]
    except:
        pass
    
    # Si llega como: Adventure, Action, Fantasy
    return [x.strip() for x in g.split(",")]

df["genres_list"] = df["genres_api"].apply(parse_genres)

mlb = MultiLabelBinarizer()

y = mlb.fit_transform(df["genres_list"])
genre_labels = mlb.classes_
df_genres = pd.DataFrame(y, columns=genre_labels)
df = pd.concat([df, df_genres], axis=1)
df.to_csv("dataset_final_con_generos_limpios.csv", index=False)
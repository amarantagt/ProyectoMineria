from sklearn.model_selection import train_test_split
import pandas as pd
#Dataset filtrado los generos poco frecuentes
df2 = pd.read_csv("dataset_api_final_filtrado.csv")

genre_cols2 = [c for c in df2.columns if c not in ["animeID","title","mean_api","num_episodes_api","synopsis_api","genres_api","genres_list"]]
X = df2["synopsis_api"]           
Y = df2[genre_cols2]     
# split 80/20 con semilla fija
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=0,
    shuffle=True
)
# Guardar para BERT
train_df2 = pd.DataFrame({"text": X_train, **Y_train.to_dict(orient="list")})
test_df2  = pd.DataFrame({"text": X_test,  **Y_test.to_dict(orient="list")})

train_df2.to_csv("train_1.csv", index=False)
test_df2.to_csv("test_1.csv", index=False)
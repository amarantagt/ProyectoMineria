"""
clustering_analysis.py

Script para preprocesar `animes.csv` y `dataset_completo.csv`, agregar características desde `ratings.csv` (procesado por chunks),
aplicar KMeans, Agglomerative y DBSCAN, y guardar métricas y etiquetas para comparar pureza respecto a género y score.

Uso:
    python clustering_analysis.py

Salida esperada (carpeta `outputs/`):
 - merged_data.csv (dataframe usado para clustering)
 - cluster_results.csv (métricas por configuración)
 - labels_<method>_<params>.csv (etiquetas por anime)

Notas:
 - `ratings.csv` puede ser grande; el script lo procesa en chunks para extraer agregados por anime.
 - adapta columnas en la sección `COLUMNS_TO_USE` si tu dataset tiene nombres diferentes.
"""

import os
import argparse
import logging
from ast import literal_eval

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Utilidades ---

def extract_primary_genre(genres_field):
    """Extrae el primer género de una representación tipo lista o string separada por coma."""
    if pd.isna(genres_field):
        return np.nan
    if isinstance(genres_field, list):
        return genres_field[0] if len(genres_field) > 0 else np.nan
    s = str(genres_field).strip()
    # Try parse python list
    try:
        parsed = literal_eval(s)
        if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
            return parsed[0]
    except Exception:
        pass
    # fallback: comma-separated
    if ',' in s:
        return s.split(',')[0].strip().strip("'\"")
    return s.strip().strip("'\"")


def cluster_purity(labels_true, labels_pred):
    """Calcula la pureza de clusters cuando labels_true es single-label por fila."""
    dfc = pd.DataFrame({'true': labels_true, 'pred': labels_pred})
    dfc['pred'] = dfc['pred'].fillna(-1)
    total = len(dfc)
    if total == 0:
        return np.nan
    pur_sum = 0
    for c in dfc['pred'].unique():
        sub = dfc[dfc['pred'] == c]
        if len(sub) == 0:
            continue
        top = sub['true'].value_counts().iloc[0]
        pur_sum += top
    return pur_sum / total


# --- Lectura y merge de datasets ---

def load_and_merge(animes_path='animes.csv', dataset_path='dataset_completo.csv', ratings_path='ratings.csv'):
    logging.info('Cargando animes desde %s', animes_path)
    animes = pd.read_csv(animes_path)
    logging.info('animes shape: %s', animes.shape)

    logging.info('Cargando dataset_completo desde %s', dataset_path)
    ds = pd.read_csv(dataset_path)
    logging.info('dataset_completo shape: %s', ds.shape)

    # Merge por animeID (int)
    if 'animeID' in animes.columns and 'animeID' in ds.columns:
        merged = pd.merge(animes, ds, on='animeID', how='outer', suffixes=('_anime','_ds'))
    else:
        # intentar por title
        merged = pd.merge(animes, ds, on='title', how='outer', suffixes=('_anime','_ds'))
    logging.info('Merged shape: %s', merged.shape)

    # Process ratings.csv in chunks to compute per-anime aggregates
    if os.path.exists(ratings_path):
        logging.info('Procesando ratings desde %s (chunksize)...', ratings_path)
        # intentamos inferir columnas leyendo solo el header con pandas
        try:
            header = pd.read_csv(ratings_path, nrows=0)
            cols = list(header.columns)
            logging.info('Ratings columns detected: %s', cols)
        except Exception as e:
            logging.warning('No se pudo leer header de ratings.csv: %s', e)
            cols = None

        # asumimos que existe una columna con id/animeId y una columna rating
        # buscaremos nombres posibles
        anime_col_candidates = ['anime_id','animeID','animeId','animeID','animeid','anime']
        rating_col_candidates = ['rating','score','rating_score']

        reader = pd.read_csv(ratings_path, chunksize=200000)
        agg_list = []
        for chunk in reader:
            # determinar nombres si es la primera iteración
            if cols is None:
                cols = list(chunk.columns)
            # detectar anime id col
            anime_col = None
            for c in anime_col_candidates:
                if c in chunk.columns:
                    anime_col = c
                    break
            if anime_col is None:
                # Heurística: la segunda columna suele ser anime id en algunos datasets
                anime_col = chunk.columns[1]

            rating_col = None
            for c in rating_col_candidates:
                if c in chunk.columns:
                    rating_col = c
                    break
            if rating_col is None:
                # heurística: última columna podría ser rating
                rating_col = chunk.columns[-1]

            # keep only needed columns
            local = chunk[[anime_col, rating_col]].copy()
            local.columns = ['anime_id', 'rating']
            # intentar convertir a numérico
            local['rating'] = pd.to_numeric(local['rating'], errors='coerce')
            g = local.groupby('anime_id')['rating'].agg(['count','mean','std','median']).rename(columns={'count':'rating_count','mean':'rating_mean','std':'rating_std','median':'rating_median'})
            agg_list.append(g)
        if len(agg_list) > 0:
            agg_all = pd.concat(agg_list).groupby(level=0).agg({'rating_count':'sum','rating_mean':'mean','rating_std':'mean','rating_median':'mean'})
            agg_all.index.name = 'anime_id'
            agg_all = agg_all.reset_index()
            logging.info('Aggregados de ratings por anime: %s rows', agg_all.shape[0])
            # Merge con merged (buscar columna animeID o anime_id)
            if 'animeID' in merged.columns:
                merged = pd.merge(merged, agg_all, left_on='animeID', right_on='anime_id', how='left')
            else:
                merged = pd.merge(merged, agg_all, left_on='animeID', right_on='anime_id', how='left')
        else:
            logging.warning('No se obtuvieron agregados de ratings (archivo vacío o formato inesperado).')
    else:
        logging.warning('ratings.csv no encontrado en ruta: %s. Se omiten características de usuarios.', ratings_path)

    return merged


# --- Preprocesamiento y features ---

def preprocess_and_features(df):
    df = df.copy()

    # Extract primary genre from available genre columns
    # Prefer `genres` column (animes.csv) then `genres_api` (dataset_completo)
    if 'genres' in df.columns:
        df['primary_genre'] = df['genres'].apply(lambda x: extract_primary_genre(x))
    elif 'genres_api' in df.columns:
        df['primary_genre'] = df['genres_api'].apply(lambda x: extract_primary_genre(x))
    else:
        df['primary_genre'] = np.nan

    # score: prefer 'score' from animes.csv, fallback to mean_api
    if 'score' in df.columns:
        df['score_num'] = pd.to_numeric(df['score'], errors='coerce')
    else:
        df['score_num'] = np.nan
    if 'mean_api' in df.columns and df['score_num'].isna().all():
        df['score_num'] = pd.to_numeric(df['mean_api'], errors='coerce')

    # Create score_category
    df['score_num'] = df['score_num'].fillna(df['score_num'].median())
    bins = [0, 6.5, 8.0, 10]
    labels = ['low','mid','high']
    df['score_category'] = pd.cut(df['score_num'], bins=bins, labels=labels, include_lowest=True)

    # Episodes: try 'episodes' then 'num_episodes_api'
    if 'episodes' in df.columns:
        df['episodes_num'] = pd.to_numeric(df['episodes'], errors='coerce')
    elif 'num_episodes_api' in df.columns:
        df['episodes_num'] = pd.to_numeric(df['num_episodes_api'], errors='coerce')
    else:
        df['episodes_num'] = np.nan
    df['episodes_num'] = df['episodes_num'].fillna(df['episodes_num'].median())

    # Year -> numeric if available
    if 'year' in df.columns:
        df['year_num'] = pd.to_numeric(df['year'], errors='coerce')
    else:
        df['year_num'] = np.nan
    df['year_num'] = df['year_num'].fillna(df['year_num'].median())

    # Ratings aggregates: already merged as rating_count, rating_mean, rating_std
    for c in ['rating_count','rating_mean','rating_std','rating_median']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            df[c] = np.nan
    df['rating_count'] = df['rating_count'].fillna(0)
    df['rating_mean'] = df['rating_mean'].fillna(df['rating_mean'].median())
    df['rating_std'] = df['rating_std'].fillna(0)

    # Type: one-hot small set
    if 'type' in df.columns:
        df['type'] = df['type'].fillna('Unknown')
        type_ohe = pd.get_dummies(df['type'], prefix='type')
        df = pd.concat([df, type_ohe], axis=1)

    # Genres multi-label binarizer (genres from animes.csv if exists)
    if 'genres' in df.columns:
        # genres column often is string representation of list
        def parse_list_cell(x):
            if pd.isna(x):
                return []
            if isinstance(x, list):
                return x
            s = str(x)
            try:
                parsed = literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return parsed
            except Exception:
                pass
            # fallback: comma split
            return [i.strip().strip("'\"") for i in s.split(',') if i.strip()]

        genres_parsed = df['genres'].apply(parse_list_cell)
        mlb = MultiLabelBinarizer(sparse_output=False)
        try:
            genres_m = pd.DataFrame(mlb.fit_transform(genres_parsed), columns=[f'genre_{g}' for g in mlb.classes_], index=df.index)
            df = pd.concat([df, genres_m], axis=1)
        except Exception:
            logging.info('No se pudieron aplicar MultiLabelBinarizer a `genres`.')

    # Features selection for clustering
    feature_cols = ['score_num','episodes_num','year_num','rating_count','rating_mean','rating_std']
    # include type ohe columns
    feature_cols += [c for c in df.columns if c.startswith('type_')]
    # include top genre columns if present (limit to avoid too high dimensionality)
    genre_cols = [c for c in df.columns if c.startswith('genre_')]
    if len(genre_cols) > 0:
        # keep only the most frequent 12 genres
        if len(genre_cols) > 12:
            # compute freq
            freq = df[genre_cols].sum().sort_values(ascending=False)
            keep = list(freq.head(12).index)
        else:
            keep = genre_cols
        feature_cols += keep

    feature_cols = [c for c in feature_cols if c in df.columns]
    logging.info('Feature columns used for clustering: %s', feature_cols)

    # Fill missing numeric with median
    df_features = df[feature_cols].copy()
    for c in df_features.columns:
        if pd.api.types.is_numeric_dtype(df_features[c]):
            df_features[c] = df_features[c].fillna(df_features[c].median())

    return df, df_features


# --- Clustering y evaluación ---

def run_clustering_and_evaluate(df, X, out_dir=OUTPUT_DIR):
    results = []
    X_np = X.values

    # Escalado
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_np)

    # PCA para reducción a 5 componentes (si hay más)
    n_comp = min(10, Xs.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    Xp = pca.fit_transform(Xs)
    logging.info('Explained variance ratio (first 5): %s', pca.explained_variance_ratio_[:5].tolist())

    # 1) KMeans barrido k
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(Xp[:, :5])
        sil = silhouette_score(Xp[:, :5], labels)
        db = davies_bouldin_score(Xp[:, :5], labels)
        pur_gen = cluster_purity(df['primary_genre'].fillna('NA'), labels)
        pur_score = cluster_purity(df['score_category'].astype(str).replace('nan','NA'), labels)
        results.append({'method':'kmeans','params':f'k={k}','silhouette':sil,'db':db,'pur_genre':pur_gen,'pur_score':pur_score})
        # guardar etiquetas
        out_labels = pd.DataFrame({'animeID': df.get('animeID', df.index), 'label': labels})
        out_labels.to_csv(os.path.join(out_dir, f'labels_kmeans_k{k}.csv'), index=False)
        logging.info('KMeans k=%d done: silhouette=%.4f purity_genre=%.4f', k, sil, pur_gen)

    # 2) Agglomerative (ward, average, complete) for k=2..8
    for linkage in ['ward','average','complete']:
        for k in range(2,9):
            # ward requires euclidean metric and no connectivity; skip if incompatible
            if linkage == 'ward' and Xp.shape[1] < 1:
                continue
            ac = AgglomerativeClustering(n_clusters=k, linkage=linkage)
            labels = ac.fit_predict(Xp[:, :5])
            sil = silhouette_score(Xp[:, :5], labels)
            db = davies_bouldin_score(Xp[:, :5], labels)
            pur_gen = cluster_purity(df['primary_genre'].fillna('NA'), labels)
            pur_score = cluster_purity(df['score_category'].astype(str).replace('nan','NA'), labels)
            results.append({'method':'agglomerative','params':f'link={linkage},k={k}','silhouette':sil,'db':db,'pur_genre':pur_gen,'pur_score':pur_score})
            out_labels = pd.DataFrame({'animeID': df.get('animeID', df.index), 'label': labels})
            out_labels.to_csv(os.path.join(out_dir, f'labels_aggl_{linkage}_k{k}.csv'), index=False)

    # 3) DBSCAN grid search (eps scaled after PCA spread)
    eps_list = [0.3, 0.5, 0.7, 1.0]
    min_samples_list = [3,5,7]
    for eps in eps_list:
        for ms in min_samples_list:
            dbs = DBSCAN(eps=eps, min_samples=ms)
            labels = dbs.fit_predict(Xp[:, :5])
            # if all noise or single cluster skip silhouette
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters <= 1:
                logging.info('DBSCAN eps=%.2f min_samples=%d produced %d clusters — skipping metrics', eps, ms, n_clusters)
                continue
            sil = silhouette_score(Xp[:, :5], labels)
            db = davies_bouldin_score(Xp[:, :5], labels)
            pur_gen = cluster_purity(df['primary_genre'].fillna('NA'), labels)
            pur_score = cluster_purity(df['score_category'].astype(str).replace('nan','NA'), labels)
            results.append({'method':'dbscan','params':f'eps={eps},min_samples={ms}','silhouette':sil,'db':db,'pur_genre':pur_gen,'pur_score':pur_score})
            out_labels = pd.DataFrame({'animeID': df.get('animeID', df.index), 'label': labels})
            out_labels.to_csv(os.path.join(out_dir, f'labels_dbscan_eps{eps}_ms{ms}.csv'), index=False)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(out_dir, 'cluster_results.csv'), index=False)
    logging.info('Saved cluster_results.csv with %d rows', len(results_df))

    return results_df


# --- Main ---

def main(args):
    merged = load_and_merge(animes_path=args.animes, dataset_path=args.dataset, ratings_path=args.ratings)
    merged.to_csv(os.path.join(OUTPUT_DIR, 'merged_raw.csv'), index=False)
    logging.info('Saved merged_raw.csv')

    df, X = preprocess_and_features(merged)
    df.to_csv(os.path.join(OUTPUT_DIR, 'merged_data.csv'), index=False)
    logging.info('Saved merged_data.csv; features shape: %s', X.shape)

    results_df = run_clustering_and_evaluate(df, X)
    logging.info('Clustering complete. Results saved in %s', OUTPUT_DIR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clustering analysis for anime datasets')
    parser.add_argument('--animes', default='animes.csv', help='Ruta a animes.csv')
    parser.add_argument('--dataset', default='dataset_completo.csv', help='Ruta a dataset_completo.csv')
    parser.add_argument('--ratings', default='ratings.csv', help='Ruta a ratings.csv (puede ser grande)')
    args = parser.parse_args()
    main(args)

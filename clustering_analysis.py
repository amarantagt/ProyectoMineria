"""clustering_analysis.py

Versión comentada: se añaden explicaciones en español sobre lo que hace cada sección
y las líneas clave. El comportamiento del programa no cambia.

Resumen rápido:
- Carga `animes.csv` y `dataset_completo.csv` y los une.
- Agrega agregados por anime a partir de `ratings.csv` (procesado por chunks para archivos grandes).
- Extrae y genera features (score, episodios, año, géneros, etc.).
- Aplica KMeans, Agglomerative y DBSCAN; calcula métricas internas y pureza externa.
- Guarda resultados y etiquetas en la carpeta `outputs/`.
"""

import os
import argparse
import logging
from ast import literal_eval  # para evaluar cadenas que contienen listas literales

import numpy as np
import pandas as pd
import time
# Importamos utilidades de scikit-learn para preprocesamiento, reducción y clustering
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Configuración básica del logger para imprimir progreso
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Directorio donde guardamos salidas (crea si no existe)
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------ UTILIDADES ------------------

def extract_primary_genre(genres_field):
    """Extrae el primer género de una columna que puede ser:
    - NaN
    - una lista de Python (ej. ['Action','Adventure']) serializada como string
    - una cadena separada por comas ('Action, Adventure')

    Devuelve None (np.nan) si no hay género.
    """
    # Si es NaN devolvemos NaN
    if pd.isna(genres_field):
        return np.nan
    # Si ya es una lista Python en memoria, tomar el primer elemento
    if isinstance(genres_field, list):
        return genres_field[0] if len(genres_field) > 0 else np.nan
    # Convertir a string y limpiar espacios
    s = str(genres_field).strip()
    # Intentar interpretar cadenas que contienen una lista literal (p. ej. "['A','B']")
    try:
        parsed = literal_eval(s)
        if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
            return parsed[0]
    except Exception:
        # Si falla, seguimos con el fallback
        pass
    # Si tiene comas, asumimos formato 'A, B, C' y tomamos el primero
    if ',' in s:
        return s.split(',')[0].strip().strip("'\"")
    # En cualquier otro caso devolvemos el string limpio
    return s.strip().strip("'\"")


def cluster_purity(labels_true, labels_pred):
    """Calcula la pureza de clusters (proporción de elementos correctamente asignados
    si usamos la etiqueta más frecuente de cada cluster).

    labels_true: array-like de etiquetas verdaderas (single-label por fila)
    labels_pred: etiquetas de cluster producidas por el algoritmo
    """
    # Creamos un DataFrame temporal para agrupar
    dfc = pd.DataFrame({'true': labels_true, 'pred': labels_pred})
    # Rellenar predictores nulos con -1 (ruido) para contar adecuadamente
    dfc['pred'] = dfc['pred'].fillna(-1)
    total = len(dfc)
    if total == 0:
        return np.nan
    pur_sum = 0
    # Para cada cluster, sumar el número de elementos de la clase mayoritaria
    for c in dfc['pred'].unique():
        sub = dfc[dfc['pred'] == c]
        if len(sub) == 0:
            continue
        top = sub['true'].value_counts().iloc[0]
        pur_sum += top
    # Dividir por el total para obtener la pureza global
    return pur_sum / total


# ------------------ LECTURA Y MERGE DE DATASETS ------------------

def load_and_merge(animes_path='animes.csv', dataset_path='dataset_completo.csv', ratings_path='ratings.csv'):
    """Carga `animes.csv` y `dataset_completo.csv`, los une y agrega estadísticas de `ratings.csv`.

    ratings.csv se procesa por chunks para no cargar todo en memoria cuando es grande.
    """
    # Cargar archivo de animes
    logging.info('Cargando animes desde %s', animes_path)
    animes = pd.read_csv(animes_path)
    logging.info('animes shape: %s', animes.shape)

    # Cargar dataset adicional con sinopsis/genres API
    logging.info('Cargando dataset_completo desde %s', dataset_path)
    ds = pd.read_csv(dataset_path)
    logging.info('dataset_completo shape: %s', ds.shape)

    # Intentar hacer merge por `animeID`; si no existe, intentamos por `title`
    if 'animeID' in animes.columns and 'animeID' in ds.columns:
        merged = pd.merge(animes, ds, on='animeID', how='outer', suffixes=('_anime','_ds'))
    else:
        merged = pd.merge(animes, ds, on='title', how='outer', suffixes=('_anime','_ds'))
    logging.info('Merged shape: %s', merged.shape)

    # Si existe ratings.csv lo procesamos en chunks
    if os.path.exists(ratings_path):
        logging.info('Procesando ratings desde %s (chunksize)...', ratings_path)
        # Intentar leer sólo el header para conocer columnas
        try:
            header = pd.read_csv(ratings_path, nrows=0)
            cols = list(header.columns)
            logging.info('Ratings columns detected: %s', cols)
        except Exception as e:
            logging.warning('No se pudo leer header de ratings.csv: %s', e)
            cols = None

        # En nuestros datasets la columna identificadora es `animeID`, la usamos directamente.
        # Para la columna de rating en `ratings.csv` esperamos el rating individual que da
        # un usuario ('rating'). Como fallback usamos 'score' si el CSV tuviera otro nombre.
        rating_col_candidates = ['rating', 'score']

        # Leer por chunks para ahorrar memoria
        reader = pd.read_csv(ratings_path, chunksize=200000)
        agg_list = []
        # Por cada chunk calculamos count, mean, std, median por anime
        for chunk in reader:
            if cols is None:
                cols = list(chunk.columns)
            # Preferimos la columna explícita 'animeID' en los chunks.
            if 'animeID' in chunk.columns:
                anime_col = 'animeID'
            else:
                # Fallback heurístico: tomar la segunda columna si no existe 'animeID'
                anime_col = chunk.columns[1]

            # Detectar columna con rating
            rating_col = None
            for c in rating_col_candidates:
                if c in chunk.columns:
                    rating_col = c
                    break
            if rating_col is None:
                # heuristic fallback: última columna
                rating_col = chunk.columns[-1]

            # Mantener solo las columnas relevantes y normalizar nombres
            local = chunk[[anime_col, rating_col]].copy()
            local.columns = ['anime_id', 'rating']
            # Forzar a numérico y convertir valores inválidos a NaN
            local['rating'] = pd.to_numeric(local['rating'], errors='coerce')
            # Agregados por anime
            g = local.groupby('anime_id')['rating'].agg(['count','mean','std','median']).rename(columns={'count':'rating_count','mean':'rating_mean','std':'rating_std','median':'rating_median'})
            agg_list.append(g)

        # Concatenar agregados y unir por anime_id
        if len(agg_list) > 0:
            agg_all = pd.concat(agg_list).groupby(level=0).agg({'rating_count':'sum','rating_mean':'mean','rating_std':'mean','rating_median':'mean'})
            agg_all.index.name = 'anime_id'
            agg_all = agg_all.reset_index()
            logging.info('Aggregados de ratings por anime: %s rows', agg_all.shape[0])
            # Merge con la tabla principal
            if 'animeID' in merged.columns:
                merged = pd.merge(merged, agg_all, left_on='animeID', right_on='anime_id', how='left')
            else:
                merged = pd.merge(merged, agg_all, left_on='animeID', right_on='anime_id', how='left')
        else:
            logging.warning('No se obtuvieron agregados de ratings (archivo vacío o formato inesperado).')
    else:
        logging.warning('ratings.csv no encontrado en ruta: %s. Se omiten características de usuarios.', ratings_path)

    return merged


# ------------------ PREPROCESAMIENTO Y FEATURES ------------------

def preprocess_and_features(df, tfidf_max_features=0):
    """Genera columnas y features a partir del DataFrame unido.

    - Extrae `primary_genre`.
    - Normaliza `score` y crea `score_category`.
    - Limpia `episodes` y `year`.
    - Codifica `type` y los `genres` multi-etiqueta.
    - Devuelve (df_completo, df_features) donde df_features está listo para clustering.
    """
    # Trabajar sobre una copia para no mutar el original externo
    df = df.copy()

    # PRIMARY GENRE: preferimos la columna `genres` de animes.csv; si no existe usamos `genres_api`
    if 'genres' in df.columns:
        df['primary_genre'] = df['genres'].apply(lambda x: extract_primary_genre(x))
    elif 'genres_api' in df.columns:
        df['primary_genre'] = df['genres_api'].apply(lambda x: extract_primary_genre(x))
    else:
        df['primary_genre'] = np.nan

    # SCORE: preferimos `score` de animes.csv; si no está, usamos `mean_api` del otro dataset
    if 'score' in df.columns:
        df['score_num'] = pd.to_numeric(df['score'], errors='coerce')
    else:
        df['score_num'] = np.nan
    if 'mean_api' in df.columns and df['score_num'].isna().all():
        df['score_num'] = pd.to_numeric(df['mean_api'], errors='coerce')

    # CATEGORÍAS DE SCORE: low/mid/high (ejemplo de discretización simple)
    df['score_num'] = df['score_num'].fillna(df['score_num'].median())
    bins = [0, 6.5, 8.0, 10]
    labels = ['low','mid','high']
    df['score_category'] = pd.cut(df['score_num'], bins=bins, labels=labels, include_lowest=True)

    # EPISODIOS -> numeric
    if 'episodes' in df.columns:
        df['episodes_num'] = pd.to_numeric(df['episodes'], errors='coerce')
    elif 'num_episodes_api' in df.columns:
        df['episodes_num'] = pd.to_numeric(df['num_episodes_api'], errors='coerce')
    else:
        df['episodes_num'] = np.nan
    df['episodes_num'] = df['episodes_num'].fillna(df['episodes_num'].median())

    # YEAR -> numeric
    if 'year' in df.columns:
        df['year_num'] = pd.to_numeric(df['year'], errors='coerce')
    else:
        df['year_num'] = np.nan
    df['year_num'] = df['year_num'].fillna(df['year_num'].median())

    # RATINGS agregados ya añadidos: asegurar tipo numérico y rellenar nulos
    for c in ['rating_count','rating_mean','rating_std','rating_median']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            df[c] = np.nan
    df['rating_count'] = df['rating_count'].fillna(0)
    df['rating_mean'] = df['rating_mean'].fillna(df['rating_mean'].median())
    df['rating_std'] = df['rating_std'].fillna(0)

    # TYPE -> one-hot encoding para unas pocas categorías
    if 'type' in df.columns:
        df['type'] = df['type'].fillna('Unknown')
        type_ohe = pd.get_dummies(df['type'], prefix='type')
        df = pd.concat([df, type_ohe], axis=1)

    # GENRES multi-etiqueta -> usar MultiLabelBinarizer para crear columnas binarias por género
    if 'genres' in df.columns:
        def parse_list_cell(x):
            # Devuelve lista vacía si NaN
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
            # fallback: split por comas
            return [i.strip().strip("'\"") for i in s.split(',') if i.strip()]

        genres_parsed = df['genres'].apply(parse_list_cell)
        mlb = MultiLabelBinarizer(sparse_output=False)
        try:
            genres_m = pd.DataFrame(mlb.fit_transform(genres_parsed), columns=[f'genre_{g}' for g in mlb.classes_], index=df.index)
            df = pd.concat([df, genres_m], axis=1)
        except Exception:
            logging.info('No se pudieron aplicar MultiLabelBinarizer a `genres`.')

    # ---------------- TF-IDF EN SINOPSIS (opcional) ----------------
    # Si el usuario pide tfidf_max_features>0 y existe la columna `synopsis_api`,
    # transformamos el texto en features numéricas usando TF-IDF y las añadimos al df.
    if tfidf_max_features and 'synopsis_api' in df.columns:
        logging.info('Transformando sinopsis con TF-IDF (max_features=%d). Esto puede incrementar mucho el uso de memoria.', tfidf_max_features)
        df['synopsis_api'] = df['synopsis_api'].fillna('')
        tfidf = TfidfVectorizer(stop_words='english', max_features=tfidf_max_features)
        synopsis_features = tfidf.fit_transform(df['synopsis_api'])
        feature_names = [f'syn_tf_{w}' for w in tfidf.get_feature_names_out()]
        # Convertir a DataFrame denso — atención al uso de memoria
        synopsis_df = pd.DataFrame(synopsis_features.toarray(), columns=feature_names, index=df.index)
        df = pd.concat([df, synopsis_df], axis=1)
        logging.info('Añadidos %d features de sinopsis (TF-IDF).', len(feature_names))
    else:
        feature_names = []

    # SELECCIÓN DE FEATURES que usaremos para clustering (puedes modificar esta lista)
    feature_cols = ['score_num','episodes_num','year_num','rating_count','rating_mean','rating_std']
    # Añadir columnas one-hot de tipo
    feature_cols += [c for c in df.columns if c.startswith('type_')]
    # Añadir algunos géneros si existen (limitamos a 12 para no explotar dimensionalidad)
    genre_cols = [c for c in df.columns if c.startswith('genre_')]
    if len(genre_cols) > 0:
        if len(genre_cols) > 12:
            freq = df[genre_cols].sum().sort_values(ascending=False)
            keep = list(freq.head(12).index)
        else:
            keep = genre_cols
        feature_cols += keep

    # Incluir features de TF-IDF si se generaron
    if feature_names:
        feature_cols += feature_names

    # Filtrar sólo las columnas que realmente existen en df
    feature_cols = [c for c in feature_cols if c in df.columns]
    logging.info('Feature columns used for clustering: %s', feature_cols)

    # Crear el DataFrame de features y rellenar nulos numéricos con la mediana
    df_features = df[feature_cols].copy()
    for c in df_features.columns:
        if pd.api.types.is_numeric_dtype(df_features[c]):
            df_features[c] = df_features[c].fillna(df_features[c].median())

    return df, df_features


# ------------------ CLUSTERING Y EVALUACIÓN ------------------

def run_clustering_and_evaluate(df, X, out_dir=OUTPUT_DIR, skip_kmeans=False, skip_agglomerative=False, skip_dbscan=False, aggl_sample_size=5000):
    """Aplica varios algoritmos de clustering sobre X y guarda métricas y etiquetas.

    Parámetros:
    - df: DataFrame original (usado para calcular pureza con `primary_genre` y `score_category`)
    - X: DataFrame con features numéricos listos para clustering
    - out_dir: directorio donde guardar resultados
    """
    results = []
    # Extraer numpy array para sklearn
    X_np = X.values

    # Escalado: StandardScaler (media 0, varianza 1)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_np)

    # PCA: reducir dimensionalidad a n_comp componentes (útil para acelerar clustering y medir varianza)
    n_comp = min(10, Xs.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    Xp = pca.fit_transform(Xs)
    logging.info('Explained variance ratio (first 5): %s', pca.explained_variance_ratio_[:5].tolist())

    # ---------------- KMEANS (barrido de k) ----------------
    if skip_kmeans:
        logging.info('Skipping KMeans because skip_kmeans=True')
    else:
        for k in range(2, 11):
            # Crear modelo KMeans con semilla fija para reproducibilidad
            t0 = time.time()
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            # Ajustar sobre las primeras 5 componentes PCA (reduce ruido)
            labels = km.fit_predict(Xp[:, :5])
            # Métricas internas
            sil = silhouette_score(Xp[:, :5], labels)
            db = davies_bouldin_score(Xp[:, :5], labels)
            t1 = time.time()
            logging.info('KMeans k=%d done: silhouette=%.4f purity_genre=%.4f (%.1fs)', k, sil, cluster_purity(df['primary_genre'].fillna('NA'), labels), t1 - t0)
            # Pureza externa usando primary_genre y score_category
            pur_gen = cluster_purity(df['primary_genre'].fillna('NA'), labels)
            pur_score = cluster_purity(df['score_category'].astype(str).replace('nan','NA'), labels)
            results.append({'method':'kmeans','params':f'k={k}','silhouette':sil,'db':db,'pur_genre':pur_gen,'pur_score':pur_score, 'time_s': t1-t0})
            # Guardar etiquetas por anime para análisis posterior
            out_labels = pd.DataFrame({'animeID': df.get('animeID', df.index), 'label': labels})
            out_labels.to_csv(os.path.join(out_dir, f'labels_kmeans_k{k}.csv'), index=False)

    # ---------------- AGGLOMERATIVE (jerárquico) ----------------
    # AGGLOMERATIVE: esta sección puede ser muy costosa en tiempo y memoria para >~6000 puntos.
    if not skip_agglomerative:
        for linkage in ['ward','average','complete']:
            for k in range(2,9):
                if linkage == 'ward' and Xp.shape[1] < 1:
                    continue
                try:
                    n_samples = Xp.shape[0]
                    # Si el dataset es grande, trabajamos sobre una muestra aleatoria para evitar OOM/CPU
                    if n_samples > aggl_sample_size:
                        idx = np.random.choice(n_samples, size=aggl_sample_size, replace=False)
                        Xp_sub = Xp[idx, :5]
                        t0 = time.time()
                        ac = AgglomerativeClustering(n_clusters=k, linkage=linkage)
                        labels_sub = ac.fit_predict(Xp_sub)
                        t1 = time.time()
                        sil = silhouette_score(Xp_sub, labels_sub)
                        db = davies_bouldin_score(Xp_sub, labels_sub)
                        pur_gen = cluster_purity(df.iloc[idx]['primary_genre'].fillna('NA'), labels_sub)
                        pur_score = cluster_purity(df.iloc[idx]['score_category'].astype(str).replace('nan','NA'), labels_sub)
                        results.append({'method':'agglomerative_sampled','params':f'link={linkage},k={k},sample={aggl_sample_size}','silhouette':sil,'db':db,'pur_genre':pur_gen,'pur_score':pur_score, 'time_s': t1-t0})
                        # Guardar etiquetas sólo para la muestra
                        out_labels = pd.DataFrame({'animeID': df.iloc[idx].get('animeID', df.iloc[idx].index), 'label': labels_sub})
                        out_labels.to_csv(os.path.join(out_dir, f'labels_aggl_{linkage}_k{k}_sample{aggl_sample_size}.csv'), index=False)
                        logging.info('Agglomerative (sample) link=%s k=%d done: silhouette=%.4f purity_genre=%.4f (%.1fs)', linkage, k, sil, pur_gen, t1-t0)
                    else:
                        t0 = time.time()
                        ac = AgglomerativeClustering(n_clusters=k, linkage=linkage)
                        labels = ac.fit_predict(Xp[:, :5])
                        t1 = time.time()
                        sil = silhouette_score(Xp[:, :5], labels)
                        db = davies_bouldin_score(Xp[:, :5], labels)
                        pur_gen = cluster_purity(df['primary_genre'].fillna('NA'), labels)
                        pur_score = cluster_purity(df['score_category'].astype(str).replace('nan','NA'), labels)
                        results.append({'method':'agglomerative','params':f'link={linkage},k={k}','silhouette':sil,'db':db,'pur_genre':pur_gen,'pur_score':pur_score, 'time_s': t1-t0})
                        out_labels = pd.DataFrame({'animeID': df.get('animeID', df.index), 'label': labels})
                        out_labels.to_csv(os.path.join(out_dir, f'labels_aggl_{linkage}_k{k}.csv'), index=False)
                        logging.info('Agglomerative link=%s k=%d done: silhouette=%.4f purity_genre=%.4f (%.1fs)', linkage, k, sil, pur_gen, t1-t0)
                except Exception as e:
                    logging.warning('Agglomerative link=%s k=%d failed: %s', linkage, k, e)
                    continue

    # ---------------- DBSCAN (densidad) ----------------
    eps_list = [0.5, 0.7]
    min_samples_list = [5,7]
    for eps in eps_list:
        for ms in min_samples_list:
            if skip_dbscan:
                logging.info('Skipping DBSCAN eps=%s ms=%d because skip_dbscan=True', eps, ms)
                continue
            try:
                t0 = time.time()
                dbs = DBSCAN(eps=eps, min_samples=ms)
                labels = dbs.fit_predict(Xp[:, :5])
                t1 = time.time()
            except Exception as e:
                logging.warning('DBSCAN eps=%.2f ms=%d failed: %s', eps, ms, e)
                continue
            # Calcular número de clusters (ignorando etiqueta -1 = ruido)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            # Si no hay clusters válidos, saltar métricas que no tienen sentido
            if n_clusters <= 1:
                logging.info('DBSCAN eps=%.2f min_samples=%d produced %d clusters — skipping metrics', eps, ms, n_clusters)
                continue
            sil = silhouette_score(Xp[:, :5], labels)
            db = davies_bouldin_score(Xp[:, :5], labels)
            pur_gen = cluster_purity(df['primary_genre'].fillna('NA'), labels)
            pur_score = cluster_purity(df['score_category'].astype(str).replace('nan','NA'), labels)
            results.append({'method':'dbscan','params':f'eps={eps},min_samples={ms}','silhouette':sil,'db':db,'pur_genre':pur_gen,'pur_score':pur_score, 'time_s': t1-t0})
            out_labels = pd.DataFrame({'animeID': df.get('animeID', df.index), 'label': labels})
            out_labels.to_csv(os.path.join(out_dir, f'labels_dbscan_eps{eps}_ms{ms}.csv'), index=False)
            logging.info('DBSCAN eps=%.2f ms=%d done: silhouette=%.4f purity_genre=%.4f (%.1fs)', eps, ms, sil, pur_gen, t1-t0)

    # Guardar resultados en un CSV para análisis posterior
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(out_dir, 'cluster_results.csv'), index=False)
    logging.info('Saved cluster_results.csv with %d rows', len(results_df))

    return results_df


# ------------------ MAIN ------------------

def main(args):
    # Cargar y fusionar datasets (animes + dataset_completo + rating aggregates)
    merged = load_and_merge(animes_path=args.animes, dataset_path=args.dataset, ratings_path=args.ratings)
    merged.to_csv(os.path.join(OUTPUT_DIR, 'merged_raw.csv'), index=False)
    logging.info('Saved merged_raw.csv')

    # Preprocesar y obtener la matriz de features
    df, X = preprocess_and_features(merged, tfidf_max_features=args.tfidf_features)
    df.to_csv(os.path.join(OUTPUT_DIR, 'merged_data.csv'), index=False)
    logging.info('Saved merged_data.csv; features shape: %s', X.shape)

    # Ejecutar clustering y evaluación
    results_df = run_clustering_and_evaluate(df, X,
                                            skip_agglomerative=args.skip_agglomerative,
                                            skip_dbscan=args.skip_dbscan,
                                            aggl_sample_size=args.aggl_sample_size)
    logging.info('Clustering complete. Results saved in %s', OUTPUT_DIR)


if __name__ == '__main__':
    # Argumentos opcionales para especificar rutas de los CSVs
    parser = argparse.ArgumentParser(description='Clustering analysis for anime datasets')
    parser.add_argument('--animes', default='animes.csv', help='Ruta a animes.csv')
    parser.add_argument('--dataset', default='dataset_completo.csv', help='Ruta a dataset_completo.csv')
    parser.add_argument('--ratings', default='ratings.csv', help='Ruta a ratings.csv (puede ser grande)')
    parser.add_argument('--skip-agglomerative', dest='skip_agglomerative', action='store_true', help='No ejecutar Agglomerative (muy costoso)')
    parser.add_argument('--skip-dbscan', dest='skip_dbscan', action='store_true', help='No ejecutar DBSCAN (costoso)')
    parser.add_argument('--aggl-sample-size', dest='aggl_sample_size', type=int, default=5000, help='Si Agglomerative > sample_size, ejecutar sobre muestra aleatoria de este tamaño')
    parser.add_argument('--tfidf-features', dest='tfidf_features', type=int, default=0, help='Número de features TF-IDF desde sinopsis (0 desactiva)')
    args = parser.parse_args()
    main(args)

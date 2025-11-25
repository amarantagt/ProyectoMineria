"""
generate_visualizations.py

Carga `outputs/cluster_results.csv`, `outputs/merged_data.csv` y archivos de etiquetas (labels_kmeans_*.csv, labels_aggl_*.csv, labels_dbscan_*.csv)
Genera y guarda los siguientes gráficos en `outputs/plots/`:
 - silhouette_vs_k_kmeans.png
 - purity_genre_vs_k_kmeans.png
 - purity_score_vs_k_kmeans.png
 - best_kmeans_clusters_2d.png (UMAP/PCA 2D)
 - cluster_genre_composition_best_k.png
 - cluster_scorecategory_composition_best_k.png

Ejecución:
    python generate_visualizations.py

Dependencias: pandas, numpy, matplotlib, seaborn, scikit-learn, umap-learn (opcional)
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

OUT = 'outputs'
PLOTS = os.path.join(OUT, 'plots')
os.makedirs(PLOTS, exist_ok=True)

# Cargar resultados
res_path = os.path.join(OUT, 'cluster_results.csv')
merged_path = os.path.join(OUT, 'merged_data.csv')
if not os.path.exists(res_path) or not os.path.exists(merged_path):
    logging.error('No se encuentran los archivos requeridos en outputs/. Ejecuta primero clustering_analysis.py')
    raise SystemExit(1)

res = pd.read_csv(res_path)
df = pd.read_csv(merged_path)

# Filtrar KMeans
km_res = res[res['method']=='kmeans'].copy()
km_res['k'] = km_res['params'].str.replace('k=','').astype(int)
km_res = km_res.sort_values('k')

# Plot silhouette vs k
plt.figure(figsize=(8,5))
sns.lineplot(data=km_res, x='k', y='silhouette', marker='o')
plt.title('KMeans: Silhouette vs k')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, 'silhouette_vs_k_kmeans.png'), dpi=150)
plt.close()
logging.info('Saved silhouette_vs_k_kmeans.png')

# Plot purity (genre and score) vs k
plt.figure(figsize=(8,5))
sns.lineplot(data=km_res, x='k', y='pur_genre', marker='o', label='purity_genre')
sns.lineplot(data=km_res, x='k', y='pur_score', marker='o', label='purity_score')
plt.title('KMeans: Purity (genre & score) vs k')
plt.xlabel('k')
plt.ylabel('Purity')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, 'purity_vs_k_kmeans.png'), dpi=150)
plt.close()
logging.info('Saved purity_vs_k_kmeans.png')

# Elegir el k con mayor silhouette
best_row = km_res.loc[km_res['silhouette'].idxmax()]
best_k = int(best_row['k'])
logging.info('Best KMeans k by silhouette: %d', best_k)

# Cargar etiquetas correspondientes
labels_file = os.path.join(OUT, f'labels_kmeans_k{best_k}.csv')
if not os.path.exists(labels_file):
    logging.error('No se encuentra %s', labels_file)
    raise SystemExit(1)
labels = pd.read_csv(labels_file)

# Unir con df
# labels may contain animeID; df contains animeID
if 'animeID' in df.columns and 'animeID' in labels.columns:
    merged = pd.merge(df, labels, on='animeID', how='inner')
else:
    # fallback: use index
    labels = labels.rename(columns={labels.columns[0]:'animeID'})
    merged = pd.concat([df.reset_index(), labels['label']], axis=1)

# Proyección 2D: intentar UMAP, si no está disponible usar t-SNE o PCA
X_cols = [c for c in df.columns if c in ['score_num','episodes_num','year_num','rating_count','rating_mean','rating_std'] or c.startswith('type_') or c.startswith('genre_')]
X = merged[X_cols].fillna(0).values

try:
    import umap
    reducer = umap.UMAP(n_components=2, random_state=42)
    X2 = reducer.fit_transform(X)
    method_used = 'UMAP'
except Exception:
    try:
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        X2 = tsne.fit_transform(X)
        method_used = 't-SNE'
    except Exception:
        pca = PCA(n_components=2, random_state=42)
        X2 = pca.fit_transform(X)
        method_used = 'PCA'

logging.info('Projection method used: %s', method_used)
merged['x2_1'] = X2[:,0]
merged['x2_2'] = X2[:,1]

plt.figure(figsize=(9,7))
unique_labels = merged['label'].unique()
palette = sns.color_palette('tab20', n_colors=len(unique_labels))

sns.scatterplot(data=merged, x='x2_1', y='x2_2', hue='label', palette=palette, s=15, linewidth=0, alpha=0.8)
plt.title(f'{method_used} 2D projection colored by KMeans k={best_k} labels')
plt.legend(title='cluster', bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, f'best_kmeans_clusters_2d_k{best_k}.png'), dpi=150)
plt.close()
logging.info('Saved best_kmeans_clusters_2d_k%d.png', best_k)

# Composición por género por cluster (usando primary_genre)
comp = merged.groupby(['label','primary_genre']).size().reset_index(name='count')
# calcular proporciones por cluster
total_per_cluster = comp.groupby('label')['count'].transform('sum')
comp['prop'] = comp['count'] / total_per_cluster

# Tomar top genres overall for plotting clarity
top_genres = merged['primary_genre'].value_counts().head(8).index.tolist()
comp_small = comp[comp['primary_genre'].isin(top_genres)]

plt.figure(figsize=(10,6))
# pivot for stacked bar
pv = comp_small.pivot(index='label', columns='primary_genre', values='prop').fillna(0)
pv.plot(kind='bar', stacked=True, colormap='tab20', width=0.8)
plt.ylabel('Proportion')
plt.title(f'Cluster composition by primary_genre (KMeans k={best_k})')
plt.legend(title='genre', bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, f'cluster_genre_composition_k{best_k}.png'), dpi=150)
plt.close()
logging.info('Saved cluster_genre_composition_k%d.png', best_k)

# Composición por score_category
if 'score_category' in merged.columns:
    comp2 = merged.groupby(['label','score_category']).size().reset_index(name='count')
    total_per_cluster2 = comp2.groupby('label')['count'].transform('sum')
    comp2['prop'] = comp2['count'] / total_per_cluster2
    pv2 = comp2.pivot(index='label', columns='score_category', values='prop').fillna(0)
    plt.figure(figsize=(8,5))
    pv2.plot(kind='bar', stacked=True, colormap='Set2', width=0.8)
    plt.ylabel('Proportion')
    plt.title(f'Cluster composition by score_category (KMeans k={best_k})')
    plt.legend(title='score_category', bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, f'cluster_scorecategory_composition_k{best_k}.png'), dpi=150)
    plt.close()
    logging.info('Saved cluster_scorecategory_composition_k%d.png', best_k)
else:
    logging.info('score_category no está disponible en merged_data.csv; omitiendo composición por score_category')

logging.info('All plots saved in %s', PLOTS)

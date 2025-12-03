"""Wrapper para ejecutar sólo Agglomerative (usa muestreo si el dataset es grande).

Ejemplo:
    python clustering_agglomerative.py --aggl-sample-size 4000
"""
import argparse
import logging
from clustering_analysis import load_and_merge, preprocess_and_features, run_clustering_and_evaluate, OUTPUT_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def main(args):
    merged = load_and_merge(animes_path=args.animes, dataset_path=args.dataset, ratings_path=args.ratings)
    df, X = preprocess_and_features(merged, tfidf_max_features=args.tfidf_features)
    # Ejecutar sólo Agglomerative (puede usar muestreo interno)
    run_clustering_and_evaluate(df, X, out_dir=OUTPUT_DIR, skip_kmeans=True, skip_agglomerative=False, skip_dbscan=True, aggl_sample_size=args.aggl_sample_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Agglomerative clustering only')
    parser.add_argument('--animes', default='animes.csv')
    parser.add_argument('--dataset', default='dataset_completo.csv')
    parser.add_argument('--ratings', default='ratings.csv')
    parser.add_argument('--aggl-sample-size', dest='aggl_sample_size', type=int, default=5000)
    parser.add_argument('--tfidf-features', dest='tfidf_features', type=int, default=0, help='Número de features TF-IDF desde sinopsis (0 desactiva)')
    args = parser.parse_args()
    main(args)

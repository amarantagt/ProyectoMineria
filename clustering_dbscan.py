"""Wrapper para ejecutar sólo DBSCAN.

Ejemplo:
    python clustering_dbscan.py
"""
import argparse
import logging
from clustering_analysis import load_and_merge, preprocess_and_features, run_clustering_and_evaluate, OUTPUT_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def main(args):
    merged = load_and_merge(animes_path=args.animes, dataset_path=args.dataset, ratings_path=args.ratings)
    df, X = preprocess_and_features(merged, tfidf_max_features=args.tfidf_features)
    # Ejecutar sólo DBSCAN
    run_clustering_and_evaluate(df, X, out_dir=OUTPUT_DIR, skip_kmeans=True, skip_agglomerative=True, skip_dbscan=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run DBSCAN clustering only')
    parser.add_argument('--animes', default='animes.csv')
    parser.add_argument('--dataset', default='dataset_completo.csv')
    parser.add_argument('--ratings', default='ratings.csv')
    parser.add_argument('--tfidf-features', dest='tfidf_features', type=int, default=0, help='Número de features TF-IDF desde sinopsis (0 desactiva)')
    args = parser.parse_args()
    main(args)

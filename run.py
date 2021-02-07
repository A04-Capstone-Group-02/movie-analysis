import json
from src.eda import generate_figures
from src.etl import get_data, download_dataset
from src.clustering import calc_all_embeddings
from src.classification import model
import sys


def main(targets):
    etl_params = json.load(open('config/etl.json'))
    autophrase_params = json.load(open('config/autophrase.json'))
    eda_params = json.load(open('config/eda.json'))
    clustering_params = json.load(open('config/clustering.json'))
    clf_params = json.load(open('config/classification.json'))

    if 'download' in targets:
        download_dataset()

    if 'data' in targets:
        get_data(autophrase_params, **etl_params)

    if 'eda' in targets:
        generate_figures(**eda_params)
    
    if 'clustering' in targets:
        calc_all_embeddings(clustering_params)
        
    if 'classification' in targets:
        model(clf_params)
        
    if 'train' in targets:
        ...

    if 'test' in targets:
        ...


if __name__ == '__main__':
    main(sys.argv[1:])

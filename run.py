import json
from src.eda import generate_figures
from src.etl import get_data, download_dataset
import sys


def main(targets):
    autophrase_params = json.load(open('config/autophrase.json'))
    eda_params = json.load(open('config/eda.json'))

    if 'download' in targets:
        download_dataset()

    if 'data' in targets:
        get_data(autophrase_params)

    if 'eda' in targets:
        generate_figures(**eda_params)

    if 'train' in targets:
        ...

    if 'test' in targets:
        ...


if __name__ == '__main__':
    main(sys.argv[1:])

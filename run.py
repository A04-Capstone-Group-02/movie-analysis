import json
from src.etl import get_data
import sys


def main(targets):
    autophrase_params = json.load(open('config/autophrase_params.json'))

    if 'data' in targets:
        get_data(autophrase_params)

    if 'train' in targets:
        ...

    if 'test' in targets:
        ...


if __name__ == '__main__':
    main(sys.argv[1:])

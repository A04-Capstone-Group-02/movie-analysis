import json
import os
from src import etl
import sys


def main(targets):
    autophrase_params = json.load(open('config/autophrase_params.json'))
    autophrase_params = ' '.join([f'{param}={value}' for param, value in autophrase_params.items()])

    if 'etl' in targets:
        etl.main()

    if 'train' in targets:
        # Run AutoPhrase
        os.system(f'cd AutoPhrase && {autophrase_params} ./auto_phrase.sh && {autophrase_params} ./phrasal_segmentation.sh')

    if 'test' in targets:
        ...


if __name__ == '__main__':
    main(sys.argv[1:])
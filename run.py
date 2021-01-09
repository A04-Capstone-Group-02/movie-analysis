from src import etl
import sys


def main(targets):
    if 'etl' in targets:
        etl.main()


if __name__ == '__main__':
    main(sys.argv[1:])
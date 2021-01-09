import os
import pandas as pd
from pandas_profiling import ProfileReport


def main():
    def normalize(x):
        dictionary = eval(x)
        if dictionary:
            return list(dictionary.values())

    def clean_summary(summary):
        return (
            summary
            .str.replace('{{.*?}}', '')  # Remove Wikipedia tags
            .str.replace('http\S+', '')  # Remove URLs
            .str.replace('\s+', ' ')  # Combine whitespace
            .str.strip()  # Strip whitespace
        )

    movies = pd.read_csv(
        'data/raw/movie.metadata.tsv',
        converters={'languages': normalize, 'countries': normalize, 'genres': normalize},
        delimiter='\t',
        header=None,
        index_col='id',
        names='id name date revenue runtime languages countries genres'.split(),
        usecols=[0, 2, 3, 4, 5, 6, 7, 8]
    ).assign(date=lambda x: pd.to_datetime(x.date, errors='coerce'))

    summaries = pd.read_csv(
        'data/raw/plot_summaries.txt',
        delimiter='\t',
        header=None,
        index_col='id',
        names='id summary'.split()
    ).assign(summary=lambda x: clean_summary(x.summary))

    df = movies.merge(summaries, on='id').sort_values('date').reset_index(drop=True)
    os.mkdir('data/out')
    df.to_pickle('data/out/data.pkl')
    ProfileReport(df).to_file('data/out/report.html')

    with open(f'data/out/summaries.txt', 'w') as f:
        f.write('\n'.join(df.summary))


if __name__ == '__main__':
    main()
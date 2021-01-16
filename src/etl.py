import os
import pandas as pd
from pandas_profiling import ProfileReport


def get_data(autophrase_params):
    # Make data directories
    os.makedirs('data/temp', exist_ok=True)
    os.makedirs('data/out', exist_ok=True)

    # Read in raw data
    def normalize(x):
        dictionary = eval(x)
        if dictionary:
            return list(dictionary.values())

    def clean_summary(summary):
        return (
            summary
            .str.replace(r'{{.*?}}', '')  # Remove Wikipedia tags
            .str.replace(r'http\S+', '')  # Remove URLs
            .str.replace(r'\s+', ' ')  # Combine whitespace
            .str.strip()  # Strip whitespace
            .replace('', pd.NA)  # Replace empty strings with NA
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
    ).assign(summary=lambda x: clean_summary(x.summary)).dropna()

    # Combine movie metadata and plot summaries into df
    df = movies.merge(summaries, on='id').sort_values('date').reset_index(drop=True)

    # Run AutoPhrase on plot summaries
    with open(f'data/temp/summaries.txt', 'w') as f:
        f.write('\n'.join(df.summary))

    autophrase_params = ' '.join([f'{param}={value}' for param, value in autophrase_params.items()])
    os.system(f'cd AutoPhrase && {autophrase_params} ./auto_phrase.sh && {autophrase_params} ./phrasal_segmentation.sh')

    # Add phrases to df
    df['phrases'] = pd.read_csv(
        'model/autophrase/segmentation.txt',
        delimiter=r'\n',
        engine='python',
        header=None,
        squeeze=True
    ).str.findall(r'<phrase>(.+?)</phrase>').values

    # Export df
    df.to_pickle('data/out/data.pkl')
    ProfileReport(df).to_file('data/out/report.html')

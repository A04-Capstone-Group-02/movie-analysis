import numpy as np
import os
import pandas as pd
from pandas_profiling import ProfileReport
import re


def get_data(autophrase_params):
    # Make data directories
    os.makedirs('data/temp', exist_ok=True)
    os.makedirs('data/out', exist_ok=True)

    # Read in raw data
    def normalize_languages(x):
        def is_utf8(value):
            try:
                value.encode()
            except UnicodeEncodeError:
                return False
            return True

        def sub(value):
            return re.sub(r' [Ll]anguages?', '', value)

        return sorted(np.unique([sub(value) for value in eval(x).values() if is_utf8(value)]))

    def normalize_countries(x):
        return sorted(eval(x).values())

    def normalize_genres(x):
        def sub(value):
            # Replace with a more common genre name
            if value == 'Animal Picture':
                return 'Animals'
            if value in ['Biographical film', 'Biopic [feature]']:
                return 'Biography'
            if value == 'Buddy Picture':
                return 'Buddy'
            if value == 'Comdedy':
                return 'Comedy'
            if value == 'Coming of age':
                return 'Coming-of-age'
            if value == 'Detective fiction':
                return 'Detective'
            if value == 'Education':
                return 'Educational'
            if value in ['Gay Interest', 'Gay Themed']:
                return 'Gay'
            if value == 'Gross out':
                return 'Gross-out'
            if value == 'Pornography':
                return 'Pornographic'
            if value == 'Social issues':
                return 'Social problem'
            return re.sub(' [Ff]ilms?| [Mm]ovies?', '', value)

        return sorted(np.unique([sub(value) for value in eval(x).values()]))

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
        converters={'languages': normalize_languages, 'countries': normalize_countries, 'genres': normalize_genres},
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
    with open('data/temp/summaries.txt', 'w') as f:
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
    ).str.findall(r'<phrase>(.+?)</phrase>').apply(lambda x: [s.lower() for s in x]).apply(np.unique).apply(list).values

    # Export df
    df.to_pickle('data/out/data.pkl')
    ProfileReport(df).to_file('data/out/report.html')

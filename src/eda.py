from collections import Counter
import matplotlib.pyplot as plt
from math import ceil
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer


def top_phrases_by_year(df, year_start, year_end, phrase_count_threshold):
    """Return a Figure with the top phrases by year, ranked by tf-idf"""
    # Create a Series with a list of phrases (allowing duplicates) for each year
    phrases_by_year = df.query(f'{year_start} <= date.dt.year <= {year_end}').groupby('year').phrases.agg(lambda x: sum(x, []))

    # Create a DataFrame with the count of each phrase for each year
    phrase_counts_by_year = pd.DataFrame(phrases_by_year.apply(Counter).tolist(), index=phrases_by_year.index).fillna(0).sort_index(1)
    # Only include phrases that appear at least `phrase_count_threshold` times total
    phrase_counts_by_year = phrase_counts_by_year.loc[:, phrase_counts_by_year.sum().ge(phrase_count_threshold)]

    # Create a DataFrame with the tf-idf of each phrase for each year
    phrase_tfidfs_by_year = pd.DataFrame(
        TfidfTransformer(sublinear_tf=True).fit_transform(phrase_counts_by_year).toarray(),
        index=phrase_counts_by_year.index,
        columns=phrase_counts_by_year.columns
    )

    # Plot the top phrases by year, ranked by tf-idf
    ncols = 5
    nrows = ceil(len(phrase_tfidfs_by_year) / ncols)
    fig, axes = plt.subplots(nrows, ncols, sharex=True, figsize=(ncols * 2.5, nrows * 2), constrained_layout=True)
    years, axes = phrase_tfidfs_by_year.index, axes.flatten()
    for year, ax in zip(years, axes):
        phrase_tfidfs_by_year.loc[year].nlargest(10)[::-1].plot.barh(ax=ax)
        ax.set(title=year, xlabel='tf-idf')
        ax.tick_params(bottom=False, labelbottom=False)
    for ax in axes[len(years):]:
        ax.axis('off')
    return fig


def generate_figures(data_in, data_out, dpi, **kwargs):
    """Generate and save figures"""
    # Read in data, add new columns as needed
    df = pd.read_pickle(data_in)
    df['year'] = df.date.dt.year.astype('Int64')

    # Generate and save figures
    os.makedirs(data_out, exist_ok=True)
    top_phrases_by_year(df, **kwargs).savefig(f'{data_out}/top_phrases_by_year.png', dpi=dpi, bbox_inches='tight')

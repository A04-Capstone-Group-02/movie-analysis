from bar_chart_race import bar_chart_race
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MultipleLocator
from math import ceil
import numpy as np
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer


def number_movies_per_year_bar_chart(df, data_out, dpi, **kwargs):
    """Save a Figure with a bar chart of the number of movies per year"""
    fig, ax = plt.subplots(figsize=(7, 1.5))
    number_movies_per_year = df.year.value_counts().sort_index()
    number_movies_per_year.plot.bar(ax=ax)
    ax.set(title='# Movies in Dataset', xlabel='Year', ylabel='# Movies')
    ax.tick_params(rotation=0)
    ax.xaxis.set_major_locator(FixedLocator([i for i, year in enumerate(number_movies_per_year.index) if year % 10 == 0]))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    fig.savefig(f'{data_out}/number_movies_per_year_bar_chart.png', dpi=dpi, bbox_inches='tight')


def phrase_tfidfs_by_year(df, year_start, year_end, phrase_count_threshold, **kwargs):
    """Return a DataFrame with the tf-idf of each phrase for each year (phrases are terms and years are documents)"""
    # Create a Series with a list of phrases (allowing duplicates) for each year
    phrases_by_year = df.query(f'{year_start} <= date.dt.year <= {year_end}').groupby('year').phrases.agg(lambda x: sum(x, []))

    # Create a DataFrame with the count of each phrase for each year
    phrase_counts_by_year = pd.DataFrame(phrases_by_year.apply(Counter).tolist(), index=phrases_by_year.index).fillna(0).sort_index(1)
    # Only include phrases that appear at least `phrase_count_threshold` times total
    phrase_counts_by_year = phrase_counts_by_year.loc[:, phrase_counts_by_year.sum().ge(phrase_count_threshold)]

    # Create a DataFrame with the tf-idf of each phrase for each year
    tfidfs = pd.DataFrame(
        TfidfTransformer(sublinear_tf=True).fit_transform(phrase_counts_by_year).toarray(),
        index=phrase_counts_by_year.index,
        columns=phrase_counts_by_year.columns
    )
    return tfidfs


def top_phrases_by_year_bar_chart(df, data_out, stop_words, dpi, **kwargs):
    """Save a Figure with a bar chart of the top phrases (ranked by tf-idf) for each year"""
    tfidfs = phrase_tfidfs_by_year(df, **kwargs).drop(columns=stop_words, errors='ignore')

    ncols = 5
    nrows = ceil(len(tfidfs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, sharex=True, figsize=(ncols * 2.5, nrows * 1.9), constrained_layout=True)
    years, axes = tfidfs.index, axes.flatten()
    for year, ax in zip(years, axes):
        tfidfs.loc[year].nlargest(10)[::-1].plot.barh(ax=ax)
        ax.set(title=year, xlabel='tf-idf')
        ax.tick_params(bottom=False, labelbottom=False)
    for ax in axes[len(years):]:
        ax.axis('off')
    fig.savefig(f'{data_out}/top_phrases_by_year_bar_chart.png', dpi=dpi, bbox_inches='tight')


def top_phrases_by_year_bar_chart_race(df, data_out, stop_words, n_bars, dpi, fps, seconds_per_period, **kwargs):
    """Save an MP4 with a bar chart race of the top phrases (ranked by tf-idf) for each year"""
    tfidfs = phrase_tfidfs_by_year(df, **kwargs).drop(columns=stop_words, errors='ignore')
    # Only keep columns for the top phrases to reduce unnecessary computation
    tfidfs = tfidfs[np.unique(tfidfs.apply(lambda x: x.nlargest(n_bars).index.tolist(), 1).sum())]

    # Prepare figure
    fig, ax = plt.subplots(dpi=dpi, figsize=(6, 3.5))
    fig.suptitle(f'Top Phrases From Wikipedia Movie Plot Summaries\n{tfidfs.index[0]}–{tfidfs.index[-1]}', y=.94)
    ax.set_facecolor('.9')
    ax.set_xlabel('tf-idf', style='italic')
    ax.tick_params(labelbottom=False, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.subplots_adjust(left=.4, right=.95, top=.82)

    # Create bar chart race
    bar_chart_race(
        df=tfidfs,
        fig=fig,
        filename=f'{data_out}/top_phrases_by_year_bar_chart_race.mp4',
        label_bars=False,
        n_bars=n_bars,
        period_fmt='{x:.0f}',
        period_label=dict(x=.96, y=.04, ha='right', va='bottom', size=30),
        period_length=seconds_per_period * 1000,
        steps_per_period=int(seconds_per_period * fps),
    )


def generate_figures(data_in, data_out, **kwargs):
    """Generate figures"""
    # Read in data, add new columns as needed
    df = pd.read_pickle(data_in)
    df['year'] = df.date.dt.year.astype('Int64')

    # Generate figures
    os.makedirs(data_out, exist_ok=True)
    number_movies_per_year_bar_chart(df, data_out, **kwargs)
    top_phrases_by_year_bar_chart(df, data_out, **kwargs)
    top_phrases_by_year_bar_chart_race(df, data_out, **kwargs)

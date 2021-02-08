# Analyzing Movies Using Phrase Mining

## Setup

### Clone the repository

```
git clone --recursive https://github.com/A04-Capstone-Group-02/movie-analysis.git
```

### Download dataset

Download the [CMU Movie Summary Corpus dataset](http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz) and move its files to `data/raw/`, or run the `download` target.

Note that to run this repository on the UCSD DSMLP server, the dataset must be manually uploaded, since the DSMLP server cannot connect to the data source link.

### Docker

Build a docker container with the `Dockerfile` or the remote image `991231/movie-analysis` in the docker hub.

### Note

To run the `clustering` target, we highly recommend enabling GPU to ensure reasonable running time, since this target heavily interacts with a transformer model. Running other targets without GPU will not be an issue.

## Run

Execute the running script with the following command:

```
python run.py [all] [test] [download] [data] [eda] [classification] [clustering]
```

### `all` target

Run `data`, `eda`, `classification` and `clustering` targets in this exact order.

### `test` target

Runs the same 4 targets in the same order as the `all` target, but using the test data in `test/data/raw` and the test configurations.

### `download` target

Download the CMU Movie Summary Corpus dataset and set up the `data` directory.

### `data` target

Run the ETL pipeline to process the raw data. This target will run AutoPhrase to extract quality phrases, clean categories, combine the processed data into a dataframe, and generate a profile report of the dataset.

The configuration file for this target is `etl.json` (or `etl_test.json` for `test` target), which contains the following items:

- `data_in`: the path to the input data (relative to the root)
- `false_positive_phrases`: phrases to remove from the quality phrase list
- `false_positive_substrings`: substrings to remove from the quality phrase list

The configuration file for the AutoPhrase submodule is `autophrase.json`, which contains the following items:

- `MIN_SUP`: the minimum count of a phrase to include in the training process
- `MODEL`: the path to the output model (relative to the root)
- `RAW_TRAIN`: the path to the raw corpus for training (relative to the root)
- `TEXT_TO_SEG`: the path to the raw corpus for segmentation (relative to the root)
- `THREAD`: the number of threads to use

### `eda` target

Run the EDA pipeline. This target will find the temporal change of quality phrase distributions and generate visualizations to show the findings.

The configuration file for this target is `eda.json` (or `eda_test.json` for `test` target), which contains the following items:

- `data_in`: the path to the input data (relative to the root)
- `data_out`: the path to the output directory (relative to the root)
- `year_start`: the earliest year to analyze
- `year_end`: the latest year to analyze
- `phrase_count_threshold`: the minimum count of a quality phrase to be included in the analysis
- `stop_words`: the stop words to ignore in the analysis
- `n_bars`: number of bars to display in the bar plots
- `dpi`: subplot dpi (dot per inches)
- `fps`: fps (frame per second) of the bar chart race animation
- `seconds_per_period`: the time each subplot will take in the bar chart race animation

### `classification` target

Run the classification pipeline. This target will transform the data into a TF-IDF matrix, fit a one-vs-rest logistic regression as the classifier and tune the parameters if specified.

The configuration file for this target is `classification.json`, which contains the following items:

- `data`: the path to the input data (relative to the root)
- `baseline`: a boolean indicator to specify running baseline (true-like) or parameter tuning (false-like)

### `clustering` target

Run the clustering pipeline. This target will pick representative sentences based on average sublinear TF-IDF score on the quality phrases, calculate document embeddings by average the Sentence-BERT embeddings of the representative sentences, and visualize the clusters.

The configuration file for this target is `clustering.json`, which contains the following items:

- `clu_num_workers`: the number of workers to use
- `clu_rep_sentences_path`: the path to the checkpoint representative sentences file (relative to the root), or an empty string `""` to disable the checkpoint
- `clu_doc_embeddings_path`: the path to the checkpoint document embeddings file (relative to the root), or an empty string `""` to disable the checkpoint
- `clu_dim_reduction`: the dimensionality reduction method to apply on the document embeddings, choose one from `{"PCA", "TSNE"}`

## Contributors

- Daniel Lee
- Huilai Miao
- Yuxuan Fan

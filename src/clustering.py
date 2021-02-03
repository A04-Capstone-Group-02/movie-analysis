import spacy
import numpy as np
import pandas as pd
from tqdm import tqdm
from pandarallel import pandarallel
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

spacy_en = spacy.load("en_core_web_sm")


def pick_rep_sentences(row, n=5):
    """
    Pick sentences with highest average sublinear word tf-idfs
    as the representatives of the full text.
    """
    summary, phrases = row.summary, row.phrases

    spacy_doc = spacy_en(summary)
    sents = np.array([s.text for s in spacy_doc.sents])
    if len(sents) <= n:
        return sents
    if not phrases:
        return np.random.choice(sents, n, replace=False)

    cntv = CountVectorizer(vocabulary=phrases)
    tfidf = TfidfTransformer(sublinear_tf=True)
    ranks = (
        tfidf.fit_transform(cntv.fit_transform(sents)).toarray().sum(axis=1)
    )
    ranks /= np.array([len(s) for s in spacy_doc.sents])
    return sents[np.argpartition(-ranks, n)[:n]]


def embed_doc(rep_sents, model):
    """
    Acquire document embedding by averaging the embeddings
    of representative sentences of the document.
    """
    rep_embed = np.apply_along_axis(model.encode, 0, rep_sents)
    return np.mean(rep_embed, axis=0)


def calc_all_embeddings(df, config):
    """Calculate all document embeddings."""
    # setup
    num_workers = config.get("clu_num_workers", 1)
    rep_sents_path = config.get("clu_rep_sentences_path", "")
    doc_embs_path = config.get("clu_doc_embeddings_path", "")
    sbert = SentenceTransformer("distilbert-base-nli-mean-tokens")

    # read data
    df = pd.read_pickle("../data/out/data.pkl")

    # pick representative sentences from documents
    if rep_sents_path != "":
        rep_sents = np.load(rep_sents_path, allow_pickle=True)
    elif num_workers == 1:
        rep_sents = df.apply(pick_rep_sentences, axis=1).to_numpy()
    else:
        pandarallel.initialize(nb_workers=num_workers)
        rep_sents = df.parallel_apply(pick_rep_sentences, axis=1).to_numpy()
    np.save("../data/temp/rep_sents.npy", rep_sents, allow_pickle=True)

    # calculate document embeddings
    if doc_embs_path != "":
        doc_embs = np.load(doc_embs_path, allow_pickle=False)
    else:
        rs_tqdm = tqdm(rep_sents, position=0)
        doc_embs = np.array([embed_doc(rs, sbert) for rs in rs_tqdm])
    np.save("../data/temp/doc_embs.npy", doc_embs, allow_pickle=False)


class SimilarityQueryHandler:
    """Handle cosine similarity queries given document embeddings."""

    def __init__(self, embs):
        """Constructor."""
        # euclidean distance on unit vectors 
        # is equivalent to cosine similarity
        self.embs = normalize(embs)
        self.nn = NearestNeighbors().fit(self.embs)

    def most_similar(self, idx, n):
        """Find indices of n most similar movies given a movie's index."""
        emb = [self.embs[idx]]
        return self.nn.kneighbors(emb, n, return_distance=False)[0]

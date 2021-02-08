import spacy
import numpy as np
import pandas as pd
from tqdm import tqdm
from pandarallel import pandarallel
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
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


def calc_all_embeddings(df, model, rep_sents_path, doc_embs_path, num_workers):
    """Calculate all document embeddings."""

    # pick representative sentences from documents
    print("[clustering] choosing representative sentences...")
    if rep_sents_path != "":
        rep_sents = np.load(rep_sents_path, allow_pickle=True)
    elif num_workers == 1:
        rep_sents = df.apply(pick_rep_sentences, axis=1).to_numpy()
    else:
        pandarallel.initialize(nb_workers=num_workers)
        rep_sents = df.parallel_apply(pick_rep_sentences, axis=1).to_numpy()
    np.save("data/temp/rep_sents.npy", rep_sents, allow_pickle=True)

    # calculate document embeddings
    print("[clustering] calculating document embeddings...")
    if doc_embs_path != "":
        doc_embs = np.load(doc_embs_path, allow_pickle=False)
    else:
        rs_tqdm = tqdm(rep_sents, position=0)
        doc_embs = np.array([embed_doc(rs, model) for rs in rs_tqdm])
    np.save("data/temp/doc_embs.npy", doc_embs, allow_pickle=False)
    return doc_embs


def dimensionality_reduction(embs, method="PCA"):
    dr = PCA(n_components=2)
    if isinstance(method, str) and method.upper() == "TSNE":
        dr = TSNE(n_components=2, n_jobs=12, verbose=1)
    print(f"[clustering] running {dr.__class__.__name__}...")
    dr_embs = dr.fit_transform(embs)
    np.save("data/temp/dr_embs.npy", dr_embs, allow_pickle=False)


def run_clustering(config):
    # read config
    num_workers = config.get("clu_num_workers", 1)
    rep_sents_path = config.get("clu_rep_sentences_path", "")
    doc_embs_path = config.get("clu_doc_embeddings_path", "")
    dim_reduction_method = config.get("clu_dim_reduction", "PCA")

    # setup
    print("[clustering] setting up...")
    df = pd.read_pickle("data/out/data.pkl")
    sbert = SentenceTransformer("distilbert-base-nli-mean-tokens")

    # document embedding
    doc_embs = calc_all_embeddings(
        df, sbert, rep_sents_path, doc_embs_path, num_workers
    )

    # run dim reduction
    dimensionality_reduction(doc_embs, dim_reduction_method)


class SimilarityQueryHandler:
    """Handle cosine similarity queries given document embeddings."""

    def __init__(self, embs):
        """Constructor."""
        # euclidean distance on unit vectors ~ cosine similarity
        self.embs = normalize(embs)
        self.nn = NearestNeighbors().fit(self.embs)

    def most_similar(self, idx, n):
        """Find indices of n most similar movies given a movie's index."""
        emb = [self.embs[idx]]
        return self.nn.kneighbors(emb, n, return_distance=False)[0]

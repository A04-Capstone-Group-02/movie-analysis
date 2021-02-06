import pandas as pd
import numpy as np
import nltk
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

def summary_preprocessing(plot):
    """preprocessing the movie plot summary through tokenization and removing stopwords"""
    #remove punctuation
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    no_punct = plot.apply(lambda x: tokenizer.tokenize(x))
    
    #remove stopwords
    nltk.download('stopwords')
    stop_words = set(nltk.corpus.stopwords.words('english'))
    no_stopwords = no_punct.apply(lambda x: ' '.join([w for w in x if w not in stop_words]))
    
    return no_stopwords



def model(summary,label):
    """classify movie genres"""
    #tf-idf vectorizer

    xtrain, xval, ytrain, yval = train_test_split(summary, label, test_size=0.2, random_state=9)
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)
    xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
    xval_tfidf = tfidf_vectorizer.transform(xval)
    
    #OneVsRest classifier
    lr = sklearn.linear_model.LogisticRegression()
    clf = OneVsRestClassifier(lr,n_jobs = 8)
#     scores = cross_validate(clf, xtrain_tfidf, ytrain, scoring=scoring)
    clf.fit(xtrain_tfidf,ytrain)
    y_pred = clf.predict(xval_tfidf)
    
    
    #evaluation
    f1 = sklearn.metrics.f1_score(yval, y_pred, average="micro")
    return multilabel_binarizer.inverse_transform(yval), multilabel_binarizer.inverse_transform(y_pred)

def tunning_params(summary,label,genres):
    """parameter tuned classify models""""
    #tf-idf vectorizer
    multilabel_binarizer = sklearn.preprocessing.MultiLabelBinarizer()
    multilabel_binarizer.fit(genres)
    label = multilabel_binarizer.transform(genres)
    xtrain, xval, ytrain, yval = train_test_split(summary, label, test_size=0.2, random_state=9)
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)
    xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
    xval_tfidf = tfidf_vectorizer.transform(xval)
    
    #OneVsRest classifier
    

    parameters = {
        "estimator__penalty": ['l1'],
        "estimator__solver":['newton-cg','lbfgs', 'liblinear', 'sag', 'saga'],
        "estimator__solver":[ 'liblinear', 'lbfgs'],
        "estimator__C":[1,3],
    }
    lr = sklearn.linear_model.LogisticRegression()
    clf = OneVsRestClassifier(lr,n_jobs = 8)
    model_tunning = GridSearchCV(clf, param_grid=parameters,cv = 3, verbose = 3, scoring = 'f1_micro',refit = True)
    model_tunning.fit(xtrain_tfidf,ytrain)
    y_pred = model_tunning.best_estimator_.predict(xval_tfidf)
    
    f1 = sklearn.metrics.f1_score(yval, y_pred, average="micro")
    return model_tunning.best_score_,model_tunning.best_params_, model_tunning.best_estimator_,f1
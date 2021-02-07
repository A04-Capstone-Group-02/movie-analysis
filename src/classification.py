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
import warnings
warnings.filterwarnings("ignore")



def baseline_model(plot_summary,genre):

    """preprocessing, vectorization, and classifcation with OneVsRestClassifier and Logistic Regression"""
    #remove punctuation
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    no_punct = plot_summary.apply(lambda x: tokenizer.tokenize(x))
    
    #remove stopwords
    nltk.download('stopwords')
    stop_words = set(nltk.corpus.stopwords.words('english'))
    summary = no_punct.apply(lambda x: ' '.join([w for w in x if w not in stop_words]))
    
    #label binarizer
    multilabel_binarizer = sklearn.preprocessing.MultiLabelBinarizer()
    multilabel_binarizer.fit(genre)
    label = multilabel_binarizer.transform(genre)
   
    #split training and validation set
    xtrain, xval, ytrain, yval = train_test_split(summary, label, test_size=0.2, random_state=9)
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)
    xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
    xval_tfidf = tfidf_vectorizer.transform(xval)
    
    #OneVsRest classifier
    lr = sklearn.linear_model.LogisticRegression(solver = 'liblinear')
    clf = OneVsRestClassifier(lr,n_jobs = 8)
#     scores = cross_validate(clf, xtrain_tfidf, ytrain, scoring=scoring)
    clf.fit(xtrain_tfidf,ytrain)
    y_pred = clf.predict(xval_tfidf)
    
    # Predicted label
    actual_genre = multilabel_binarizer.inverse_transform(yval)
    predicted_genre = multilabel_binarizer.inverse_transform(y_pred)
    
    
    #evaluation
    f1 = sklearn.metrics.f1_score(yval, y_pred, average="micro")
    
    e1 = 'percentage of genres that are correctly predicted: '+ str(np.sum([len(set(a).intersection(b)) for a, b in \
                  zip(pd.Series(predicted_genre), pd.Series(actual_genre))])/sum(genre.apply(len)))
    e2 = 'percentage of movies that have at least one gnere predicted right: '+str(np.sum([len(set(a).intersection(b))>0 for a, b in\
                  zip(pd.Series(predicted_genre), pd.Series(actual_genre))])/len(genre))
    return f1,e1,e2

def tunning_params(summary,genre):
    """parameter tuned classify models"""
    #label binarizer
    multilabel_binarizer = sklearn.preprocessing.MultiLabelBinarizer()
    multilabel_binarizer.fit(genre)
    label = multilabel_binarizer.transform(genre)
   
    #split training and validation set
    xtrain, xval, ytrain, yval = train_test_split(summary, label, test_size=0.2, random_state=9)
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)
    xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
    xval_tfidf = tfidf_vectorizer.transform(xval)
    
  
    #hyperparameter grid search
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
    return model_tunning.best_estimator_,f1

def model(config):
    pikl = pd.read_pickle(config['data'])
    df = pikl.dropna(subset = ['genres'])
    summary = df['summary']
    genre = df['genres']
    phrase = df['phrases'].apply(lambda x: ' '.join(x))
    baseline = True
    if baseline:
        s = baseline_model(summary,genre)
        p = baseline_model(phrase,genre)
    else:
        s = tunning_params(summary_genre)
        p = tunning_params(summary_genre)
    print('model performance using movie plot summary: '+ str(s)+'\n' + 'model performance using phrases: '+str(p))
        
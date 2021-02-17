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


# def preprocessing(plot_summary):
#     #remove punctuation
#     tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
#     no_punct = plot_summary.apply(lambda x: tokenizer.tokenize(x))
#     #remove stopwords
#     nltk.download('stopwords')
#     stop_words = set(nltk.corpus.stopwords.words('english'))
#     summary = no_punct.apply(lambda x: ' '.join([w for w in x if w not in stop_words]))
#     return summary
    
# def binarizer(label):
#     #label binarizer
#     multilabel_binarizer = sklearn.preprocessing.MultiLabelBinarizer()
#     multilabel_binarizer.fit(label)
#     label = multilabel_binarizer.transform(label)
#     return multilabel_binarizer,label

# def transform_and_build(data, max_df):
#     xtrain, xval, ytrain, yval = train_test_split(summary, label, test_size=0.2, random_state=9)
#     tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)
#     xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
#     xval_tfidf = tfidf_vectorizer.transform(xval)
#     #OneVsRest classifier
#     lr = sklearn.linear_model.LogisticRegression(solver = 'liblinear')
#     clf = OneVsRestClassifier(lr,n_jobs = 8)
# #     scores = cross_validate(clf, xtrain_tfidf, ytrain, scoring=scoring)
#     clf.fit(xtrain_tfidf,ytrain)
#     y_pred = clf.predict(xval_tfidf)
    
# #     # Predicted label
# #     actual_genre = multilabel_binarizer.inverse_transform(yval)
# #     predicted_genre = multilabel_binarizer.inverse_transform(y_pred)
#     return y_pred
    

# def baseline_model(plot_summary,genre):

#     """preprocessing, vectorization, and classifcation with OneVsRestClassifier and Logistic Regression"""
    #remove punctuation
#     tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
#     no_punct = plot_summary.apply(lambda x: tokenizer.tokenize(x))
    
#     #remove stopwords
#     nltk.download('stopwords')
#     stop_words = set(nltk.corpus.stopwords.words('english'))
#     summary = no_punct.apply(lambda x: ' '.join([w for w in x if w not in stop_words]))
    
#     label binarizer
#     multilabel_binarizer = sklearn.preprocessing.MultiLabelBinarizer()
#     multilabel_binarizer.fit(genre)
#     label = multilabel_binarizer.transform(genre)
   
#     split training and validation set
#     xtrain, xval, ytrain, yval = train_test_split(summary, label, test_size=0.2, random_state=9)
#     tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)
#     xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
#     xval_tfidf = tfidf_vectorizer.transform(xval)
    
#     OneVsRest classifier
#     lr = sklearn.linear_model.LogisticRegression(solver = 'liblinear')
#     clf = OneVsRestClassifier(lr,n_jobs = 8)
# #     scores = cross_validate(clf, xtrain_tfidf, ytrain, scoring=scoring)
#     clf.fit(xtrain_tfidf,ytrain)
#     y_pred = clf.predict(xval_tfidf)
    
#     # Predicted label
#     actual_genre = multilabel_binarizer.inverse_transform(yval)
#     predicted_genre = multilabel_binarizer.inverse_transform(y_pred)
    
    
#     #evaluation
#     f1 = "f1-score: "+str(sklearn.metrics.f1_score(yval, y_pred, average="micro"))
    
#     e1 = 'percentage of genres that are correctly predicted: '+ str(np.sum([len(set(a).intersection(b)) for a, b in \
#                   zip(pd.Series(predicted_genre), pd.Series(actual_genre))])/sum(genre.apply(len)))
#     e2 = 'percentage of movies that have at least one gnere predicted right: '+str(np.sum([len(set(a).intersection(b))>0 for a, b in\
#                   zip(pd.Series(predicted_genre), pd.Series(actual_genre))])/len(genre))
#     return (f1+"\n"+e1+"\n"+e2+"\n")

def build(summary,genre, text_feature, baseline = 1,top_genre = 10,top_phrases = 10):
    """parameter tuned classify models"""
    #remove punctuation
    if text_feature:
        tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        no_punct = summary.apply(lambda x: tokenizer.tokenize(x))

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
    
  
    #hyperparameter grid search
    parameters = {
        "estimator__penalty": ['l1'],
        "estimator__solver":['newton-cg','lbfgs', 'liblinear', 'sag', 'saga'],
        "estimator__solver":[ 'liblinear', 'lbfgs'],
        "estimator__C":[1,3],
    }
    lr = sklearn.linear_model.LogisticRegression()
    clf = OneVsRestClassifier(lr,n_jobs = 8)
    if baseline:
        clf.fit(xtrain_tfidf,ytrain)
        y_pred = clf.predict(xval_tfidf)
    else:
        clf = GridSearchCV(clf, param_grid=parameters,cv = 3, verbose = 3, scoring = 'f1_micro',refit = True)
        clf.fit(xtrain_tfidf,ytrain)
        y_pred = clf.predict(xval_tfidf)

    # Predicted label
    actual_genre = multilabel_binarizer.inverse_transform(yval)
    predicted_genre = multilabel_binarizer.inverse_transform(y_pred)
    
    
    #evaluation
    f1 = "f1-score: "+str(sklearn.metrics.f1_score(yval, y_pred, average="micro"))
    
    e1 = 'percentage of genres that are correctly predicted: '+ str(np.sum([len(set(a).intersection(b)) for a, b in \
                  zip(pd.Series(predicted_genre), pd.Series(actual_genre))])/sum(genre.apply(len)))
    e2 = 'percentage of movies that have at least one gnere predicted right: '+str(np.sum([len(set(a).intersection(b))>0 for a, b in\
                  zip(pd.Series(predicted_genre), pd.Series(actual_genre))])/len(genre))
    
    print("========model interpretation========")
    lst = []
    new_genre_label = []
    genre_label = multilabel_binarizer.classes_
    for a,b in zip(clf.estimators_, genre_label):
        try:
            lst.append(a.coef_)
            new_genre_label.append(b)
        except:
            pass

    dist = genre.explode().value_counts(ascending = False)
    genre_coef = dict(zip(new_genre_label,np.vstack(lst)))
    for g in dist[top_genre:].index:
        c = genre_coef[g]
        words = tfidf_vectorizer.inverse_transform(c)[0]
        evd = [t for t in c if t >0]
        d = dict(zip(words,evd))
        sorted_words = sorted(d.items(), key=lambda item: item[1])[-top_phrases:]
        x = [i[0] for i in sorted_words]
        y = [i[1] for i in sorted_words]
#         plt.figure()
#         plt.barh(x,y,title = genre)
        print(list(zip(x,y)))
    return (f1+"\n"+e1+"\n"+e2+"\n")

def model(config):
    # read dataset path from config
    pikl = pd.read_pickle(config['data'])
    # read if run on baseline model from config
    baseline = config['baseline']
    # drop movie entries that do not have genres
    df = pikl.dropna(subset = ['genres'])
    ngenre = config['top_genre']
    nphrase = config['top_phrase']
    
    summary = df['summary']
    genre = df['genres']
    phrase = df['phrases'].apply(lambda x: ' '.join(x))
    

    s = build(summary,genre,1,baseline,ngenre,nphrase) # 1 means using text summary as feature
    p = build(phrase,genre,0,baseline,ngenre,nphrase)# 0 means using phrases as feature 

    print("=============Results=============")
    print('model performance using movie plot summary: '+ s +'\n' )
    print('model performance using phrases: '+p)
      
        
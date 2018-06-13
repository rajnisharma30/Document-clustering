# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 10:59:37 2017

@author: CG, RM, RS, SK
"""
from __future__ import print_function
import numpy as np
import pandas as pd
import nltk
import re
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn import metrics
#from sklearn import metrics


# Basic Declarations Begin
file_path="records.txt"
data=np.loadtxt(file_path, dtype="string",delimiter='\t') # Load complete file in data
titles=data[:,0]    # Seperate titles list
synopsis=data[:,1]  #Seperate synopsis list
rating=np.array(data[:,2],dtype='float')
stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer("english")
punctuation=[',','.','!','\'','\"','#','$','&',';','?','\'\'','``','\'s','\'am','\'re']
#Basic Declarations End

def tokenize_stem(text):
    tokens = [nltk.word_tokenize(i) for i in text]
    tokens_only=[]
    for i in tokens:
        tokens_only.extend([word.lower() for word in i if re.search('[a-zA-Z]',word)  if word.lower() not in stopwords and word.lower() not in punctuation])
    tokens_stemmed = [stemmer.stem(t) for t in tokens_only]
    return tokens_only,tokens_stemmed

    
def tokenize_stem_re(text):
    tokens = [nltk.word_tokenize(text)]
    tokens_only=[]
    for i in tokens:
        tokens_only.extend([word.lower() for word in i if re.search('[a-zA-Z]',word.lower())  if word.lower() not in stopwords and word.lower() not in punctuation])
    tokens_stemmed = [stemmer.stem(t) for t in tokens_only]
    return tokens_stemmed
    
tokens_only,tokens_stemmed=tokenize_stem(synopsis)
word_frame = pd.DataFrame({'words': tokens_only}, index = tokens_stemmed)
tfidf_vectorizer = TfidfVectorizer(use_idf=True,tokenizer=tokenize_stem_re,stop_words='english',analyzer='word',min_df=0.05)
tfidf_matrix=tfidf_vectorizer.fit_transform(synopsis)   
tfidf_vectorizer.get_feature_names()
terms = tfidf_vectorizer.get_feature_names()
dist = 1 - cosine_similarity(tfidf_matrix)
num_clusters = 5
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()
films = { 'title': titles,'rating': rating, 'synopsis': synopsis, 'cluster': clusters}
frame = pd.DataFrame(films, index = [clusters] , columns = ['title','rating','synopsis','cluster'])
frame['cluster'].value_counts()
grouped = frame['rating'].groupby(frame['cluster'])
print(grouped.mean())
print("Top terms per cluster:")
print()
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')
    
    for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
        print(' %s' % word_frame.ix[terms[ind].split(' ')].values.tolist()[0][0], end=',')
    print() #add whitespace
    print() #add whitespace
    
    print("Cluster %d titles:" % i, end='')
    for title in frame.ix[i]['title'].values.tolist():
        print(' %s,' % title, end='')
    print() #add whitespace
    print() #add whitespace
    
print()
print()
print("Homogeneity: %0.3f" % metrics.homogeneity_score(data[:,1], km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(data[:,1], km.labels_))

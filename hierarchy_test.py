# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 10:59:37 2017

@author: Chinmay Gadkari
"""
from __future__ import print_function
import numpy as np
import pandas as pd
import nltk
import re
from nltk.stem.snowball import SnowballStemmer
from tokenize import tokenize,STRING
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import string
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
import urllib2
from bs4 import BeautifulSoup
from scipy.cluster.hierarchy import ward, dendrogram
import matplotlib.pyplot as plt

#text scrapping code for IMDb webpage
url='http://www.imdb.com/list/ls055592025/'
page = urllib2.urlopen(url)
soup = BeautifulSoup(page, 'html.parser')
#Fetch movie titles
div_info=soup.select('div.info b a')
titles=[i.text.strip() for i in div_info]
#Fetch titles ends
#Fetch movie descriptions
item_description=soup.find_all('div',attrs={'class':'item_description'})
for i in item_description:
    i.span.decompose() 
description=[i.text.strip() for i in item_description ]
#fetch movie descriptions ends
#fetch ratings
rate=soup.find_all('span',attrs={'class':'value'})
ratings=[i.text.strip() for i in rate]
#fetch ratings ends
#Text scrapping ends
#zipping and arraying
data=zip(titles,description,ratings)
data=np.array(data,dtype='string')
#zipping and arraying ends

# Basic Declarations Begin
#file_path="C:\\Users\HP\Desktop\Movie_Synopsis.txt"
#data=np.loadtxt(file_path, dtype="string",delimiter='\t') # Load complete file in data
titles=data[:,0]    # Seperate titles list
synopsis=data[:,1]  #Seperate synopsis list
ratings=np.array(data[:,2],dtype='float')
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
tfidf_vectorizer = TfidfVectorizer(use_idf=True,tokenizer=tokenize_stem_re,stop_words='english',analyzer='word')
tfidf_matrix=tfidf_vectorizer.fit_transform(synopsis)   
tfidf_vectorizer.get_feature_names()
terms = tfidf_vectorizer.get_feature_names()
dist = 1 - cosine_similarity(tfidf_matrix)    
linkage_matrix = ward(dist)
fig, ax = plt.subplots(figsize=(15, 20))
ax = dendrogram(linkage_matrix, orientation="right", labels=titles);
plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')
plt.tight_layout()
#print("Homogeneity: %0.3f" % metrics.homogeneity_score(data[:,0], km.labels_))
#print("Completeness: %0.3f" % metrics.completeness_score(data[:,0], km.labels_))

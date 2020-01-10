#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 09:52:29 2020

@author: kevin
"""

import pyLDAvis
import pandas as pd
import pyLDAvis.sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
data = pd.read_csv("cleaned_small_email.csv")

#data_vis = pyLDAvis.prepare(**data)

tf_vectorizer = CountVectorizer(max_df = 0.5, 
                                min_df = 10)
dtm_tf = tf_vectorizer.fit_transform(data.message.dropna())
#print(dtm_tf.shape)

tfidf_vectorizer = TfidfVectorizer(**tf_vectorizer.get_params())
dtm_tfidf = tfidf_vectorizer.fit_transform(data.message.dropna())
#print(dtm_tfidf.shape)

# # for TF DTM
# lda_tf = LatentDirichletAllocation(n_components=20, random_state=0)
# lda_tf.fit(dtm_tf)
# for TFIDF DTM
lda_tfidf = LatentDirichletAllocation(n_components=20, random_state=0)
lda_tfidf.fit(dtm_tfidf)

pyLDAvis.show(pyLDAvis.sklearn.prepare(lda_tfidf, dtm_tf, tf_vectorizer))
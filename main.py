#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 11:42:37 2019

@author: kevin
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_dangerously_set_inner_html
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set()
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer  

import pyLDAvis
import pyLDAvis.sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
#from string import punctuation
#import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
import plotly.graph_objs as go
import time
#nltk.download('stopwords')
#nltk.download('wordnet')
stopwords = stopwords.words("english")
stopwords.append(["","cc"])
lemmatizer = WordNetLemmatizer()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


def vectorize_messages(messages, min_df=10, max_df = 0.6):
    vectorizer = TfidfVectorizer(min_df=min_df, max_df = max_df )
    wordvector_fit = vectorizer.fit_transform(messages)
    feature_names = vectorizer.get_feature_names()
    #dense = wordvector_fit.todense()
    return wordvector_fit, feature_names

def do_clustering (wordvector_fit, num_cluster = 4) :
    
    clf = MiniBatchKMeans(n_clusters=num_cluster, 
                max_iter=50, 
                init='k-means++',
                n_init=4)
    labels = clf.fit_predict(wordvector_fit)
    
    return clf, labels
    
def print_2d_clusters (wordvector_fit, clf, labels):
    wordvector_fit_2d = wordvector_fit.todense()
    pca = PCA(n_components=2).fit(wordvector_fit_2d)
    datapoint = pca.transform(wordvector_fit_2d)
    
    label = ["#e05f14", "#e0dc14", "#2fe014", "#14d2e0"]
    color = [label[i] for i in labels]
    plt.scatter(datapoint[:, 0], datapoint[:, 1],c  = color)
    
    centroids = clf.cluster_centers_
    centroidpoint = pca.transform(centroids)
    plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1], marker='^', s=150, c='#000000')
    plt.show()


file_name = "cleaned_small_email.csv"
data = pd.read_csv(file_name)
data = data.dropna()
# wordvector_fit, feature_names = vectorize_messages(data.message)
# clf, labels = do_clustering(wordvector_fit)

tf_vectorizer = CountVectorizer(max_df = 0.5, 
                                min_df = 10)

tfidf_vectorizer = TfidfVectorizer(**tf_vectorizer.get_params())
dtm_tfidf = tfidf_vectorizer.fit_transform(data.message)
lda_tfidf = LatentDirichletAllocation(n_components=20, random_state=0)
lda_tfidf.fit(dtm_tfidf)




app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True
app.layout = html.Div([dcc.Location(id='url', refresh=False),
              html.Div(id="core",className = "container-fluid", style={'width': '70%', 'display': 'inline-block', 'padding' : {'left' : 10000}})])

    
hex_col = ["#e05f14", "#e0dc14", "#2fe014", "#14d2e0", "#d11141", "#00b159",  "#00aedb" ,"#f37735"," #ffc425", "#d3a625"]

    
page_kmean_layout = [
                    html.Div([
                        html.H1("Projet Fil Rouge - IBM"),
                
                        html.Div([
                            html.H4("Number of clusters"),
                            dcc.Dropdown(
                                id='nb_clusters',
                                options=[{'label': i, 'value': i} for i in range(1,11)],
                                value=4
                            ),
                            html.H4("Occurence minimum"),
                            dcc.Dropdown(
                                id='min_df',
                                options=[{'label': i, 'value': i} for i in [1,5,10,15,20,25,30,35,40,45,50] ],
                                value=10
                            ),
                                                
                        ],
                        style={'width': '48%', 'display': 'inline-block'}),
                
                        html.Div([
                            html.H4("Chunksize of data"),
                            dcc.Dropdown(
                                id='chunksize',
                                options=[{'label': i, 'value': i} for i in [50,500,5000,50000]],
                                value=500
                            ),
                            html.H4("Occurence maximum"),
                            dcc.Dropdown(
                                id='max_df',
                                options=[{'label': i, 'value': i} for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]],
                                value=0.6
                            ),
                            
                        ],style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
                    ]),
                dcc.Graph(id = "my-graph")]

@app.callback(
    dash.dependencies.Output('my-graph', 'figure'),
    [dash.dependencies.Input('chunksize', 'value'),
     dash.dependencies.Input('nb_clusters', 'value'),
     dash.dependencies.Input('min_df', 'value'),
     dash.dependencies.Input('max_df', 'value'),
     dash.dependencies.Input('url', 'pathname')])
def update_figure(chunksize, nb_clusters, min_df, max_df, pathname):

        before = time.time()
        wordvector_fit, feature_names = vectorize_messages(data.message, min_df= min_df, max_df= max_df)
        after3 = time.time()
        print("time d execution tfidf :", after3-before)
        clf, labels = do_clustering(wordvector_fit, num_cluster= nb_clusters)
        
        colors = [hex_col[i] for i in np.unique(labels)]
        after = time.time()
        print("time d execution  kmean:", after-after3)
        wordvector_fit_2d = wordvector_fit.todense()
        pca = PCA(n_components=2).fit(wordvector_fit_2d)
        datapoint = pca.transform(wordvector_fit_2d)
        centroids = clf.cluster_centers_
        centroidpoint = pca.transform(centroids)
        
        data_to_print = []
        after1 = time.time()
        print("time d execution pca:", after1-after)
        for i in range (len(np.unique(labels))) : 
            data_label = datapoint[labels == i]
            
            # data_to_print.append(go.Scatter(x = data_label[:, 0], y = data_label[:, 1], mode = "markers", 
            #                          name = f"Cluster {i}"
            #                          ))
            word_cluster = sum(wordvector_fit[labels==i].todense())
            common_words = sorted(zip(np.array(word_cluster)[0], feature_names), reverse=True)
            hovertext = ""
            for el in common_words[:10] :
                hovertext +=el[1] + "\n"
                
            data_to_print.append(go.Scatter(
                x = [centroidpoint[i, 0]],
                y = [centroidpoint[i, 1]],
                mode = "markers",
                marker_size = 300*len(data_label)/len(labels),
                name = f"Cluster {i}",
                hovertext = hovertext,
                hoverinfo="text",
                opacity = 0.7
            ))
           
        after2 = time.time()
        print("time d execution fill data_to_print:", after2-after1)
        # trace1 = go.Scatter(x=[1, 2, 3, 4],
        #                     y=[10, 11, 12, 13],
        #                     mode='markers',
        #                     marker=dict(size=[40, 60, 80, 100],color=[0, 1, 2, 3]))
        
        #data = [trace1]
        after = time.time()
        print("time d execution :", after-before)
        return {"data": data_to_print,
                'layout': dict(
                margin={'l': 40, 'b': 40, 't': 100, 'r': 10},
                legend={'x': 1, 'y': 1},
                hovermode='closest',
                transition = {'duration': 500},
                title = "Cluster from enron dataset"
            )}
 

@app.callback(dash.dependencies.Output('core', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    print(pathname)
    if "kmean" in pathname : 
        return page_kmean_layout
    elif "lda" in pathname : 
        
        #data_prepared = pyLDAvis.sklearn.prepare(lda_tfidf, dtm_tfidf, tfidf_vectorizer)
        #, d3_url="js/d3.v3.min.js", ldavis_url="js/ldavis.js", ldavis_css_url="js/ldavis.css"
        #html = pyLDAvis.prepared_data_to_html(data_prepared)
        #print(html)
        pyLDAvis.show(pyLDAvis.sklearn.prepare(lda_tfidf, dtm_tfidf, tfidf_vectorizer))
        return html.A("LDA visualisation on other tab for now")
    
if __name__ == '__main__':
    
    #print_2d_clusters(wordvector_fit, clf)
    
    app.run_server(host='0.0.0.0',debug=False, port = 8050)
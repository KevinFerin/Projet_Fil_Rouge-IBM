#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 17:43:58 2019

@author: kevin
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set()
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer  
from string import punctuation
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.graph_objs as go

nltk.download('stopwords')
nltk.download('wordnet')
stopwords = stopwords.words("english")
stopwords.append(["","cc"])
lemmatizer = WordNetLemmatizer()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
def get_cleaned_data (file_name, chunksize=500) :
        
    pd.options.mode.chained_assignment = None
    
    chunk = pd.read_csv(file_name, chunksize=chunksize)
    data = next(chunk)
    
    data.info()
    data["Date"] = data.message.str.split("\n").str[1].str.split("Date: ").str.join('')
    data.Date = pd.to_datetime(data.Date)
    data["From"] = data.message.str.split("\n").str[2].str.split("From: ").str.join('')
    data["To"] = data.message.str.split("\n").str[3].str.split("To: ").str.join('')
    data["Subject"] = data.message.str.split("\n").str[4].str.split("Subject: ").str.join('')
    data.message = data.message.str.split("\n").str[15:].str.join('\n').str.lower()
    for i,message in enumerate(data.message) :
        if "Forwarded by" in message : 
            data.message[i] = "\n".join(message.split("\n")[9:]).strip()
            
    data.message = data.message.str.replace("\n", ' ').str.strip()
    
    return data

def remove_punctuation(data) : 
    for punc in punctuation :
        data.message = data.message.str.replace("\\d", '')
        data.message = data.message.str.replace(punc,'')
    return data

def tokenize(data) : 
    data.message = data.message.str.split(" ")
    return data

def remove_stop_words (data) : 
    for i,message in enumerate(data.message) :
        
        data["message"][i] = [word for word in message if word not in stopwords]
        
    return data


def lem_words (data) :
    for i,message in enumerate(data.message) :
        for k, word in enumerate(message) :
            message[k] = lemmatizer.lemmatize(word)
        data["message"][i] = message
    return data

def get_back_to_text(data) : 
    data.message = data.message.str.join(" ")
    return data 


def vectorize_messages(messages, min_df=10, max_df = 0.6):
    vectorizer = TfidfVectorizer(min_df=min_df, max_df = max_df )
    wordvector_fit = vectorizer.fit_transform(messages)
    feature_names = vectorizer.get_feature_names()
    #dense = wordvector_fit.todense()
    return wordvector_fit, feature_names

def do_clustering (wordvector_fit, num_cluster = 4) :

    clf = KMeans(n_clusters=num_cluster, 
                max_iter=50, 
                init='k-means++',
                n_init=4)
    labels = clf.fit_predict(wordvector_fit)
    
    return clf, labels
    
def print_2d_clusters (wordvector_fit, clf):
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
    
    
file_name = "small_email.csv"
data = get_cleaned_data(file_name)
data = remove_punctuation(data)
data = tokenize(data)
data = remove_stop_words(data)
data = lem_words(data)
data = get_back_to_text(data)
wordvector_fit, feature_names = vectorize_messages(data.message)
clf, labels = do_clustering(wordvector_fit)

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
                    html.Div([
                        html.H1("Projet Fil Rouge IBM, un café ? "),
                
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
                    
                    dcc.Graph(id = "my-graph"),
                    
                    ],className = "container-fluid", style={'width': '70%', 'display': 'inline-block', 'padding' : {'left' : 10000}})

    
hex_col = ["#e05f14", "#e0dc14", "#2fe014", "#14d2e0", "#d11141", "#00b159",  "#00aedb" ,"#f37735"," #ffc425", "#d3a625"]

@app.callback(
    dash.dependencies.Output('my-graph', 'figure'),
    [dash.dependencies.Input('chunksize', 'value'),
     dash.dependencies.Input('nb_clusters', 'value'),
     dash.dependencies.Input('min_df', 'value'),
     dash.dependencies.Input('max_df', 'value')])
def update_figure(chunksize, nb_clusters, min_df, max_df):
    file_name = "small_email.csv"
    data = get_cleaned_data(file_name, chunksize=chunksize)
    data = remove_punctuation(data)
    data = tokenize(data)
    data = remove_stop_words(data)
    data = lem_words(data)
    data = get_back_to_text(data)
    wordvector_fit, feature_names = vectorize_messages(data.message, min_df= min_df, max_df= max_df)
    clf, labels = do_clustering(wordvector_fit, num_cluster= nb_clusters)
    
    colors = [hex_col[i] for i in np.unique(labels)]
    
    wordvector_fit_2d = wordvector_fit.todense()
    pca = PCA(n_components=2).fit(wordvector_fit_2d)
    datapoint = pca.transform(wordvector_fit_2d)
    centroids = clf.cluster_centers_
    centroidpoint = pca.transform(centroids)
    
    data_to_print = []

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
            marker_size = len(data_label)//2,
            name = f"Cluster {i}",
            hovertext = hovertext,
            hoverinfo="text",
            opacity = 0.7
        ))
       
    
    # trace1 = go.Scatter(x=[1, 2, 3, 4],
    #                     y=[10, 11, 12, 13],
    #                     mode='markers',
    #                     marker=dict(size=[40, 60, 80, 100],color=[0, 1, 2, 3]))
    
    #data = [trace1]
    return {"data": data_to_print,
            'layout': dict(
            margin={'l': 40, 'b': 40, 't': 100, 'r': 10},
            legend={'x': 1, 'y': 1},
            hovermode='closest',
            transition = {'duration': 500},
            title = "Cluster from enron dataset"
        )}



    
if __name__ == '__main__':
    
    #print_2d_clusters(wordvector_fit, clf)
    
    app.run_server(host='0.0.0.0',debug=False, port = 8050)
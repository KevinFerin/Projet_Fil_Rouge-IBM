# Projet Fil Rouge - IBM : Clustering de document

Ce projet corresponds au projet fil rouge effectué dans le cadre de notre mastère spécialisé BigData à Télécom Paris en partenariat avec IBM.
L'équipe se compose de quatre membres : 
- Guillaume Lehericy
- Hicham Elouatiki
- Yang Wang
- Kevin Ferin

Le sujet de ce projet est de produire une analyse non-supervisé d'un corpus de document afin d'en extraire un certains nombres de cluster. Chacun de ses clusters doivent poséder un sens compréhensible par une personne lambda. L'objectif est, étant donné un corpus de document, d'être capable de diviser le corpus en plusieurs clusters avec un thème par cluster et plusieurs mots clés le caractérisant.

## Procédure de lancement du projet 
La création d'un environnemnet virtuel est recommandé mais pas obligatoire. 
- **Récupération du projet :** 
```
git clone https://github.com/KevinFerin/Projet_Fil_Rouge-IBM.git
cd Projet_Fil_Rouge-IBM
virtualenv venv
source venv/bin/acivate
```

- **Installation des librairies requises :** 
```
pip install -r requirements.txt
```

- **Lancement du projet :** 
```
python Dashboard.py 
```

- **Visualisation du dashboard sur votre navigateur:** 
```
localhost:8050
```

## Ce que nous faisons :

Dataset utilisé : enron dataset 

Pour l'instant notre approche est très succinte. Elle corresponds à :
- Un **nettoyage des données** pour en extraire le contenu des emails, les destinataires, les sujets. 
- Un nettoyage du contenu de chaque email afin d'éliminer les "**stopwords**", la ponctuation, éliminer les conjugaisons ("**lemmatization**")
- Une vectorisation simple corresondant à la création d'un vocabulaire et transformation de chaque email en vecteur gràce à la **vectorisation TFIDF**. Cette vectorisation prends en compte la fréquence d'occurence de chaque mot dans le texte mais aussi dans le corpus entier. 
- Un algorithme de clusterisation également simple : **K-mean**. Cet algortihme consiste à, étant donner un nombre initial de cluster, affecter au mieux les données dans K clusters en affectant chaque données au cluster le plus proche (proche au sens de la distance euclidienne) 
- Une **visualiation sous forme de dashboard** en utilisant la librairie python Dash. Cette visualisation permets de modifier les différents hyperparamètres et d'observer la répartition des données dans les clusters. (Visualisation 2D effectuer grâce à une projection PCA après le K-mean)

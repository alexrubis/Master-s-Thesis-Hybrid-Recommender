# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 07:14:05 2023

@author: Aleksander Rubis
"""
##############################################################################################
# Import bibliotek
##############################################################################################
import os
from zipfile import ZipFile

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
import umap 

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Embedding, Flatten, Input,  Dot,Dropout, Dense, BatchNormalization, Concatenate
from tensorflow.keras.utils import model_to_dot, get_file
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras.constraints import NonNeg
from IPython.display import SVG

import r_metrics
import recmetrics

##############################################################################################
### Wczytanie i edycja zbioru movielens
##############################################################################################

# # Pobieranie zbioru danych ze strony grouplens.org 
# Odkomentować jeśli zbiory mają zostać pobrane!
# ml_latest_small_url = ("http://files.grouplens.org/datasets/movielens/ml-latest-small.zip")

# ml_25m_url = ("https://files.grouplens.org/datasets/movielens/ml-25m.zip")

# cwd = os.getcwd()

# ml_latest_small_url_zipped = get_file(cwd+"\\"+"ml-latest-small.zip", ml_latest_small_url)
# ml_25m_url_zipped = get_file(cwd+"\\"+"ml-25m.zip", ml_25m_url)

# # Wypakowywanie ml_latest_small_url
# with ZipFile(ml_latest_small_url_zipped, "r") as zip:
#         # Wypakowywanie pliku
#         print("Wypakowywanie pliku...")
#         zip.extractall(path=cwd)
#         print("Done!")
# # Wypakowywanie ml_25m_url
# with ZipFile(ml_25m_url_zipped, "r") as zip:
#         # Wypakowywanie pliku
#         print("Wypakowywanie pliku...")
#         zip.extractall(path=cwd)
#         print("Done!")


ratings_df = pd.read_csv('./ml-latest-small/ratings.csv', header=0, names=['user_id', 'movie_id', 'rating', 'timestamp'])
movies_df = pd.read_csv('./ml-latest-small/movies.csv', header=0, names=['movie_id', 'title', 'genres'])
tags_df = pd.read_csv('./ml-latest-small/tags.csv', header=0, names=['user_id', 'movie_id', 'tag', 'timestamp'])

genome_scores = pd.read_csv('./ml-25m/genome-scores.csv', header=0, names=['movie_id', 'tag_id', 'relevance'])
genome_tags = pd.read_csv('./ml-25m/genome-tags.csv', header=0, names=['tag_id', 'tag'])

# Unikalne filmy w zbiorze tagów genome_scores
genome_scores_unique_movies = genome_scores.movie_id.unique()
# Unikalne filmy w zbiorze ocen
ratings_df_unique_movies = ratings_df.movie_id.unique()

# Sprawdzenie ile z filmów w zbiorze ratings_df nie posiada tagów w zbiorze genome_scores
diff = np.isin(ratings_df_unique_movies,genome_scores_unique_movies)
diff = pd.DataFrame({'movie_id': ratings_df_unique_movies, 'present': diff})
diff = diff[diff["present"] == False]
print("Liczba filmów w zbiorze ratings_df bez tagów: ", len(diff))


# Liczba unikalnych użytkowników i filmów przed usuwaniem filmów bez tagów
print("Liczba unikalnych użytkowników przed usuwaniem filmów bez tagów: ", len(ratings_df.user_id.unique()))
print("Liczba unikalnych filmów przed usuwaniem filmów bez tagów: ", len(ratings_df.movie_id.unique()))

# Usunięcie filmów ze zbioru ratings_df nieposiadających tagów w zbiorze genome_scores
ratings_df = ratings_df[ratings_df["movie_id"].isin(genome_scores_unique_movies)]
# Usunięcie filmów ze zbioru genome_scores nieobecnych w zbiorze ratings_df
genome_scores = genome_scores[genome_scores["movie_id"].isin(ratings_df.movie_id.unique())]

# Liczba unikalnych użytkowników i filmów po usuwaniu filmów bez tagów
print("Liczba unikalnych użytkowników po usuwaniu filmów bez tagów: ", len(ratings_df.user_id.unique()))
print("Liczba unikalnych filmów po usuwaniu filmów bez tagów: ", len(ratings_df.movie_id.unique()))


# # ID filmów ze zbioru MovieLens są ponumerowane w sposób utrudniający ich przetwarzanie (movie_id:1,3,6,50,101,223...),
# Zmiana numeracji movie_id
fixed_movie_id_list = list(ratings_df["movie_id"])

old_to_new_id_dict = dict()

new_index = 0
for index, movie_id in enumerate(fixed_movie_id_list):
    if old_to_new_id_dict.get(movie_id) == None:
        old_to_new_id_dict[movie_id] = new_index
        fixed_movie_id_list[index] = new_index
        new_index += 1
    else:
        fixed_movie_id_list[index] = old_to_new_id_dict[movie_id]

ratings_df["old_movie_id"] = ratings_df["movie_id"] 
ratings_df["movie_id"] = fixed_movie_id_list

#Zmiana numeracji ID użytkowników na zaczynające się od 0 w ratings_df
ratings_df["user_id"] = ratings_df["user_id"].apply(lambda x: x-1)

ratings_df = ratings_df.reset_index(drop = True)

# Normalizacja ocen w zbiorze ratings_df
ratings_df["old_rating"] = ratings_df["rating"]
ratings_df["rating"] = MinMaxScaler(feature_range=(0,1)).fit_transform(ratings_df[["rating"]])

# Zmiana numeracji ID filmów w movies_df na movie_id z ratings_df.
# movies_df zawiera dodatkowe filmy, nieobecne w ratings_df, ich id zostaje zamienione na NaN 
movies_df["old_movie_id"] = movies_df["movie_id"]
movies_df["movie_id"] = movies_df["old_movie_id"].apply(lambda x: old_to_new_id_dict.get(x))

absent_movies = movies_df[pd.isna(movies_df["movie_id"])]
# Usunięcie nieobecnych filmów z movies_df
movies_df = movies_df[pd.isna(movies_df["movie_id"])==False].sort_values("movie_id")

# # Ewentualne zmienienie id nieobecnych filmów (filmów posiadająch NaN w kolumnie "movie_id") na kolejne po filmach obecnych
# absent_movies["movie_id"] = range(int(movies_df["movie_id"].iloc[-1])+1, int(movies_df["movie_id"].iloc[-1])+len(absent_movies)+1)
# movies_df = pd.concat([movies_df, absent_movies], axis=0).reset_index(drop=True)

# Utworzenie list gatunków dla każdego z filmów
movies_df["genres_list"] = movies_df["genres"].apply(lambda genres_string: genres_string.split("|"))

genres_counts_in_movies = [len(strings) for strings in movies_df["genres_list"]]

all_genres = set(movies_df.explode('genres_list')['genres_list'].tolist())

genres_type_counts = pd.Series([item for sublist in movies_df['genres_list'] for item in sublist]).value_counts()

movies_df = movies_df.reset_index(drop = True)

# Sortowanie tagów w genome_scores według wartości trafności (relevance) dla każdego filmu
genome_scores_sorted = genome_scores.sort_values(by=['movie_id', 'relevance'], ascending=[True, False])

# Dodanie kolumny z rankingiem tagów wg trafności do danego filmu od 1 do N
genome_scores_sorted['relevance_rank'] = genome_scores_sorted.groupby('movie_id')['relevance'].rank(ascending=False, method='first')

# Wybranie pierwszych N=50 najtrafniejszych tagów dla każdego filmu
genome_scores_sorted = genome_scores_sorted[genome_scores_sorted["relevance_rank"] <= 50]

# Zmiana numeracji ID filmów genome_scores
genome_scores_sorted["old_movie_id"] = genome_scores_sorted["movie_id"]
# POWOLNE ROZWIĄZANIE
genome_scores_sorted["movie_id"] = genome_scores_sorted["old_movie_id"].apply(lambda x: movies_df["movie_id"][movies_df["old_movie_id"] == x].values[0])

genome_scores_sorted = genome_scores_sorted.reset_index(drop = True)

# Dodanie tagu do każdego id tagu
#genome_scores_sorted["tag"] = genome_scores_sorted["tag_id"].apply
genome_scores_sorted["tag"] = genome_scores_sorted['tag_id'].map(genome_tags.set_index('tag_id')['tag'])

# Dodanie kolumny z listą tagów dla każdego filmu
tags_by_movie = genome_scores_sorted.groupby("movie_id")["tag"].apply(list)
movies_df["tags_list"] = movies_df["movie_id"].map(tags_by_movie)
movies_df.loc[movies_df['tags_list'].isnull(),['tags_list']] = movies_df.loc[movies_df['tags_list'].isnull(),'tags_list'].apply(lambda x: [])
# Dodanie kolumny z tagami dla każdego filmu jako string
movies_df["tags"] = movies_df["tags_list"].apply(lambda x: ','.join(x))


##############################################################################################
### Analiza danych ###
##############################################################################################
example_user = 0

print(movies_df[["movie_id", 'title', 'genres', 'tags_list']].head(10))

# Liczba filmów
print("Liczba filmów w zbiorze movies_df: ", len(ratings_df.movie_id.unique()))
# Liczba użytkowników
print("Liczba filmów w zbiorze movies_df: ", len(ratings_df.user_id.unique()))

# Lista gatunków filmowych
print(all_genres)
print("(",len(all_genres) ,")")

# Rozkład gatunków w filmach
fig, ax = plt.subplots(figsize=(15,5))
sns.barplot(x=genres_type_counts.index, y=genres_type_counts.values, color='steelblue')
ax.set_xlabel('Gatunki', fontsize = 12)
ax.set_ylabel('Liczba filmów', fontsize = 12)
ax.set_title('Liczba filmów posiadających danych gatunek', fontsize = 18)
plt.xticks(rotation=45, fontsize = 12)
plt.yticks(fontsize = 12)
#plt.tight_layout()
plt.show()


# Lista wszystkich możliwych ocen przed i po normalizacji
print(np.sort(ratings_df.old_rating.unique()))
print(np.sort(ratings_df.rating.unique()))

# Rozkład ocen
rating_type_counts = ratings_df.groupby("old_rating").count()["movie_id"]

fig, ax = plt.subplots(figsize=(15,5))
sns.barplot(x=rating_type_counts.index, y=rating_type_counts.values, color='steelblue')
ax.set_xlabel('Wysokość oceny', fontsize = 12)
ax.set_ylabel('Liczba filmów', fontsize = 12)
ax.set_title('Rozkład wysokości ocen wśród filmów', fontsize = 18)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
#plt.tight_layout()
plt.show()

def find_x_at_halfarea(counts):
    """
    Znajduje takie x przy którym pole pod wykresem ocen podzielić można na pół.

    Parameters
    ----------
    rating_counts : TYPE pd.Series
        Seria z liczbą ocen dla każdego z użytkowników/filmów.

    Returns
    -------
    x_at_halfarea : TYPE int

    """
    rating_counts_integral_to_x = np.array([np.trapz(counts.values[:i], counts.index.values[:i]) for i in range(len(counts))])
    half_area = rating_counts_integral_to_x[-1] / 2
    
    x_at_halfarea = next(i for i, v in enumerate(rating_counts_integral_to_x) if v >= half_area)
    return x_at_halfarea

    
# Liczba ocen wystawionych dla każdego z filmów
movies_ratings_counts = ratings_df.groupby("movie_id").count()["rating"].sort_values(ascending=False).reset_index(drop=True) 
x_h_movies = find_x_at_halfarea(movies_ratings_counts)

fig, ax = plt.subplots(figsize=(15,5))
sns.lineplot(x=movies_ratings_counts.index, y=movies_ratings_counts.values, color='steelblue')
ax.set_xlabel('Liczba filmów', fontsize = 12)
ax.set_ylabel('Liczba ocen', fontsize = 12)
ax.set_title('Liczba ocen wystawionych dla każdego z filmów', fontsize = 18)
ax.fill_between(movies_ratings_counts.index, movies_ratings_counts.values, alpha=0.2)
plt.axvline(x=x_h_movies, color='red', linewidth=2)
plt.text(525, -35, str(x_h_movies), fontsize=12)
plt.yticks(fontsize = 12)
#plt.tight_layout()
plt.show()

# Liczba ocen wystawionych przez każdego z użytkowników
users_ratings_counts = ratings_df.groupby("user_id").count()["rating"].sort_values(ascending=False).reset_index(drop=True)
x_h_users = find_x_at_halfarea(users_ratings_counts)

fig, ax = plt.subplots(figsize=(15,5))
sns.lineplot(x=users_ratings_counts.index, y=users_ratings_counts.values, color='steelblue')
ax.set_xlabel('Liczba użytkowników', fontsize = 12)
ax.set_ylabel('Liczba ocen', fontsize = 12)
ax.set_title('Licza ocen wystawionych przez każdego z użytkowników', fontsize = 18)
ax.fill_between(users_ratings_counts.index, users_ratings_counts.values, alpha=0.2)
plt.axvline(x=x_h_users, color='red', linewidth=2)
plt.text(65, -291, str(x_h_users), fontsize=12)
plt.yticks(fontsize = 12)
#plt.tight_layout()
plt.show()

# Utworzenie macierzy rzadkiej ocen użytkowników i filmów
ratings_sparse_matrix = ratings_df.pivot_table(index='user_id', columns='movie_id', values='rating', fill_value=np.nan)

# Wykres macierzy rzadkiej
fig, ax = plt.subplots(figsize=(20,5))
sns.heatmap(ratings_sparse_matrix.isnull(), vmin=0, vmax=1, cbar=False, ax=ax).set_title("Macierz rzadka ocen użytkowników i filmów")
ax.set_xlabel('Użytkownicy', fontsize = 12)
ax.set_ylabel('Filmy', fontsize = 12)
ax.set_title('Macierz rzadka ocen użytkowników i filmów', fontsize = 18)
plt.show()


##############################################################################################
### Modele systemów rekomendacyjnych
##############################################################################################

# Podział zbioru ratings_df na uczący i testowy. Podział losowy.
train, test = train_test_split(ratings_df, test_size=0.2, stratify=ratings_df['user_id'], random_state=1)
#print(len(test.user_id.unique()))


## Model losowy
def random_recommender(users_list, movies_list, k):
    '''
    Tworzy listę list K losowych filmów do zarekomendowania użytkownikowi.

    Parameters
    ----------
    users_list : lista  id użytkowników
    movies_list : lista  id flimóW
    k : liczba filmów do zarekomendowania

    Returns
    -------
    predicted_ratings_random_list : lista list K losowych rekomendacji dla wszystkich użytkowników

    '''
    
    predicted_ratings_list = [np.array(random.sample(list(movies_list), k)) for user in range(len(users_list))]
    
    return predicted_ratings_list


## Model popularnościowy
def popularity_recommender(users_list, movies_popularity_list:pd.Series, k):
    '''
    Tworzy listę list K najpopularniejszych filmów do zarekomendowania użytkownikowi.

    Parameters
    ----------
    users_list : lista  id użytkowników 
    movies_popularity_list: pd.Series  id flimóW z ilością ocen każdego z nich
    k : liczba filmów do zarekomendowania

    Returns
    -------
    predicted_ratings_random_list : lista list K najpopularniejszych rekomendacji dla wszystkich użytkowników

    '''
    topk = movies_popularity_list.iloc[:k].index.values
    predicted_ratings_list = [topk for user in users_list]
    
    return predicted_ratings_list


## Model SVD z biblioteki surprise
from surprise import accuracy, Dataset, SVD, Reader
from surprise.model_selection import train_test_split as surprise_train_test_split
svd_reader = Reader(rating_scale=(0.0, 1.0))

svd_data = Dataset.load_from_df(ratings_df[['user_id', 'movie_id', 'rating']], svd_reader)

svd_trainset, svd_testset = surprise_train_test_split(svd_data, test_size=0.1)

svd_algo = SVD()

svd_algo.fit(svd_trainset)

svd_predictions = svd_algo.test(svd_testset)

#accuracy.mae(svd_predictions)

svd_test_df = pd.DataFrame(svd_testset, columns=['user_id', 'movie_id', 'rating'])

svd_unique_users = svd_test_df.user_id.unique()
svd_unique_movies =  svd_test_df.movie_id.unique()

def pred_user_svd(user_id, k):
    """
    Tworzy listę top K rekomendacji dla danego user_id według modelu SVD.
    
    Parameters
    ----------
    user_id : int
        id użytkownika.
    k : int 
        liczba filmów do rekomendacji.
    Returns
    -------
    top_movies : list
        list top K rekomendacji.
    
    """

    predictions = []
    for item in svd_unique_movies:
        prediction = svd_algo.predict(user_id, item)
        predictions.append(prediction)
      
    predictions.sort(key=lambda x: x.est, reverse=True)
      
    top_movies = [pred.iid for pred in predictions[:k]]
      
    return top_movies


## Model Content Based - Doc2Vec

doc2vec_vector_size = 20

#  Odkomentować w celu uczenia nowego modelu Doc2Vec !!!! ######
# Najczęstsze i zbędne wyrazy w języku angielskim
stop_words = stopwords.words('english')

def create_tokens(string_of_tags):
    """
    Funkcja czyszcząca listę tagów i tworząca z nich listę tokenów.

    Parameters
    ----------
    string_of_tags : TYPE string
        String z wszystkimi tagami odzielonymi przecinkami.

    Returns
    -------
    tokens : TYPE list
        Lista tokenów.

    """
    # Zamiana na małe litery
    string_of_tags.lower()
    # Podzielenie stringa na pojedyńcze wyrazy
    tokens = word_tokenize(string_of_tags)    
    # Usunięcie z listy tokenów wyrazów będacych jednym ze stop words oraz nie będących literowymi
    tokens = [token for token in tokens if not token in stop_words and token.isalpha()]
      
    return tokens

tags_tokens = [create_tokens(tags) for tags in movies_df["tags"]]

# Lista obiektów TaggedDocument zawierających tokeny dla każdego z filmów
tagged_docs = [TaggedDocument(words=item, tags=[str(index)]) for index,item in enumerate(tags_tokens)]


# Tworzenie modelu Doc2Vec.
# dm=0 -> wykorzystanie algorytmu distributed bag of words (PV-DBOW) 
doc2vec_model = Doc2Vec(vector_size=doc2vec_vector_size, alpha=0.025, min_alpha=0.00025, min_count=1, dm=0, workers=4)
doc2vec_model.build_vocab(tagged_docs)

# Uczenie modelu Doc2Vec
epoch_num = 50
print('Epoka: ')
for epoch in range(epoch_num):
  print(epoch)
  doc2vec_model.train(tagged_docs, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)
  doc2vec_model.alpha -= 0.0002
  doc2vec_model.min_alpha = doc2vec_model.alpha

# Zapisywanie modelu do pliku
#doc2vec_model.save('./models/doc2vecModel')

# Wczytywanie modelu z pliku
doc2vec_model = Doc2Vec.load('./models/doc2vecModel')

# Wektory reprezentujące filmy w 20 wymiarowej przestrzeni
doc2vec_movies_embbedings = doc2vec_model.dv.vectors
print(doc2vec_movies_embbedings.shape)


# Top20 filmów podobnych do filmu "Schindler's List"
example_movie = 28
sims = doc2vec_model.dv.most_similar(positive = [example_movie], topn = 20)
sims_df = pd.DataFrame(sims,columns=["movie_id", "cosine_similarity"])
sims_df.movie_id = sims_df.movie_id.astype('float64')
sims_df['title'] = sims_df['movie_id'].map(movies_df.set_index('movie_id')['title'])
sims_df['tags_list'] = sims_df['movie_id'].map(movies_df.set_index('movie_id')['tags_list'])

print("Top 20 filmów podobnych do filmu movie_id =", example_movie, "(",movies_df["title"][movies_df["movie_id"] == example_movie ].values[0], ")")
print(sims_df)

# Opisanie filmu za pomocą chmury słów (WordCloud) z tagów filmów podobnych
wordcloud_text = ' '.join([','.join(t) for t in sims_df.tags_list])
plt.rcParams["figure.figsize"] = (15,10)
# Wygenerowanie WordCloud
wordcloud = WordCloud(width = 1024, height = 1024, background_color = 'white').generate(wordcloud_text)
plt.figure(figsize=(10,8),facecolor = 'white', edgecolor='blue')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Top20 filmów podobnych do filmu "Lord of the Rings: Fellowship of the Ring"
example_movie = 747
sims = doc2vec_model.dv.most_similar(positive = [example_movie], topn = 20)
sims_df = pd.DataFrame(sims,columns=["movie_id", "cosine_similarity"])
sims_df.movie_id = sims_df.movie_id.astype('float64')
sims_df['title'] = sims_df['movie_id'].map(movies_df.set_index('movie_id')['title'])
sims_df['tags_list'] = sims_df['movie_id'].map(movies_df.set_index('movie_id')['tags_list'])

print("Top 20 filmów podobnych do filmu movie_id =", example_movie, "(",movies_df["title"][movies_df["movie_id"] == example_movie ].values[0], ")")
print(sims_df)

# Opisanie filmu za pomocą chmury słów (WordCloud) z tagów filmów podobnych
wordcloud_text = ' '.join([','.join(t) for t in sims_df.tags_list])
plt.rcParams["figure.figsize"] = (15,10)
# Wygenerowanie WordCloud
wordcloud = WordCloud(width = 1024, height = 1024, background_color = 'white').generate(wordcloud_text)
plt.figure(figsize=(10,8),facecolor = 'white', edgecolor='blue')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# Wizualizacja wektorów z pomocą narzędzia UMAP

# redukcja do 2 wymiarów
doc2vec_movies_embbedings_umap = umap.UMAP(n_components=2, n_neighbors = 10, min_dist = 0.005, metric = 'cosine').fit_transform(doc2vec_movies_embbedings)

x = doc2vec_movies_embbedings_umap[:,0]
y = doc2vec_movies_embbedings_umap[:,1]

fig, ax = plt.subplots(figsize=(15, 10))
plt.axis('off')
plt.title('Wektory filmów zredukowane do 2D')
plt.scatter(x, y, s=1)

# Wskazanie pozycji wybranych filmów
ax.annotate("Toy Story",
            xy=(x[0], y[0]),
            xytext=(x[0]+1, y[0]+0),
            arrowprops=dict(facecolor='black', width=0.001, headwidth=4))
ax.annotate("Toy Story 2",
            xy=(x[729], y[729]),
            xytext=(x[729]+1, y[729]+1),
            arrowprops=dict(facecolor='black', width=0.001, headwidth=4))
ax.annotate("Shrek",
            xy=(x[738], y[738]),
            xytext=(x[738]+0.5, y[738]+1),
            arrowprops=dict(facecolor='black', width=0.001, headwidth=4))
ax.annotate("The Shining",
            xy=(x[7855], y[7855]),
            xytext=(x[7855]+1, y[7855]+1),
            arrowprops=dict(facecolor='black', width=0.001, headwidth=4))
ax.annotate("2001: A Space Odyssey",
            xy=(x[717], y[717]),
            xytext=(x[717]+1, y[717]+0),
            arrowprops=dict(facecolor='black', width=0.001, headwidth=4))
ax.annotate("Contact",
            xy=(x[720], y[720]),
            xytext=(x[720]+1.5, y[720]+0.8),
            arrowprops=dict(facecolor='black', width=0.001, headwidth=4))
ax.annotate("Arrival",
            xy=(x[1367], y[1367]),
            xytext=(x[1367]+2, y[1367]+2),
            arrowprops=dict(facecolor='black', width=0.001, headwidth=4))
ax.annotate("Pretty Woman",
            xy=(x[477], y[477]),
            xytext=(x[477]-1, y[477]+1),
            arrowprops=dict(facecolor='black', width=0.001, headwidth=4))
ax.annotate("Bridget Jones's Diary",
            xy=(x[444], y[444]),
            xytext=(x[444]-2, y[444]+2),
            arrowprops=dict(facecolor='black', width=0.001, headwidth=4))
plt.show()


## Tworzenie profili użytkowników poprzez uśrednienie wektorów obejrzanych przez nich filmów

# Wybranie filmów ocenionych wyższej niż 3.0 ( 0.55... po normalizacji) ze zbioru TRENINGOWEGO, zakładając, 
# że oceny niższe oznaczają, że użytkownik nie jest zainteresowany danymi filmami

doc2vec_train_filtered = train[train["rating"] > 0.55]

def get_doc2vec_user_vector(user_id):
    """
    Zwraca wektor stanowiący profil użytkownika. Jeżeli user_id nie istnieje w doc2vec_train_filtered, 
    zwraca średnią wszystkich wektorów filmów
    """        
    user_movies = doc2vec_train_filtered["movie_id"][doc2vec_train_filtered["user_id"] == user_id]  
    if user_movies.empty:
        print("Doc2Vec - użytkownik: ",str(user_id)," nie istnieje w zbiorze. Wektor tego użytkownika == średnia wszystkich wektorów filmów")
        return np.mean(doc2vec_movies_embbedings, axis=0)
    
    user_vectors = doc2vec_movies_embbedings[user_movies, :]    
    user_vectors_mean = np.mean(user_vectors, axis=0) 
    
    return user_vectors_mean

    
def recommend_items_doc2vec(user_id, k, show_cos_sims=False):
    """
    Zwraca top K przedmiotów najbardziej podobnych wg. podobieństwa cosinusowego do wektora użytkownika.

    Parameters
    ----------
    user_id : int
        id użytkownika.
    k : int
        liczba filmów do zarekomendowania.
    show_cos_sims : boolean
        dodaj wartości podobieństwa cosinusowego filmów do wektora użytkownika       
    Returns
    -------
    predicted_ratings_list : list
        lista top K filmów.
    """
    user_vector = get_doc2vec_user_vector(user_id)
    
    predicted_ratings_list = doc2vec_model.dv.most_similar(positive = [user_vector], topn = k)
    
    array = np.array([[t[0], t[1]] for t in predicted_ratings_list]).astype('float64')
    
    if show_cos_sims:
       return array
        
    return array[:,0]


## Model Collaborative Filtering - Neural Collaborative Filtering

# Liczba użytkowników i filmów
users_len  = len(ratings_df.user_id.unique())
movies_len  = len(ratings_df.movie_id.unique())

# Wielkość wektorów cech ukrytych
movie_embedding = 50
user_embedding = 50

# Warstwy wejściowe #
input_movie = Input(shape=[1], name='input-movie')
input_user = Input(shape=[1], name='input-user')

# Warstwy faktoryzacji macierzy #
# Embeddingi faktoryzacji macierzy
mf_movie_embedding = Embedding(input_dim = movies_len + 1, output_dim = movie_embedding, name='mf_movie_embedding')(input_movie)
mf_user_embedding = Embedding(input_dim = users_len + 1, output_dim = user_embedding, name='mf_user_embedding')(input_user)
# Spłaszczenie embeddingów
mf_movie_flatten = Flatten(name='mf_movie_flatten')(mf_movie_embedding)
mf_user_flatten = Flatten(name='mf_user_flatten')(mf_user_embedding)
# Wyjście części modelu faktoryzującej macierz
mf_output = Dot(axes=1)([mf_movie_flatten, mf_user_flatten]) 

# Warstwy perceptronu wielowarstwoego #
# Embeddingi MLP
mlp_movie_embedding = Embedding(input_dim = movies_len + 1, output_dim = movie_embedding, name='mlp_movie_embedding')(input_movie)
mlp_user_embedding = Embedding(input_dim = users_len + 1, output_dim = user_embedding, name='mlp_user_embedding')(input_user)
# Spłaszczenie embeddingów
mlp_movie_flatten = Flatten(name='mlp_movie_flatten')(mlp_movie_embedding)
mlp_user_flatten = Flatten(name='mlp_user_flatten')(mlp_user_embedding)
# Konkatenacja spłaszczonych embeddingów
mlp_concatenate = Concatenate(axis=1)([mlp_movie_flatten, mlp_user_flatten]) 
mlp_concatenate_dropout = Dropout(0.2)(mlp_concatenate)

mlp_dense_1 = Dense(32, activation='relu', name='mlp_dense_1')(mlp_concatenate_dropout)
mlp_batch_norm_1 = BatchNormalization(name='mlp_batch_norm_1')(mlp_dense_1)
mlp_dropout_1 = Dropout(0.2)(mlp_batch_norm_1)
mlp_dense_2 = Dense(16, activation='relu', name='mlp_dense_2')(mlp_dropout_1)
mlp_batch_norm_2 = BatchNormalization(name='mlp_batch_norm_2')(mlp_dense_2)
mlp_dropout_2 = Dropout(0.2)(mlp_batch_norm_2)
# Wyjście części modelu MLP
mlp_output = Dense(8, activation='relu', name='mlp_output')(mlp_dropout_2)

# Konkatenacja wyjść obu części modelu
mf_mlp_concat = Concatenate(axis=1)([mf_output, mlp_output])

# Predykcja sieci
output = Dense(1, name='output', activation='relu')(mf_mlp_concat)

NeuCF_model = Model([input_user, input_movie], output)
NeuCF_model.compile(optimizer=Adam(), loss='mean_absolute_error')

# Uczenie modelu NeuCF
history = NeuCF_model.fit([train.user_id, train.movie_id], train.rating, epochs=10, validation_data=[[test.user_id, test.movie_id], test.rating])


# Zapisywanie modelu
#NeuCF_model.save('./models/NeuCF_50d_10e')
# Wczytywanie modelu
# NeuCF_model = keras.models.load_model('./models/NeuMF_20d_60e_local')

# Wykres wartości błędu MAE
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('NeuCF MAE loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Predykcje ocen dla wszystkich par użytkownik film w zbiorze testowym
y_hat = np.round(NeuCF_model.predict([test.user_id, test.movie_id]), decimals=4)
y_true = test.rating

print("MAE modelu NeuCF_model w zbiorze testowym: ", mean_absolute_error(y_true, y_hat))

def recommend_items_neucf(user_id, k):
    """
    Zwraca top K przedmiotów najbardziej podobnych wg. wysokości predykowanych ocen dla użytkownika.

    Parameters
    ----------
    user_id : int
        id użytkownika.
    k : int
        liczba filmów do zarekomendowania.
    
    Returns
    -------
    predicted_ratings_list : list
        lista top K filmów.
    """
      
    all_movies = test['movie_id'].unique()  
    
    predictions = pd.DataFrame({'user_id': user_id, 'movie_id': all_movies})

    # Wygenerowanie predykcji dla każdej pary user_id i movie_id
    predictions['rating'] = np.round(NeuCF_model.predict([predictions['user_id'], predictions['movie_id']], verbose=False), decimals=4)
    
    predictions = predictions.sort_values(by="rating", ascending=False).head(k)
    
    return np.array(predictions["movie_id"]).astype('float64')

# Model hybrydowy - połączenie NeuCF i Doc2Vec

#Wektory doc2vec opisujące wszystkich użytkowników
doc2vec_users_embbedings = np.array([get_doc2vec_user_vector(user_id) for user_id in ratings_df.user_id.unique()])

# Liczba użytkowników i filmów
users_len  = len(ratings_df.user_id.unique())
movies_len  = len(ratings_df.movie_id.unique())

# Wielkość wektorów cech ukrytych
movie_embedding = 50
user_embedding = 50

# Warstwy wejściowe #
input_movie = Input(shape=[1], name='input-movie')
input_user = Input(shape=[1], name='input-user')

# Warstwy faktoryzacji macierzy #
# Embeddingi faktoryzacji macierzy
mf_movie_embedding = Embedding(input_dim = movies_len + 1, output_dim = movie_embedding, name='mf_movie_embedding')(input_movie)
mf_user_embedding = Embedding(input_dim = users_len + 1, output_dim = user_embedding, name='mf_user_embedding')(input_user)
# Spłaszczenie embeddingów
mf_movie_flatten = Flatten(name='mf_movie_flatten')(mf_movie_embedding)
mf_user_flatten = Flatten(name='mf_user_flatten')(mf_user_embedding)
# Wyjście części modelu faktoryzującej macierz
mf_output = Dot(axes=1)([mf_movie_flatten, mf_user_flatten]) 

# Warstwy perceptronu wielowarstwoego #
# Embeddingi MLP
mlp_movie_embedding = Embedding(input_dim = movies_len + 1, output_dim = movie_embedding, name='mlp_movie_embedding')(input_movie)
mlp_user_embedding = Embedding(input_dim = users_len + 1, output_dim = user_embedding, name='mlp_user_embedding')(input_user)
# Spłaszczenie embeddingów
mlp_movie_flatten = Flatten(name='mlp_movie_flatten')(mlp_movie_embedding)
mlp_user_flatten = Flatten(name='mlp_user_flatten')(mlp_user_embedding)
# Konkatenacja spłaszczonych embeddingów
mlp_concatenate = Concatenate(axis=1)([mlp_movie_flatten, mlp_user_flatten]) 
mlp_concatenate_dropout = Dropout(0.2)(mlp_concatenate)

mlp_dense_1 = Dense(32, activation='relu', name='mlp_dense_1')(mlp_concatenate_dropout)
mlp_batch_norm_1 = BatchNormalization(name='mlp_batch_norm_1')(mlp_dense_1)
mlp_dropout_1 = Dropout(0.2)(mlp_batch_norm_1)
mlp_dense_2 = Dense(16, activation='relu', name='mlp_dense_2')(mlp_dropout_1)
mlp_batch_norm_2 = BatchNormalization(name='mlp_batch_norm_2')(mlp_dense_2)
mlp_dropout_2 = Dropout(0.2)(mlp_batch_norm_2)
# Wyjście części modelu MLP
mlp_output = Dense(8, activation='relu', name='mlp_output')(mlp_dropout_2)

# Warstwy części content based Doc2Vec

doc2vec_movie_embedding = Embedding(input_dim = movies_len, output_dim = doc2vec_vector_size,
                                                       weights=[doc2vec_movies_embbedings], 
                                                       trainable=False, 
                                                       name='doc2vec_movie_embedding')(input_movie)
doc2vec_user_embedding = Embedding(input_dim = users_len, output_dim = doc2vec_vector_size,
                                                      weights=[doc2vec_users_embbedings], 
                                                      trainable=False,
                                                      name='doc2vec_user_embedding')(input_user)

doc2vec_movie_flatten = Flatten(name='doc2vec_movie_flatten')(doc2vec_movie_embedding)
doc2vec_user_flatten = Flatten(name='doc2vec_user_flatten')(doc2vec_user_embedding)

doc2vec_concatenate = Concatenate()([doc2vec_movie_flatten, doc2vec_user_flatten])
doc2vec_dense_1 = Dense(units=16, activation='relu', name='doc2vec_dense_1')(doc2vec_concatenate)
doc2vec_output = Dense(8, activation='relu', name='doc2vec_output')(doc2vec_dense_1)
# Konkatenacja wyjść części modelu
mf_mlp_doc2vec_concat = Concatenate(axis=1)([mf_output, mlp_output, doc2vec_output])

# Predykcja sieci
output = Dense(1, name='output', activation='relu')(mf_mlp_doc2vec_concat)

Hybrid_model = Model([input_user, input_movie], output)
Hybrid_model.compile(optimizer=Adam(), loss='mean_absolute_error')


SVG(model_to_dot(Hybrid_model, show_shapes= True, show_layer_names=True, dpi=65).create(prog='dot', format='svg'))


# Uczenie modelu hybrydowego
history_hybrid = Hybrid_model.fit([train.user_id, train.movie_id], train.rating, epochs=10, validation_data=[[test.user_id, test.movie_id], test.rating])

# Zapisywanie modelu
#Hybrid_model.save('./models/Hybrid_model_50d_10e')
# Wczytywanie modelu
# Hybrid_model = keras.models.load_model('./models/Hybrid_model_20d_60e_local')

# Wykres wartości błędu MAE
plt.plot(history_hybrid.history['loss'])
plt.plot(history_hybrid.history['val_loss'])
plt.title('Hybrid model MAE loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Predykcje ocen dla wszystkich par użytkownik film w zbiorze testowym
y_hat = np.round(Hybrid_model.predict([test.user_id, test.movie_id]), decimals=4)
y_true = test.rating

print("MAE modelu Hybrid_model w zbiorze testowym: ", mean_absolute_error(y_true, y_hat))


def recommend_items_hybrid(user_id, k):
    """
    Zwraca top K przedmiotów najbardziej podobnych wg. wysokości predykowanych ocen dla użytkownika.

    Parameters
    ----------
    user_id : int
        id użytkownika.
    k : int
        liczba filmów do zarekomendowania.
    
    Returns
    -------
    predicted_ratings_list : list
        lista top K filmów.
    """
      
    all_movies = test['movie_id'].unique()  
    
    predictions = pd.DataFrame({'user_id': user_id, 'movie_id': all_movies})

    # Wygenerowanie predykcji dla każdej pary user_id i movie_id
    predictions['rating'] = np.round(Hybrid_model.predict([predictions['user_id'], predictions['movie_id']], verbose=False), decimals=4)
    
    predictions = predictions.sort_values(by="rating", ascending=False).head(k)
    
    return np.array(predictions["movie_id"]).astype('float64')


##############################################################################################
### Ewaluacja modeli ###
##############################################################################################
# Sprawdzanie rekomendacji dla przykładowego użytkownika
def check_user_recs(user_id,users_predicted_ratings_list):
    """
    Wyświetla listę topk filmów rzeczywiście ocenionych przez podanego użytkownika 
    oraz listę topk filmów zarekomendowanych dla użytkownika przez dany model.

    Parameters
    ----------
    user_id : int
        id użytkownika.
    users_predicted_ratings_list : list
        lista list topk rekomendacji utworzyonych prez jeden z modelu rekomendacji.

    Returns
    -------
    None.

    """
    
    #pd.set_option('display.max_columns', 6)
    print("Obejrzane filmy: ")
    example_true_ratings_list = top_ratings_real[user_id]
    example_true_ratings_list_df = movies_df[movies_df['movie_id'].isin(example_true_ratings_list)]
    print(example_true_ratings_list_df.loc[:, ['title', 'genres']].to_string(index=False))
    
    print("Rekomendowane filmy: ")
    example_predicted_ratings_list = users_predicted_ratings_list[user_id]
    example_predicted_ratings_list_df = movies_df[movies_df['movie_id'].isin(example_predicted_ratings_list)]
    print(example_predicted_ratings_list_df.loc[:, ['title', 'genres']].to_string(index=False))


# Tabela wyników ewaluacji
evaluation_df = pd.DataFrame(columns=["MTP", "MRR", "MAP", "MAR", "Personalization", "Intra-list Similarity", "Coverage", ])
    
def evaluate_model(model_name, list_of_topk_lists_true, list_of_topk_lists_pred, k, catalog_dataset=test):
    '''
    Funkcja obliczająca dla danego modelu wszystkie zastosowane metryki i dodające ich wyniki do evaluation_df.
    
    Parameters
    ----------
    model_name : nazwa modelu
    list_of_topk_lists_true : lista list najwyzej ocenionych filmów przez użytkowników
    list_of_topk_lists_pred : lista list top k rekomendowanych filmów
    k : liczba rekomendacji
    catalog_dataset : dataset z którego wybrane zostaną movie_id.unique()

    Returns
    -------
    evaluation_list : wyniki metryk dla danego modelu

    '''
    real_list = [list(x) for x in list_of_topk_lists_true]
    pred_list = [list(x) for x in list_of_topk_lists_pred]
    catalog = list(catalog_dataset.movie_id.unique())
    
    evaluation_list = [r_metrics.mean_true_positives_percentage(list_of_topk_lists_true, list_of_topk_lists_pred),
                       r_metrics.mean_reciprocal_rank(list_of_topk_lists_true, list_of_topk_lists_pred),
                       r_metrics.mean_average_precision(list_of_topk_lists_true,list_of_topk_lists_pred),
                       #recmetrics.mark(real_list, pred_list,  k),
                       r_metrics.mean_average_recall(list_of_topk_lists_true,list_of_topk_lists_pred),
                       recmetrics.personalization(pred_list),
                       recmetrics.intra_list_similarity(pred_list, pd.DataFrame(doc2vec_movies_embbedings)),
                       recmetrics.prediction_coverage(pred_list, catalog)] 
    
    evaluation_df.loc[model_name] = evaluation_list


k = 20

################## Prawdziwe oceny ##################
# Lista z listami najlepszych filmów dla każdego użytkownika (wg. ocen)
top_ratings_real = test.groupby('user_id').apply(
    lambda x: np.array(x.sort_values('rating', ascending=False)['movie_id'].head(k)).astype('float64')).tolist()


###### Rekomendacja losowa
random_topk_recommendations = random_recommender(np.sort(test.user_id.unique()), np.sort(test.movie_id.unique()), k)

# Sprawdzenie rekomendacji dla wybranych użytkowników
check_user_recs(1,random_topk_recommendations)

check_user_recs(100,random_topk_recommendations)

check_user_recs(150,random_topk_recommendations)

# Oblczenie metryk dla modelu
evaluate_model("Random", top_ratings_real, random_topk_recommendations, k)


################## Rekomendacja popularnościowa ##################
popularity_topk_recommendations = popularity_recommender(np.sort(test.user_id.unique()), test.movie_id.value_counts(), k)

# Sprawdzenie rekomendacji dla wybranych użytkowników
check_user_recs(1,popularity_topk_recommendations)

check_user_recs(100,popularity_topk_recommendations)

check_user_recs(150,popularity_topk_recommendations)

# Oblczenie metryk dla modelu
evaluate_model("Popularity", top_ratings_real, popularity_topk_recommendations, k)

################## Rekomendacja SVD ##################
svd_topk_recommendations = [np.array(pred_user_svd(user_id, k)) for user_id in svd_unique_users]

# Lista z listami najlepszych filmów dla każdego użytkownika (wg. ocen) dla SVD
svd_top_ratings_real = svd_test_df.groupby('user_id').apply(
    lambda x: np.array(x.sort_values('rating', ascending=False)['movie_id'].head(k)).astype('float64')).tolist()

# Sprawdzenie rekomendacji dla wybranych użytkowników
check_user_recs(1,svd_topk_recommendations)

check_user_recs(100,svd_topk_recommendations)

check_user_recs(150,svd_topk_recommendations)

# Oblczenie metryk dla modelu
evaluate_model("SVD", svd_top_ratings_real, svd_topk_recommendations, k, catalog_dataset = svd_test_df)

################## Rekomendacja content based Doc2Vec ##################

# Lista z listami najlepszych filmów dla każdego użytkownika (wg. ocen) dla modelu Doc2Vec
doc2vec_top_ratings_real = test[test['user_id'].isin(doc2vec_train_filtered['user_id'])].groupby('user_id').apply(
    lambda x: np.array(x.sort_values('rating', ascending=False)['movie_id'].head(k)).astype('float64')).tolist()

doc2vec_topk_recommendations = [recommend_items_doc2vec(user_id, k) for user_id in doc2vec_train_filtered.user_id.unique()]

# Sprawdzenie rekomendacji dla wybranych użytkowników
check_user_recs(1,doc2vec_topk_recommendations)

check_user_recs(100,doc2vec_topk_recommendations)

check_user_recs(150,doc2vec_topk_recommendations)

# Oblczenie metryk dla modelu
evaluate_model("CB - Doc2Vec", doc2vec_top_ratings_real, doc2vec_topk_recommendations, k, catalog_dataset = ratings_df)

################## Rekomendacja collaborative filtering NeuCF ##################
# Długie działanie! 
neucf_topk_recommendations = [recommend_items_neucf(user_id, k) for user_id in test.user_id.unique()]

check_user_recs(1,neucf_topk_recommendations)

check_user_recs(100,neucf_topk_recommendations)

check_user_recs(150,neucf_topk_recommendations)

# Oblczenie metryk dla modelu
evaluate_model("CF - NeuCF", top_ratings_real, neucf_topk_recommendations, k)

################## Rekomendacja hybrydowa NeuCF Doc2Vec + NeuCF ##################
hybrid_topk_recommendations = [recommend_items_hybrid(user_id, k) for user_id in ratings_df.user_id.unique()]

check_user_recs(1,hybrid_topk_recommendations)

check_user_recs(100,hybrid_topk_recommendations)

check_user_recs(150,hybrid_topk_recommendations)

# Oblczenie metryk dla modelu
evaluate_model("Hybrid model", top_ratings_real, hybrid_topk_recommendations, k)

evaluation_df = evaluation_df.round(5)

print(evaluation_df)

# Do rysowania wykresów usuwanie wyników modelu popularnościowego, przeszkadzającego w porównywniu innych modeli.
evaluation_df_for_plot = evaluation_df.drop('Popularity')

for col_name in evaluation_df_for_plot.columns:
    column_name = col_name
    #sns.set(style="whitegrid")
    #sns.set_palette("steelblue")
    ax = sns.barplot(x=evaluation_df_for_plot.index, y=column_name, data=evaluation_df_for_plot, color='steelblue')
    ax.set_xticklabels(ax.get_xticklabels())
    for i, v in enumerate(evaluation_df_for_plot[column_name]):
        ax.text(i, v, str(v), ha='center', va='bottom')
    plt.title(column_name)
    plt.xlabel('Modele')
    #plt.ylabel(column_name)
    plt.show()


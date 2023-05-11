# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 07:14:05 2023

@author: Olek
"""
##############################################################################################
# Import bibliotek
##############################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
import umap 
from sklearn.model_selection import train_test_split
import random
import r_metrics

# from tensorflow.keras.models import Model, Sequential
# from tensorflow.keras.layers import Embedding, Flatten, Input,  Dot,Dropout, Dense, BatchNormalization, Concatenate
# from tensorflow.keras.utils import model_to_dot
# from tensorflow.keras.optimizers import Adam
# from tensorflow import keras
# from tensorflow.keras.constraints import NonNeg
#from IPython.display import SVG

#import recmetrics



##############################################################################################
### Wczytanie i edycja zbioru movielens
##############################################################################################

ratings_df = pd.read_csv('./ml-latest-small/ratings.csv', header=0, names=['user_id', 'movie_id', 'rating', 'timestamp'])
movies_df = pd.read_csv('./ml-latest-small/movies.csv', header=0, names=['movie_id', 'title', 'genres'])
tags_df = pd.read_csv('./ml-latest-small/tags.csv', header=0, names=['user_id', 'movie_id', 'tag', 'timestamp'])

genome_scores = pd.read_csv('D:/magi2/ml-25m/genome-scores.csv', header=0, names=['movie_id', 'tag_id', 'relevance'])
genome_tags = pd.read_csv('D:/magi2/ml-25m/genome-tags.csv', header=0, names=['tag_id', 'tag'])

#ratings_df = pd.read_csv('D:/magi2/ml-25m/ratings.csv')
#movies_df = pd.read_csv('D:/magi2/ml-25m/movies.csv')
#tags_df = pd.read_csv('D:/magi2/ml-25m/tags.csv')

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

# Podział zbioru ratings_df na uczący i testowy. Podział losowy.
train, test = train_test_split(ratings_df, test_size=0.2, random_state=1)
#print(len(test.user_id.unique()))

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

# # Pokaż filmy bez gatunków
# movies_df[movies_df["genres_list"].apply(lambda x: "(no genres listed)" in x)]
# # Pusta tablica gatunków dla filmów bez gatunków
# movies_df["genres_list"] = movies_df["genres_list"].apply(lambda x: [] if "(no genres listed)" in x else x)

# #Zmiana numeracji ID filmów i użytkowników w tags_df
# tags_df["user_id"] = tags_df["user_id"].apply(lambda x: x-1)
# tags_df["old_movie_id"] = tags_df["movie_id"]
# # POWOLNE ROZWIĄZANIE
# tags_df["movie_id"] = tags_df["movie_id"].apply(lambda x: movies_df["movie_id"][movies_df["old_movie_id"] == x].values[0])

# # Dodanie kolumny z listą tagów dla każdego filmu
# tags_by_movie = tags_df.groupby("movie_id")["tag"].apply(list)
# movies_df["tags_list"] = movies_df["movie_id"].map(tags_by_movie)
# movies_df.loc[movies_df['tags_list'].isnull(),['tags_list']] = movies_df.loc[movies_df['tags_list'].isnull(),'tags_list'].apply(lambda x: [])

# # Złączenie list tagów i gatunków
# movies_df["content_list"] = movies_df.apply(lambda row: row['genres_list'] + row['tags_list'], axis=1)

# # Wyczyszczenie movies_df z niepotrzebnych kolumn
# movies_df = movies_df.drop(["tags_list","genres_list","old_movie_id","genres"], axis=1)


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
def popularity_recommender(users_list, movies_popularity_list, k):
    '''
    Tworzy listę list K najpopularniejszych filmów do zarekomendowania użytkownikowi.

    Parameters
    ----------
    users_list : lista  id użytkowników 
    movies_popularity_list: lista  id flimóW z ilością ocen każdego z nich
    k : liczba filmów do zarekomendowania

    Returns
    -------
    predicted_ratings_random_list : lista list K najpopularniejszych rekomendacji dla wszystkich użytkowników

    '''
    top20 = movies_popularity_list.iloc[:k].index.values
    predicted_ratings_list = [top20 for user in users_list]
    
    return predicted_ratings_list

## Model Content Based - Doc2Vec

#  Odkomentować w celu uczenia nowego modelu Doc2Vec !!!! ######
# # Najczęstsze i zbędne wyrazy w języku angielskim
# stop_words = stopwords.words('english')

# def create_tokens(string_of_tags):
#     """
#     Funkcja czyszcząca listę tagów i tworząca z nich listę tokenów.

#     Parameters
#     ----------
#     string_of_tags : TYPE string
#         String z wszystkimi tagami odzielonymi przecinkami.

#     Returns
#     -------
#     tokens : TYPE list
#         Lista tokenów.

#     """
#     # Zamiana na małe litery
#     string_of_tags.lower()
#     # Podzielenie stringa na pojedyńcze wyrazy
#     tokens = word_tokenize(string_of_tags)    
#     # Usunięcie z listy tokenów wyrazów będacych jednym ze stop words oraz nie będących literowymi
#     tokens = [token for token in tokens if not token in stop_words and token.isalpha()]
      
#     return tokens

# tags_tokens = [create_tokens(tags) for tags in movies_df["tags"]]

# # Lista obiektów TaggedDocument zawierających tokeny dla każdego z filmów
# tagged_docs = [TaggedDocument(words=item, tags=[str(index)]) for index,item in enumerate(tags_tokens)]


# # Tworzenie modelu Doc2Vec.
# # dm=0 -> wykorzystanie algorytmu distributed bag of words (PV-DBOW) 
# doc2vec_model = Doc2Vec(vector_size=20, alpha=0.025, min_alpha=0.00025, min_count=1, dm=0, workers=4)
# doc2vec_model.build_vocab(tagged_docs)

# # Uczenie modelu Doc2Vec
# epoch_num = 50
# print('Epoka: ')
# for epoch in range(epoch_num):
#   print(epoch)
#   doc2vec_model.train(tagged_docs, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)
#   doc2vec_model.alpha -= 0.0002
#   doc2vec_model.min_alpha = doc2vec_model.alpha

# Zapisywanie modelu do pliku
#doc2vec_model.save('./models/doc2vecModel')

# Wczytywanie modelu z pliku
doc2vec_model = Doc2Vec.load('./models/doc2vecModel')

# Wektory reprezentujące filmy w 20 wymiarowej przestrzeni
doc2vec_movies_embbedings = doc2vec_model.dv.vectors
print(doc2vec_movies_embbedings.shape)


# Przykładowe top20 filmów podobnych do wybranego filmu
example_movie = 747
sims = doc2vec_model.docvecs.most_similar(positive = [example_movie], topn = 20)
sims_df = pd.DataFrame(sims,columns=["movie_id", "cosine_similarity"])
sims_df.movie_id = sims_df.movie_id.astype('float64')
sims_df['title'] = sims_df['movie_id'].map(movies_df.set_index('movie_id')['title'])
sims_df['tags_list'] = sims_df['movie_id'].map(movies_df.set_index('movie_id')['tags_list'])

print("Top 20 filmów podobnych do filmu movie_id =", example_movie, "(",movies_df["title"][movies_df["movie_id"] == example_movie ].values[0], ")")
print(sims_df)

# Opisanie przykładowego filmu za pomocą chmury słów
wordcloud_text = ' '.join([','.join(t) for t in sims_df.tags_list])
plt.rcParams["figure.figsize"] = (15,10)
# Wygenerowanie WordCloud
wordcloud = WordCloud(width = 1024, height = 1024, background_color = 'white').generate(wordcloud_text)
plt.figure(figsize=(10,8),facecolor = 'white', edgecolor='blue')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# Wizualizacja wektorów z pomocą narzędzia UMAP

# # redukcja do 2 wymiarów
# doc2vec_movies_embbedings_umap = umap.UMAP(n_components=2, n_neighbors = 10, min_dist = 0.005, metric = 'cosine').fit_transform(doc2vec_movies_embbedings)

# x = doc2vec_movies_embbedings_umap[:,0]
# y = doc2vec_movies_embbedings_umap[:,1]


# fig, ax = plt.subplots(figsize=(15, 10))
# plt.axis('off')
# plt.title('Wektory filmów zredukowane do 2D')
# plt.scatter(x, y, s=1)

# # Wskazanie pozycji wybranych filmów
# ax.annotate("Toy Story",
#             xy=(x[0], y[0]),
#             xytext=(x[0]+1, y[0]+0),
#             arrowprops=dict(facecolor='black', width=0.001, headwidth=4))
# ax.annotate("Toy Story 2",
#             xy=(x[729], y[729]),
#             xytext=(x[729]+1, y[729]+1),
#             arrowprops=dict(facecolor='black', width=0.001, headwidth=4))
# ax.annotate("Shrek",
#             xy=(x[738], y[738]),
#             xytext=(x[738]+0.5, y[738]+1),
#             arrowprops=dict(facecolor='black', width=0.001, headwidth=4))
# ax.annotate("The Shining",
#             xy=(x[7855], y[7855]),
#             xytext=(x[7855]+1, y[7855]+1),
#             arrowprops=dict(facecolor='black', width=0.001, headwidth=4))
# ax.annotate("2001: A Space Odyssey",
#             xy=(x[717], y[717]),
#             xytext=(x[717]+1, y[717]+0),
#             arrowprops=dict(facecolor='black', width=0.001, headwidth=4))
# ax.annotate("Contact",
#             xy=(x[720], y[720]),
#             xytext=(x[720]+1.5, y[720]+0.8),
#             arrowprops=dict(facecolor='black', width=0.001, headwidth=4))
# ax.annotate("Arrival",
#             xy=(x[1367], y[1367]),
#             xytext=(x[1367]+2, y[1367]+2),
#             arrowprops=dict(facecolor='black', width=0.001, headwidth=4))
# ax.annotate("Pretty Woman",
#             xy=(x[477], y[477]),
#             xytext=(x[477]-1, y[477]+1),
#             arrowprops=dict(facecolor='black', width=0.001, headwidth=4))
# ax.annotate("Bridget Jones's Diary",
#             xy=(x[444], y[444]),
#             xytext=(x[444]-2, y[444]+2),
#             arrowprops=dict(facecolor='black', width=0.001, headwidth=4))
# plt.show()


## Tworzenie profili użytkowników poprzez uśrednienie wektorów obejrzanych przez nich filmów

# Wybranie filmów ocenionych wyższej niż 2.0 ( 0.44... po normalizacji) ze zbioru testowego, zakładając, 
# że oceny niższe oznaczają, że użytkownik nie jest zainteresowany danymi filmami
ratings_df_filtered = ratings_df[ratings_df["rating"] > 0.44]

def get_doc2vec_user_vector(user_id):
    """
    Zwraca wektor stanowiący profil użytkownika.
    """        
    user_movies = ratings_df_filtered["movie_id"][ratings_df_filtered["user_id"] == user_id]    
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



# Sprawdzenie top 20 filmów najbardziej podobnych do przykładowego użytkownika
    
# Filmy rekomendowane dla przykładowego użytkownika
example_user_top20 = recommend_items_doc2vec(example_user, 20)
example_user_top20 = pd.DataFrame(example_user_top20,columns=["movie_id", "cosine_similarity"])
example_user_top20.movie_id = example_user_top20.movie_id.astype('float64')
example_user_top20['title'] = example_user_top20['movie_id'].map(movies_df.set_index('movie_id')['title'])
example_user_top20['tags_list'] = example_user_top20['movie_id'].map(movies_df.set_index('movie_id')['tags_list'])

# Filmy najwyżej ocenione przez przykładowego użytkownika
example_user_real_top20 = ratings_df[ratings_df['user_id'] == example_user]
example_user_real_top20 = example_user_real_top20.drop(['user_id', 'old_movie_id', 'timestamp'], axis=1)
example_user_real_top20['title'] = example_user_real_top20['movie_id'].map(movies_df.set_index('movie_id')['title'])
example_user_real_top20['genres'] = example_user_real_top20['movie_id'].map(movies_df.set_index('movie_id')['genres'])
example_user_real_top20['tags_list'] = example_user_real_top20['movie_id'].map(movies_df.set_index('movie_id')['tags_list'])
example_user_real_top20 = example_user_real_top20.sort_values(by = "rating", ascending = False).head(20)    

print("Top 20 filmów obejrzanych przez użytkownika", example_user)
print(example_user_real_top20)
print("Top 20 filmów zarekomendowanych dla użytkownika", example_user)
print(example_user_top20)


# Lista z listami najlepszych filmów dla każdego użytkownika (wg. ocen)
top_ratings_real = ratings_df.groupby('user_id').apply(
    lambda x: np.array(x.sort_values('rating', ascending=False)['movie_id'].head(20)).astype('float64')).tolist()



# Lista z rekomendacjami dla wszystkich użytkowników
doc2vec_top20_recommendations = [recommend_items_doc2vec(user_id, 20) for user_id in ratings_df.user_id.unique()]


# model Doc2Vec OCENA
print("model Doc2Vec  OCENA")
print("MTP: ", r_metrics.mean_true_positives_percentage(top_ratings_real, doc2vec_top20_recommendations))
print("MRR: ", r_metrics.mean_reciprocal_rank(top_ratings_real, doc2vec_top20_recommendations))
print("MAP: ", r_metrics.mean_average_precision(top_ratings_real,doc2vec_top20_recommendations))



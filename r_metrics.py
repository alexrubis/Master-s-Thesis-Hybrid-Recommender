# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 12:21:30 2023

@author: Aleksander Rubis
"""
import numpy as np

def reciprocal_rank(y, y_hat):
    '''
     Funkcja licząca metrykę RR (odwrotność pozycji pierwszego relewantnego przedmiotu).    
      
    Parameters
    ----------
    y : lista filmów polubionych przez użytkownika
    y_hat : lista filmów zarekomendowanych dla użytkownika

    Returns
    -------
    RR : Wartość RR od 0(najgorzej) do 1 (najlepiej).
    '''
    for i, item in enumerate(y_hat):
        if item in y:
            return 1/(i+1)   
    return 0


def mean_reciprocal_rank(y, y_hat):
    '''
     Funkcja licząca metrykę MRR - średnią z wartości RR  (odwrotności pozycji pierwszego relewantnego przedmiotu) 
     obliczonych dla każdego użytkownika w zbiorze.
      
    Parameters
    ----------
    y : lista  list filmów polubionych przez użytkownika
    y_hat : lista list filmów zarekomendowanych dla użytkownika

    Returns
    -------
    MRR : Wartość MRR od 0(najgorzej) do 1 (najlepiej).
    '''
    rr_sum = 0
    for rank_list, relevant_items in zip(y_hat, y):
        rr_sum += reciprocal_rank(relevant_items, rank_list)
        
    MRR = rr_sum / len(y_hat)    
    return MRR


def average_precision(y,y_hat):
    '''
    Funkcja licząca metrykę AP wg. wzoru:
      AP = 1/(liczba filmów w zbiorze Y) * suma precyzji podzbiorów Y_HAT 
      zawierających film polubiony przez użytkownika.
      Np. dla y = [1,2,3,4] i y_hat = [24,3,1,10] AP = (1/4) * (1/2 + 2/3) = 0.2916...
      Funkcja zakłada że zbiory Y i Y_HAT są równej długości.
      
    Parameters
    ----------
    y : lista filmów polubionych przez użytkownika
    y_hat : lista filmów zarekomendowanych dla użytkownika

    Returns
    -------
    AP : Wartość AP od 0(najgorzej) do 1 (najlepiej).

    '''
    #if len(y) != len(y_hat): return None
    
    # Lista binarnych wskaźników czy film MOVIE z Y_HAT istnieje w Y
    relevant = [1 if movie in y else 0 for movie in y_hat]
    # Suma rosnąca o kolejne relewantne filmy
    cumulative = np.cumsum(relevant)
    # Precyzja obliczona dla każdego z kolejnych podzbiorów zbioru Y_HAT
    precisions = [cumulative[i] / (i + 1) for i in range(len(relevant)) if relevant[i] == 1] 
    # Średnia z precyzji podzbiorów
    AP = sum(precisions)/len(y)
    
    return AP


def mean_average_precision(y,y_hat):
    '''
    Funkcja licząca metrykę MAP będącą średnią ze zbioru metryk AP obliczonych dla każdego użytkownika w zbiorze.
      
    Parameters
    ----------
    y : lista list filmów polubionych przez każdego z użytkowników
    y_hat : lista list filmów zarekomendowanych dla każdego z użytkowników

    Returns
    -------
    MAP : Wartość MAP od 0(najgorzej) do 1 (najlepiej).

    '''
    if len(y) != len(y_hat): return None
    
    ap_list = [average_precision(y[i],y_hat[i]) for i in range(len(y))]
    MAP = np.mean(ap_list)
    
    return MAP


def ndcg(ratings_list):
    '''
    Funkcja licząca metrykę NDCG - znormalizowany zdyskontowany kumulatywny zysk.
    
    Parameters
    ----------
    ratings_list : lista z ocenami filmów w !typie numpy.array!

    Returns
    -------
    NDCG : Wartość NDCG od 0 (najgorzej) do 1 (najlepiej).

    '''
    if type(ratings_list) != np.ndarray: print("Argument ratings_list musi być typu numpy_array!"); return None
    # Cumulative Gain
    #cg = sum(y)
    # Discounted Cumulative Gain
    DCG = np.sum(ratings_list / np.log2(np.arange(2, len(ratings_list)+2)))
    # Ideal Discounted Cumulative Gain
    IDCG = np.sum(-np.sort(-ratings_list) / np.log2(np.arange(2, len(ratings_list)+2)))
    # NDCG jako relacja między DCG i IDCG
    NDCG = DCG / IDCG

    return NDCG


def mean_ndcg(ratings_lists_list):
    '''
    Funkcja licząca średnie NDCG -w zbiorze.
    
    Parameters
    ----------
    ratings_lists_list : lista z listami ocen filmów 

    Returns
    -------
    meanNDCG : Wartość średnią  NDCG od 0 (najgorzej) do 1 (najlepiej) w zbiorze.

    '''
    meanNDCG = sum([ndcg(ratings_list) for ratings_list in ratings_lists_list])/len(ratings_lists_list)

    return meanNDCG


def mean_true_positives_percentage(y,y_hat):
    '''
    Funkcja licząca stosunek prawdziwych pozytywów (filmów które użytkownik faktycznie polubił i które zostały mu zarekomendowany) do wszystkich zarekomendowanych filmów
    w listach rekomendacji wszystkich użytkowników.     
    Parameters
    ----------
    y : lista list filmów polubionych przez użytkowników
    y_hat : lista list  filmów zarekomendowanych dla użytkowników

    Returns
    -------
    mean_true_positives_percentage : Wartość średnią  NDCG od 0 (najgorzej) do 1 (najlepiej) w zbiorze.

    '''
    
    mean_true_positives_percentage = np.mean([len(set(y[i]) & set(y_hat[i])) / len(y_hat[i]) for i in range(len(y))])

    return mean_true_positives_percentage


def evaluate_model(list_of_topk_lists_true, list_of_topk_lists_pred):
    '''
    Funkcja obliczająca wszystkie zastosowane metryki dla danego modelu.
    
    Parameters
    ----------
    args : argumenty

    Returns
    -------
    evaluation_df : dataframe z wynikami wszystkich miar

    '''
    
    return 0


def evaluate_models(list_of_topk_lists_true, list_of_topk_lists_pred):
    '''
    Funkcja obliczająca wszystkie zastosowane metryki dla wszystkich modeli.
    
    Parameters
    ----------
    args : argumenty

    Returns
    -------
    evaluation_df : dataframe z wynikami wszystkich miar

    '''
    
    return 0


########## Testy metryk
# ratings_list1 = np.array([2,3,3,1,2])
# ratings_list2 = np.array([3,3,2,2,1])


# print("NDCG@1:", ndcg(ratings_list1))
# print("NDCG@52:", ndcg(ratings_list2))


#Wolniejesze #a = sum([rating/np.log2(index+2) for index, rating in enumerate(ratings_list1)])

#start = time.time()
#for i in range(100000):
#    c = np.sum(ratings_list1 / np.log2(np.arange(2, len(ratings_list1)+2)))
#end2 = time.time() - start

# print("mrr: ", mean_reciprocal_rank(xxxx1, xxxx2))
# print("acc: ", accuracy_score(xxxx1, xxxx2))
# print("preci: ", precision_score(xxxx1, xxxx2, average=None))
# print("rec: ", recall_score(xxxx1, xxxx2, average="micro"))

# from sklearn.metrics import precision_score,recall_score,accuracy_score


# print("MAP:", mean_average_precision([[1,2,3,4], [10,11,12,13], [4,5,6,7]], [[1,2,3,4], [0,10,55,4], [0,0,0,0]]))    

# print(average_precision([1,2,3,4],[1,2,3,4]))
# print(average_precision([10,11,12,13],[0,10,55,4]))
# print(average_precision([4,5,6,7],[0,0,0,0]))
    
# A = [8305, 8681, 8550, 8063, 8466] # IDs of movies user liked
# B = [8509, 8305, 8063, 8550, 8681] # IDs of movies recommended by recommender


# correct = set(A) & set(B) # set of movies that the user liked and were recommended
# total_predicted = len(B) # total number of movies recommended by the recommender
# precision = len(correct) / total_predicted # precision of the recommendation

# print("acc: ", accuracy_score(A,B))

# print("Precision:", precision)
# print("Precision2:", precision_score(A,B, average='micro'))


# rank_lists =          [[1, 2, 3, 4, 5],   [3, 2, 1, 5, 4], [5, 4, 3, 2, 1]]
# relevant_items_list = [[1, 2, 55,66,77] , [1, 2, 3, 4, 5], [2, 11,12,13,14]]
# rank_lists = [[4, 5, 3], [1, 2, 3]]
# relevant_items_list = [[1, 2, 3], [4, 5, 6]]


# mrr = mean_reciprocal_rank(relevant_items_list, rank_lists)
# print("MRR:", mrr)

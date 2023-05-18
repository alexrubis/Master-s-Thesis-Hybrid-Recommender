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

def mean_average_recall(true_positives, top_k_predictions):
    num_users = len(true_positives)
    sum_recall_at_k = 0.0
    
    for i in range(num_users):
        tp = set(true_positives[i])
        pred = set(top_k_predictions[i])
        
        # Obliczanie elementóW wspólnych między prawdziwymi pozytywami i topK predykcjami.
        intersection = tp.intersection(pred)
        
        # Obliczanie  czułości przy K
        recall_at_k = len(intersection) / len(tp)
        
        # Dodawanie czułości przy K do sumy
        sum_recall_at_k += recall_at_k
    
    # Obliczanie średniej czułości przy K
    MAR = sum_recall_at_k / num_users
    
    return MAR

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
    Funkcja licząca stosunek prawdziwych pozytywów (filmów które użytkownik faktycznie polubił i które zostały mu zarekomendowane) do wszystkich zarekomendowanych filmów
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



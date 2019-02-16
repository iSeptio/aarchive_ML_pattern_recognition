# --------------------------------------------------------------------------
# -----------------------  Rozpoznawanie Obrazow  --------------------------
# --------------------------------------------------------------------------
#  Zadanie 4: Zadanie zaliczeniowe
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import pickle as pkl
import numpy as np


"""
Funkcja pobiera macierz przykladow zapisanych w macierzy X o wymiarach NxD i zwraca wektor y o wymiarach Nx1,
gdzie kazdy element jest z zakresu {0, ..., 35} i oznacza znak rozpoznany na danym przykladzie.
:param x: macierz o wymiarach NxD
:return: wektor o wymiarach Nx1
"""

# wczytywanie danych
with open('train.pkl','rb') as file:
    data=pkl.load(file)
    train_x,train_y = data
#koniec wczytywania danych
train_x = train_x[:,range(350,450)]
#def trojnar_features(x):
#    wynik = []
#    for i in range(0,x.shape[0]):
#        xi = x[i,:]
#        a = xi.reshape(56,56).sum(axis=1).tolist()
#        wynik.append(a)
#    return np.array(wynik)
#train_x = trojnar_features(train_x)   
    
    
wynik = []  
for j in range(0,train_x.shape[0]):
    z = [train_x[j,:] == train_x] 
    z = np.array(z.pop())
    wektor_sumobrazow =z.sum(axis=1)
    index = wektor_sumobrazow.argmax()
    wynik.append(int(train_y[index]))
train_y = np.array(wynik)

#zapisywanie danych
with open('trainsave.pkl','wb') as file:
    data=pkl.dump((train_x,train_y),file,protocol=pkl.HIGHEST_PROTOCOL)
#koniec zapisywania danych
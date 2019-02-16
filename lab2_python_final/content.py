# --------------------------------------------------------------------------
# -----------------------  Rozpoznawanie Obrazow  --------------------------
# --------------------------------------------------------------------------
#  Zadanie 2: k-NN i Naive Bayes
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

from __future__ import division
import numpy as np
import scipy
from scipy import spatial




def hamming_distance(X, X_train):
    """

    N1 = X.shape[0]
    N2 = X_train.shape[0]
    Dist = np.zeros(shape=(N1, N2))
    X = X.toarray()
    X_train = X_train.toarray()

    for i in range (0, X.shape[0]):
        for j in range(0, X_train.shape[0]):
            Dist[i, j] = np.sum(np.abs((X[i] & X_train[j]) - (X[i] | X_train[j])))

    return Dist
    """
    X = X.toarray()
    X_train = X_train.toarray()
    Dist = scipy.spatial.distance.cdist(X, X_train, metric='hamming') * X.shape[1]
    return(Dist)
    pass


def sort_train_labels_knn(Dist, y):

    y_sorted=y[np.argsort(Dist,axis=1, kind='mergesort')]
    return(y_sorted)

    pass


def p_y_x_knn(y, k):

    N1 = y.shape[0]
    M = (np.unique(y)).shape[0]
    p_y_x = np.zeros(shape=(N1, M))
    for i in range(0, N1):
        for j in range(0, M):
            p_y_x[i, j] = 1/k * np.sum(y[i,0:k] == j+1)
    return p_y_x
    pass


def classification_error(p_y_x, y_true):

    err = 0
    M = p_y_x.shape[1]
    N = len(y_true)
    for i in range(0,N):
        sorted1 = np.argsort(p_y_x[i,:])
        y = sorted1[M-1]+1
        if y == y_true[i]:
            err += 1
    err = 1 - (err*(1/N))
    return(err)




    pass


def model_selection_knn(Xval, Xtrain, yval, ytrain, k_values):

    #Xval = Xval.astype(int)
    Xtrain = Xtrain.astype(int)
    Dist = hamming_distance(Xval,Xtrain)
    y_sorted = sort_train_labels_knn(Dist, ytrain)
    y_sorted = y_sorted[:,0:max(k_values)]
    errors = np.zeros(shape=(1,len(k_values)))

    best_error = np.inf
    for i in range(len(k_values)):
        p = p_y_x_knn(y_sorted,k_values[i])
        errors[0,i] = classification_error(p, yval)
        if errors[0,i] < best_error:
            best_error = errors[0,i]
            best_k = k_values[i]
    errors = errors.squeeze()
    return (best_error, best_k, errors)
    pass


def estimate_a_priori_nb(ytrain):

    N1 = ytrain.shape[0]
    M = (np.unique(ytrain)).shape[0]
    p_y = np.zeros(shape=(M,1))
    for i in range(0, M):
        p_y[i] =1 / N1 * np.sum(ytrain == i + 1)
    p_y = p_y.transpose()
    return(p_y)
    pass


def estimate_p_x_y_nb(Xtrain, ytrain, a, b):

    Xtrain = Xtrain.toarray()
    D = Xtrain.shape[1]
    M = len(np.unique(ytrain))
    p_x_y = np.zeros(shape=(M, D))
    for i in range(M):
        for j in range(D):
            p_x_y[i, j] = (np.sum((ytrain == i + 1) * (Xtrain[:, j] == 1)) + a - 1) / (np.sum(ytrain == i + 1) + a + b - 2)
    return(p_x_y)

    pass


def p_y_x_nb(p_y, p_x_1_y, X):



    X = X.toarray().astype(int)
    N = X.shape[0]

    M = p_y.squeeze().shape[0]
    p_y = p_y.squeeze()
    p_x_0_y = 1 - p_x_1_y
    p_y_x = np.zeros(shape=(N,M))
    for i in range(0,N):
        for j in range(0,M):
            p_y_x[i,j] = np.prod(p_x_1_y[j,X[i,:]==1])*np.prod(p_x_0_y[j,X[i,:]==0])*p_y[j]
        p_y_x[i] = p_y_x[i]/np.sum(p_y_x[i])
    return(p_y_x)
    pass


def model_selection_nb(Xtrain, Xval, ytrain, yval, a_values, b_values):

    errors = np.zeros(shape=(len(a_values),len(b_values)))
    N = len(a_values)
    error_best = np.inf
    for i in range(0,N):
        for j in range(0,N):
            p_y = estimate_a_priori_nb(ytrain)
            p_x_y = estimate_p_x_y_nb(Xtrain,ytrain,a_values[i],b_values[j])
            p_y_x = p_y_x_nb(p_y,p_x_y,Xval)
            errors[i,j] = classification_error(p_y_x,yval)

            if error_best > errors[i,j]:
                www = i
                wew = j
                error_best = errors[i,j]

    best_b = b_values[wew]
    best_a = a_values[www]
    return(error_best,best_a,best_b,errors)
    pass

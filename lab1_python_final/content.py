# --------------------------------------------------------------------------
# -----------------------  Rozpoznawanie Obrazow  --------------------------
# --------------------------------------------------------------------------
#  Zadanie 1: Regresja liniowa
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import numpy as np
from utils import polynomial


def mean_squared_error(x, y, w):
    '''
    :param x: ciag wejsciowy Nx1
    :param y: ciag wyjsciowy Nx1
    :param w: parametry modelu (M+1)x1
    :return: blad sredniokwadratowy pomiedzy wyjsciami y
    oraz wyjsciami uzyskanymi z wielowamiu o parametrach w dla wejsc x
    '''
    err = np.sum((y - polynomial(x, w)) ** 2)/np.size(y, axis=0)
    return err
    pass


def design_matrix(x_train, M):
    '''
    :param x_train: ciag treningowy Nx1
    :param M: stopien wielomianu 0,1,2,...
    :return: funkcja wylicza Design Matrix Nx(M+1) dla wielomianu rzedu M
    '''
    N = np.size(x_train, axis=0)
    DesignMatrix = []
    for i in range(0, M+1):
        DesignMatrix.append(x_train ** i)

    DesignMatrix = np.reshape(DesignMatrix, (M+1, N))
    DesignMatrix = DesignMatrix.transpose()
    return(DesignMatrix)
    pass


def least_squares(x_train, y_train, M):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :return: funkcja zwraca krotke (w,err), gdzie w sa parametrami dopasowanego wielomianu, a err blad sredniokwadratowy
    dopasowania
    '''
    DesignMatrix = design_matrix(x_train, M)
    w = np.linalg.inv(DesignMatrix.transpose()@
                      DesignMatrix)@DesignMatrix.transpose()@y_train
    err = mean_squared_error(x_train, y_train, w)
    return(w, err)
    pass


def regularized_least_squares(x_train, y_train, M, regularization_lambda):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :param regularization_lambda: parametr regularyzacji
    :return: funkcja zwraca krotke (w,err), gdzie w sa parametrami dopasowanego wielomianu zgodnie z kryterium z regularyzacja l2,
    a err blad sredniokwadratowy dopasowania
    '''
    DesignMatrix = design_matrix(x_train, M)
    w = np.linalg.inv(DesignMatrix.transpose()@DesignMatrix +
                      regularization_lambda*np.eye(M+1))@DesignMatrix.transpose()@y_train
    err = mean_squared_error(x_train, y_train, w)
    return(w, err)
    pass


def model_selection(x_train, y_train, x_val, y_val, M_values):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param x_val: ciag walidacyjny wejscia Nx1
    :param y_val: ciag walidacyjny wyjscia Nx1
    :param M_values: tablica stopni wielomianu, ktore maja byc sprawdzone
    :return: funkcja zwraca krotke (w,train_err,val_err), gdzie w sa parametrami modelu, ktory najlepiej generalizuje dane,
    tj. daje najmniejszy blad na ciagu walidacyjnym, train_err i val_err to bledy na sredniokwadratowe na ciagach treningowym
    i walidacyjnym
    '''
    val_error = []
    for i in range(np.shape(M_values)[0]):
        (w, err) = least_squares(x_train, y_train, i)
        val_error.append(mean_squared_error(x_val, y_val, w))
    val_err = min(val_error)

    for b in range(len(val_error)):
        if val_err == val_error[b]:
            wielomian = b

    (w, train_err) = least_squares(x_train, y_train, wielomian)

    return(w, train_err, val_err)
    pass


def regularized_model_selection(x_train, y_train, x_val, y_val, M, lambda_values):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param x_val: ciag walidacyjny wejscia Nx1
    :param y_val: ciag walidacyjny wyjscia Nx1
    :param M: stopien wielomianu
    :param lambda_values: lista ze wartosciami roznych parametrow regularyzacji
    :return: funkcja zwraca krotke (w,train_err,val_err,regularization_lambda), gdzie w sa parametrami modelu, ktory najlepiej generalizuje dane,
    tj. daje najmniejszy blad na ciagu walidacyjnym. Wielomian dopasowany jest wg kryterium z regularyzacja. train_err i val_err to
    bledy na sredniokwadratowe na ciagach treningowym i walidacyjnym. regularization_lambda to najlepsza wartosc parametru regularyzacji
    '''
    val_error = []
    for i in range(len(lambda_values)):
        (w, err) = regularized_least_squares(
            x_train, y_train, M, lambda_values[i])
        val_error.append(mean_squared_error(x_val, y_val, w))

    val_err = min(val_error)
    for i in range(len(val_error)):
        if val_err == val_error[i]:
            lambda_gdzie = i

    regularization_lambda = lambda_values[lambda_gdzie]
    (w, train_err) = regularized_least_squares(
        x_train, y_train, M, lambda_values[lambda_gdzie])

    return(w, train_err, val_err, regularization_lambda)
    pass

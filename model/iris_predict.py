"""
Biblioteka zawiera funkcje anomaly_detect i iris_predict 

anomaly_detect sprawdza, czy czworka liczb (sepal_length,sepal_width, 
petal_length, petal_width) ma szanse byc dopasowana do gatunku irysa.
Jej dzialanie opiera sie na tescie anomalii przy pomocy nierownosci
Czebyszewa a'la "prawo trzech sigm".

iris_predict wykorzystuje model regresji softmax sluzacy do klasyfikacji 
3 gatunkow irysa (virginica,versicolor, setosa) wytrenowany na podstawie 
zbioru iris-dataset.

iris_predict najpierw wczytuje model z pliku iris_softmax.joblib,
a następnie na podstawie 4 przekazanych parametrow (sepal_length,
sepal_width, petal_length, petal_width):
1) decyduje, czy dane są sensowne, tj. czy zgadza się typ i czy ich
odchylenie od średniej nie jest zbyt duże (detekcja anomalii)
2) zwraca najbardziej prawdopodobny gatunek
3) zwraca wektor prawdopodobienstw nalezenia do danego gatunku
"""

import joblib
import numpy as np

def anomaly_detect(sepal_lenght: float, sepal_width: float, 
                 petal_length: float, petal_width: float)->bool:
    """Sprawdza, czy dane mają szanse na dopasowanie (detekcja anomalii)

    Args:
        sepal_lenght: dlugosc kielicha
        sepal_width: szerokosc kielicha
        petal_length: dlugosc platka
        petal_width: szerokosc platka

    Returns:
        bool: Czy wykryto anomalie (nietypowe dane)
    """
    
    # Srednie kolejnych 4 parametrow z iris-dataset
    m = {
        "SET_MEAN1": 5.006,
        "SET_MEAN2": 3.418,
        "SET_MEAN3": 1.464,
        "SET_MEAN4": 0.244,
        "VIR_MEAN1": 6.588,
        "VIR_MEAN2": 6.588,
        "VIR_MEAN3": 5.552,
        "VIR_MEAN4": 2.026,
        "VER_MEAN1": 5.936,
        "VER_MEAN2": 2.77,
        "VER_MEAN3": 4.26,
        "VER_MEAN4": 1.326,
    }

    # Odchylenia standardowe kolejnych 4 parametrow z iris-dataset
    d = {
        "SET_DEV1": 0.352,
        "SET_DEV2": 0.381,
        "SET_DEV3": 0.173,
        "SET_DEV4": 0.107,
        "VIR_DEV1": 0.635,
        "VIR_DEV2": 0.322,
        "VIR_DEV3": 0.551,
        "VIR_DEV4": 0.274,
        "VER_DEV1": 0.516,
        "VER_DEV2": 0.313,
        "VER_DEV3": 0.469,
        "VER_DEV4": 0.197,
    }

    # Tworzymy tablice numpy
    M = np.array(list(m.values()))
    D = np.array(list(d.values()))
    
    X_feat = np.array([sepal_lenght, sepal_width, petal_length, petal_width])

    # Sprawdzamy, czy chociaz dla jednego kwiata wszystkie parametry są odchylone od
    # średniej o mniej niż k*odchylenie std

    k = 5
    is_reasonable = (np.all(abs(X_feat - M[0:4]) < k*D[0:4]) 
                     or np.all(abs(X_feat - M[4:8]) < k*D[4:8]) 
                     or np.all(abs(X_feat - M[8:12]) < k*D[8:12]))
    
    return not is_reasonable

def iris_predict(sepal_lenght: float, sepal_width: float, 
                 petal_length: float, petal_width: float):
    """Funkcja stosuje regresje softmax dla danych z iris-dataset

    Args:
        sepal_lenght: dlugosc kielicha
        sepal_width: szerokosc kielicha
        petal_length: dlugosc platka
        petal_width: szerokosc platka

    Returns:
        y_class: najbardziej prawdopodobny gatunek
        y_prob: wektor prawdopodobienstw

    Raises:
        TypeError: gdy przekazane parametry nie sa floatami
        Exception: gdy wykryje anomalie w zmiennych

    Example:
        spec, pr = iris_predict(6.4, 2.8, 5.6, 2.2)
    """
    
    # Wczytujemy model
    model = joblib.load("iris_softmax.joblib")

    # Sprawdzamy, czy typy sie zgadzaja
    if (not isinstance(sepal_lenght, (float)) 
        or (not isinstance(sepal_width, (float))) 
        or (not isinstance(petal_length, (float))) 
        or (not isinstance(petal_width, (float)))):
        raise TypeError("Parametry musza byc typu float")

    # Tworzymy wektor cech
    X_feat = np.array([[sepal_lenght, sepal_width, petal_length, petal_width]])

    # Predykcja
    y_class = model.predict(X_feat)
    y_prob = model.predict_proba(X_feat)

    # Zwracamy klase i wektor prawdopodobienstw
    if anomaly_detect(sepal_lenght, sepal_width, petal_length, petal_width):
        raise Exception("Małe szanse poprawnego dopasowania.")
    else:
        return y_class, y_prob


if __name__ == "__main__":
    spec, pr = iris_predict(6.4, 2.8, 5.6, 2.2)
    print(spec)
    print(pr)


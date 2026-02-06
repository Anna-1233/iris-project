"""
Biblioteka zawiera funkcje iris_predict, która wykorzystuje model 
regresji softmax sluzacy do klasyfikacji 3 gatunkow irysa (virginica,
versicolor, setosa) wytrenowany na podstawie zbioru iris-dataset.

iris_predict najpierw wczytuje model z pliku iris_softmax.joblib,
a następnie na podstawie 4 przekazanych parametrow (sepal_length,
sepal_width, petal_length, petal_width):
1) decyduje, czy dane są sensowne, tj. czy zgadza się typ i czy ich
odchylenie od średniej nie jest zbyt duże (detekcja anomalii/prawo 3 sigm) TODO
2) zwraca najbardziej prawdopodobny gatunek
3) zwraca wektor prawdopodobienstw nalezenia do danego gatunku
"""

import joblib
import numpy as np

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

    Example:
        spec, pr = iris_predict(6.4, 2.8, 5.6, 2.2)
    """
    
    # Wczytujemy model
    model = joblib.load("iris_softmax.joblib")

    # Tworzymy wektor cech
    X_feat = np.array([[sepal_lenght, sepal_width, petal_length, petal_width]])

    # Predykcja
    y_class = model.predict(X_feat)
    y_prob = model.predict_proba(X_feat)

    # Zwracamy klase i wektor prawdopodobienstw
    return y_class, y_prob


if __name__ == "__main__":
    spec, pr = iris_predict(6.4, 2.8, 5.6, 2.2)
    print(spec)
    print(pr)

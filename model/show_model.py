"""
Plik do podgladania parametrow modelu
"""

import joblib

model = joblib.load("iris_softmax.joblib")

# Typ modelu
print(type(model))

# Informacje
print(model.get_params())  # parametry trenowania
print(model.classes_)      # nazwy klas
print(model.coef_)         # wagi
print(model.intercept_)    # wektor "biasow"


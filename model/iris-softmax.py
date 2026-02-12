"""
Skrypt do trenowania i zapisywania modelu
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Wczytujemy dane
iris = load_iris()
X = iris.data
y = iris.target

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Skalujemy dane
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model regresji softmax
model = LogisticRegression(
    solver="lbfgs",
    max_iter=200 # Tak naprawde dla tego modelu wagi zbiegaja po kilkunastu iteracjach 
)

# Trenowanie modelu
model.fit(X_train, y_train)

# Predykcja
y_pred = model.predict(X_test)

# Raporty modelu
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Zapis modelu
joblib.dump(model, "iris_softmax.joblib")
joblib.dump(scaler, "scaler.joblib")

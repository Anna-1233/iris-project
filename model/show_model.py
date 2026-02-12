"""
Plik do podgladania parametrow modelu
"""

import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import json



# ładowanie modelu i danych
model = joblib.load("iris_softmax.joblib")
iris = load_iris()

# odtwarzamy ten sam podział danych co w iris-softmax.py
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_pred = model.predict(X_test)

# raport klasyfikacyjny do JSON
report = classification_report(y_test, y_pred, target_names=iris.target_names, output_dict=True)
with open("classification_report.json", "w") as f:
    json.dump(report, f, indent=4)

# macierz pomyłek do pliku PNG
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()


# informacje o modelu do pliku JSON
model_details = {
        "model_type": str(type(model)),         # typ modelu
        "training_params": model.get_params(),  # parametry trenowania
        "classes": model.classes_.tolist(),     # nazwy klas
        "coefficients": model.coef_.tolist(),   # wagi (zamiana wektora NumPy na listę)
        "intercept": model.intercept_.tolist()  # wektor "biasow" (zamiana wektora NumPy na listę)
    }
with open("model_details.json", "w", encoding="utf-8") as f:
    json.dump(model_details, f, indent=4, ensure_ascii=False)

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field, model_validator
from typing import List
import joblib
import numpy as np
import os, json
from api.validator import anomaly_detect


app = FastAPI(
    title="Iris ML API",
    description="Iris Classification based on LogisticRegression algorythm"
)


@app.get("/", include_in_schema=False)
async def root():
    """Redirects the root path to the API documentation."""
    return RedirectResponse(url="/docs")


class IrisItem(BaseModel):
    """Pydantic model for a single Iris flower observation features."""
    
    sepal_length: float = Field(..., gt=0, lt=20)
    sepal_width: float = Field(..., gt=0, lt=20)
    petal_length: float = Field(..., gt=0, lt=20)
    petal_width: float = Field(..., gt=0, lt=20)

    @model_validator(mode='after')
    def check_for_anomalies(self) -> 'IrisItem':
        if anomaly_detect(
                self.sepal_length,
                self.sepal_width,
                self.petal_length,
                self.petal_width
        ):
            # if function anomaly_detect is True -> stop proces
            raise ValueError("Input data is a statistical anomaly and does not match Iris characteristics.")

        return self




# model and evaluation metrics
MODEL_PATH = "model/iris_softmax.joblib"
SCALER_PATH = "model/scaler.joblib"
REPORT_PATH = "model/classification_report.json"
MATRIX_PATH = "model/confusion_matrix.png"
MODEL_DETAILS_PATH = "model/model_details.json"

# threshold for probs -
CONFIDENCE_THRESHOLD = 0.70

# load the pre-trained model and define class names
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    iris_names = ['setosa', 'versicolor', 'virginica']
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


@app.post("/predict", tags=["Prediction"])
async def predict(data: IrisItem):
    """
    Predicts the species of a single Iris flower.

    Args:
        data: An object containing sepal and petal measurements.

    Returns:
        dict: A dictionary containing the predicted species, confidence level,
              and full probability distribution.

    Raises:
        HTTPException: If the model file is not loaded.
    """

    if not model:
        raise HTTPException(status_code=503, detail="Model not available.")

    # convert input data to a 2D NumPy array for prediction
    raw_features = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    features = scaler.transform(raw_features)


    # execute prediction and get probability scores
    preds = model.predict(features)
    probs = model.predict_proba(features)

    max_prob = np.max(probs)

    if max_prob < CONFIDENCE_THRESHOLD:
        return {
            "status": "uncertain",
            "species": None,
            "confidence": f"{np.max(probs[0]) * 100:.2f}%",
            "probabilities": {name: round(float(pr), 3) for name, pr in zip(iris_names, probs[0])},
            "message": "Model confidence is too low. Manual verification required."
        }

    return {
        "status": "success",
        "species": iris_names[int(preds[0])],
        "confidence": f"{np.max(probs[0]) * 100:.2f}%",
        "probabilities": {name: round(float(pr), 3) for name, pr in zip(iris_names, probs[0])}
    }


@app.post("/predict/batch", tags=["Prediction"])
async def predict_many(data: List[IrisItem]):
    """
    Predicts the species for a list of Iris flower observations.

    Args:
        data (List[IrisItem]): A list of objects containing Iris measurements.

    Returns:
        List[dict]: A list of dictionaries with predictions for each observation.

    Raises:
        HTTPException: If the model file is not loaded.
    """

    if not model:
        raise HTTPException(status_code=503, detail="Model not available.")

    raw_features = np.array([[d.sepal_length, d.sepal_width, d.petal_length, d.petal_width] for d in data])
    features = scaler.transform(raw_features)

    preds = model.predict(features)
    probs = model.predict_proba(features)

    results = []
    for i in range(len(preds)):
        max_prob = np.max(probs[i])
        if max_prob < CONFIDENCE_THRESHOLD:
            results.append({
                "status": "uncertain",
                "species": None,
                "confidence": f"{np.max(probs[i]) * 100:.2f}%",
                "probabilities": {name: round(float(pr), 3) for name, pr in zip(iris_names, probs[i])},
                "message": "Model confidence is too low. Manual verification required."
            })
        else:
            results.append({
                "status": "success",
                "species": iris_names[int(preds[i])],
                "confidence": f"{np.max(probs[i]) * 100:.2f}%",
                "probabilities": {name: round(float(pr), 3) for name, pr in zip(iris_names, probs[0])}
            })
    return results


@app.get("/evaluation_metrics/report", tags=["Evaluation metrics"])
async def get_report():
    """
    Retrieves the detailed classification performance metrics for the model in format JSON.

    Returns:
        A dictionary containing performance metrics:

        - species_name (dict): Individual metrics for 'setosa', 'versicolor', and 'virginica'.
            - precision (float): Ability of the classifier not to label a negative sample as positive.
            - recall (float): Ability of the classifier to find all the positive samples.
            - f1-score (float): Harmonic mean of precision and recall.
            - support (int): The number of actual occurrences of the class in the test set.
        - accuracy (float): The overall fraction of correct predictions.
        - macro avg (dict): Arithmetic mean of metrics across all classes.
        - weighted avg (dict): Mean of metrics weighted by the number of samples in each class.
    """

    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH, "r") as f:
            return json.load(f)
    return {"error": "Report file not found."}

@app.get("/evaluation_metrics/matrix", tags=["Evaluation metrics"])
async def get_matrix():
    """
    Retrieves the confusion matrix visualization as an image.

    Returns:
        FileResponse: The confusion matrix PNG file.
    """

    if os.path.exists(MATRIX_PATH):
        return FileResponse(MATRIX_PATH)
    return {"error": "Confusion matrix image not found."}


@app.get("/evaluation_metrics/model_details", tags=["Evaluation metrics"])
async def get_details():
    """
    Retrieves technical data and internal parameters of the trained model in format JSON.

    Returns:
        A dictionary containing the model specification:

            - model_type (str): The class name of the trained model.
            - training_params (dict): Hyperparameters such as regularization (C),
                solver, and multi_class settings.
            - classes (list): List of target species names (0-Setosa, 1-Versicolor, 2-Virginica).
            - coefficients (list): A 2D list of weights assigned to each feature
                per class (from model.coef_).
            - intercept (list): A list of bias values for each class (from model.intercept_).
    """

    if os.path.exists(MODEL_DETAILS_PATH):
        with open(MODEL_DETAILS_PATH, "r") as f:
            return json.load(f)
    return {"error": "Report file not found."}
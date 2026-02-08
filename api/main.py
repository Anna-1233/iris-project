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
FEATURES_PATH = "model/features.png"

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
    Retrieves the model classification report in JSON format.

    Returns:
        dict: The classification report with precision, recall, and f1-scores.
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


@app.get("/evaluation_metrics/features", tags=["Evaluation metrics"])
async def get_features():
    """
    Retrieves the feature importance visualization as an image.

    Returns:
        FileResponse: The feature importance PNG file.
    """

    if os.path.exists(FEATURES_PATH):
        return FileResponse(FEATURES_PATH)
    return {"error": "Feature importance image not found."}
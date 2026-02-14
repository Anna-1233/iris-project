# Iris Species Classification API
A comprehensive Machine Learning API built with **FastAPI** and **Scikit-learn**. 
This project is based on a Logistic Regression model to classify Iris flowers and
implements a secure data pipeline with multi-stage verification to ensure the model
processes only valid, high-quality information.

## Key Features
### 1. Data Integrity Layer (Validation)
* **Technical Validation:** Uses **Pydantic** for type enforcement and range constraints (0-20cm).
* **Statistical Anomaly Detection:** Custom logic (`validator.py`) implements the **k-sigma rule** to detect anomalies. 
Here, an anomaly is an event with probability less than 1/k^2. In the code the threshold is given by the choice k=5.
It cross-references inputs with the Iris dataset's distribution. If a measurement is statistically impossible, 
the API rejects it with a `422 Unprocessable Entity` error.

### 2. Reliable Model Architecture
* **Scaler Integration:** Automated normalization via **StandardScaler** to match training scales.
* **Generalization focus**: Model trained with L2 Regularization to prevent overfitting and 
ensure high accuracy on unseen specimens.

### 3. Confidence-based Decision Logic
* **Reliability Check:** Implements a **70% Confidence Threshold**.
If the model's highest probability is below 70%, 
the API flags these results as uncertain for further review.

---
## API Endpoints
| Endpoint                            | Method    | Input                      | Description                                                               |
|:------------------------------------|:----------|:---------------------------|:--------------------------------------------------------------------------|
| `/predict`                          | **POST**  | Single JSON object         | Predicts the species for a single flower.                                 |
| `/predict/batch`                    | **POST**  | List of JSON objects       | Processes multiple flowers from a list.                                   | 
| `/evaluation_metrics/report`        | **GET**   | None                       | Returns the Classification Report of the trained model.                   |
| `/evaluation_metrics/matrix`        | **GET**   | None                       | Returns the Confusion Matrix of the trained model.                        |
| `/evaluation_metrics/model_details` | **GET**   | None                       | Returns technical metadata and internal parameters of the trained model . |  

## Usage Examples and Testing
Access the interactive API documentation at http://localhost:8000/docs.
Alternatively, requests can be performed directly via curl.

NOTE: Ensure the FastAPI server is running (`uv run fastapi dev api/main.py`) before executing the commands below.

### Swagger UI: 
#### 1. Single Prediction (`/predict`)
* Example Request
    ```
    {
      "sepal_length": 5.1,
      "sepal_width": 3.5,
      "petal_length": 1.4,
      "petal_width": 0.2
    }
    ```
* Example Success Response:
    ```
    {
      "status": "success",
      "species": "setosa",
      "confidence": "97.90%",
      "probabilities": {
        "setosa": 0.979,
        "versicolor": 0.021,
        "virginica": 0
      }
    }
    ```
#### 2. Batch Processing (`/predict/batch`)
* Example Request
    ```
    [
      { "sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2 },
      { "sepal_length": 7.8, "sepal_width": 4.0, "petal_length": 3.2, "petal_width": 2.5 }
    ]
    ```
* Example Success Response:
    ```
    [
      {
        "status": "success",
        "species": "setosa",
        "confidence": "97.90%",
        "probabilities": {
          "setosa": 0.979,
          "versicolor": 0.021,
          "virginica": 0
        }
      },
      {
        "status": "uncertain",
        "species": null,
        "confidence": "59.74%",
        "probabilities": {
          "setosa": 0.01,
          "versicolor": 0.597,
          "virginica": 0.393
        },
        "message": "Model confidence is too low. Manual verification required."
      }
    ]
    ```
### CURL: 
#### 1. Test Single Prediction:
* Example Request
    ```
    curl -X 'POST' \
         'http://localhost:8000/predict' \
         -H 'accept: application/json' \
         -H 'Content-Type: application/json' \
         -d '{
              "sepal_length": 5.1,
              "sepal_width": 3.5,
              "petal_length": 1.4,
              "petal_width": 0.2
             }'
    ```
#### 2. Test Batch Processing:
* Example Request
    ```
    curl -X 'POST' \
         'http://localhost:8000/predict/batch' \
         -H 'accept: application/json' \
         -H 'Content-Type: application/json' \
         -d '[
              {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
              {"sepal_length": 7.8, "sepal_width": 4.0, "petal_length": 3.2, "petal_width": 2.5}
             ]'
    ```

### Automated Batch Testing (CSV) 
The project includes script and sample csv file to allow quick testing batch predictions.
```
uv run python api/test_batch_csv.py
```


## Project Structure

```
iris-project/
├── api/
│   ├── main.py                     # API endpoints and routing logic
│   ├── validator.py                # Statistical anomaly detection logic
│   ├── test_batch_csv.py           # Script for testing batch predictions
│   └── test_data.csv               # Sample data for testing
├── model/                      
│   ├── iris-softmax.joblib         # Trained Logistic Regression model
│   ├── scaler.joblib               # Pre-trained StandardScaler 
│   ├── iris-softmax.py             # Model training script
│   ├── show_model.py               # Model information generator
│   ├── classification_report.json  # Model detailed classification performance metrics
│   ├── confusion_matrix.png        # Confusion matrix visualization
│   └── model_details.json          # Model technical data
├── pyproject.toml              
├── uv.lock                     
├── README.md
└── .gitignore
```

## Getting Started
### 1. Installation
```
git clone https://github.com/Anna-1233/iris-project.git
cd iris-project
uv sync
```

### 2. Running the API
```
uv run fastapi dev api/main.py
```
The API will be available at: http://localhost:8000.

Interactive documentation: http://localhost:8000/docs.

## Interpreting Prediction Results
The API returns a label and also provides a detailed reliability report for every flower analyzed.

The `confidence` percentage represents the model's certainty based on the **Softmax probability distribution**:
* **90 - 100% (High Certainty):** The input data strongly aligns with the typical characteristics of a specific species.

* **70% - 89% (Valid Prediction):** The model identifies a clear winner even if some features may slightly deviate from the "ideal" specimen.

* **Below 70% (Uncertain):** The data falls into a **decision boundary overlap**. 
The system automatically switches the status to `uncertain` and returns message `Model confidence is too low. Manual verification required.`

NOTE: Model rarely returns 100% due to L2 Regularization and Softmax properties, which prioritize generalization over overfitting.

## Evaluation metrics
Key takeaways from the validation phase:

* **Setosa:** Identified with almost 100% accuracy.
* **Versicolor vs. Virginica:** Biological overlaps are managed via the Confidence Threshold.
* **Key Drivers:** Logistic Regression coefficients confirm that **Petal dimensions** (width/length) are the primary predictors.


## Authors
* Rami Ayoush 
* Anna Czopko

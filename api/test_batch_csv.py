import pandas as pd
import requests


URL = "http://127.0.0.1:8000/predict/batch"
CSV_FILE = "test_data.csv"

def test_batch():
    """
    Reads data from CSV and sends it to the API for batch prediction.
    """

    try:
        print(f"Reading data from {CSV_FILE}...")
        df = pd.read_csv(CSV_FILE)

        # convert DataFrame to a list of dictionaries (JSON format) expected by FastAPI
        api_load = df.to_dict(orient="records")

        # send POST request
        print(f"Sending batch request for {len(api_load)} items...")
        response = requests.post(URL, json=api_load)

        # handle response
        if response.status_code == 200:
            results = response.json()
            print("\n--- Predictions Results ---")
            for i, res in enumerate(results):
                status = res.get("status")
                species = res.get("species")
                confidence = res.get("confidence")
                print(f"Row {i + 1} [{status}]: {species} (Confidence: {confidence})")
        else:
            print(f"Error {response.status_code}: {response.text}")

    except FileNotFoundError:
        print(f"Error: File {CSV_FILE} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_batch()
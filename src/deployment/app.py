from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and transformer
model = joblib.load("best_model.pkl")
transformer = joblib.load("transformer.pkl")

# Expected columns
COLUMNS = [
    "CustomerID", "Age", "Gender", "Tenure", "Usage Frequency", 
    "Support Calls", "Payment Delay", "Subscription Type", 
    "Contract Length", "Total Spend", "Last Interaction"
]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    if not isinstance(data, list):
        data = [data]

    try:
        df = pd.DataFrame(data)

        # Ensure required columns exist
        missing_cols = set(COLUMNS) - set(df.columns)
        if missing_cols:
            return jsonify({"error": f"Missing columns: {', '.join(missing_cols)}"}), 400

        # Drop non-feature columns if needed
        X = df[COLUMNS].drop(columns=["CustomerID"])

        # Preprocess
        X_transformed = transformer.transform(X)

        # Predict
        predictions = model.predict(X_transformed)
        probabilities = model.predict_proba(X_transformed)[:, 1]

        # Return results
        result = []
        for i, row in df.iterrows():
            result.append({
                "CustomerID": row["CustomerID"],
                "Churn Prediction": int(predictions[i]),
                "Churn Probability": round(probabilities[i], 4)
            })

        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def health_check():
    return jsonify({"status": "Customer churn model is ready."})

if __name__ == '__main__':
    app.run(debug=True, port=5000)

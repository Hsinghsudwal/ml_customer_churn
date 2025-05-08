# 🧠 Customer Churn Prediction API

This project serves a machine learning model for customer churn prediction using a Flask API. The model and transformer are loaded from pre-exported files and used to return churn predictions and probabilities.

---

## 🚀 Features

- Predicts customer churn from input features
- Returns churn label and churn probability
- Easy-to-use POST `/predict` endpoint
- Dockerized for deployment

---

## 🧾 Required Files

Ensure the following files are in the `deployment_package/` directory:

- `model.pkl`: Trained classifier (e.g., RandomForest, XGBoost)
- `transformer.pkl`: Fitted preprocessor (e.g., `ColumnTransformer` or `Pipeline`)

---

## 🐳 Run with Docker

### 1. Build the Docker image

```bash
docker build -t churn-api .
'''

### 2. Run container
'''bash
docker run -p 5000:5000 churn-api
'''

### 3. Clean-up
'''bash
docker ps  # get container ID
docker stop <container_id>
docker rm <container_id>
'''


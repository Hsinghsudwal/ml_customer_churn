import requests

API_URL = "http://localhost:5000/predict"

# Sample input expected model features (excluding `CustomerID` and `Churn`)
sample_input = {
    # "CustomerID": "C12345",
    "Age": 35,
    "Gender": "Female",
    "Tenure": 12,
    "Usage Frequency": 5,
    "Support Calls": 2,
    "Payment Delay": 0,
    "Subscription Type": "Premium",
    "Contract Length": 12,
    "Total Spend": 1299.99,
    "Last Interaction": "2024-12-01"
}

def test_prediction():
    response = requests.post(API_URL, json=sample_input)

    print("Status Code:", response.status_code)
    try:
        print("Response JSON:", response.json())
    except Exception as e:
        print("Error parsing JSON:", e)
        print("Response Text:", response.text)

# if __name__ == "__main__":
#     test_prediction()

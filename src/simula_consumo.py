import requests
url = "http://127.0.0.1:8000/predict"
params = {
    "age": 49,
    "sex": 1,
    "chest_pain_type": 4,
    "resting_bp_s": 140,
    "cholesterol": 234,
    "fasting_blood_sugar": 0,
    "resting_ecg": 0,
    "max_heart_rate": 140,
    "exercise_angina": 1,
    "oldpeak": 1.0,
    "ST_slope": 2,
}
response = requests.post(url, params=params)
print(response.json())
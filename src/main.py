from fastapi import FastAPI
import joblib
app = FastAPI()
modelo = joblib.load("./resources/pkl/modelo.pkl")

@app.get("/status")
def home():
    return {"mensaje": "API de predicción de deserción activa"}
@app.post("/predict")
def predict(age,sex,chest_pain_type,resting_bp_s,cholesterol,fasting_blood_sugar,resting_ecg,max_heart_rate,exercise_angina,oldpeak,ST_slope):
    datos = [[age,sex,chest_pain_type,resting_bp_s,cholesterol,fasting_blood_sugar,resting_ecg,max_heart_rate,exercise_angina,oldpeak,ST_slope]]
    prediccion = modelo.predict(datos)[0]
    riesgo = "Alto riesgo de enfermedad cardiaca" if prediccion == 1 else "Bajo riesgo de enfermedad cardiaca"
    return {
    "prediccion": int(prediccion),
    "riesgo": riesgo
    }
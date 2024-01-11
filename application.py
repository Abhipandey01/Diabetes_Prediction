
import pickle
from flask import Flask,request , jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

model=pickle.load(open("models/Model_main.pkl",'rb'))
standardScaler_model=pickle.load(open("models/standardScaler.pkl","rb"))

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict", methods=['GET','POST'])
def home():
    if (request.method=="POST"):
        Pregnancies=int(request.form.get("Pregnancies"))
        Glucose=int(request.form.get("Glucose"))
        BloodPressure=int(request.form.get("BloodPressure"))
        SkinThickness =int(request.form.get("SkinThickness"))
        Insulin=int(request.form.get("Insulin"))
        BMI=float(request.form.get("BMI"))
        DiabetesPedigreeFunction=float(request.form.get("DiabetesPedigreeFunction"))
        Age=int(request.form.get("Age"))
        new_data_scaled= standardScaler_model.transform([[Pregnancies,Glucose, BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        result=model.predict(new_data_scaled)

        return render_template("home.html", result=result[0])

    else:
        return render_template("home.html")

if __name__=="__main__":
    app.run(host="0.0.0.0")

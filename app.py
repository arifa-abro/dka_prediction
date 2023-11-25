import streamlit as st
import joblib 
import numpy as np
import time
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

filename = 'finalized_model_grad_boost (2).sav'
loaded_model = pickle.load(open(filename, 'rb'))

# function to receive users' information.
def inputs():
	with st.form(key="diabetes_data"):
		gender_obj = st.selectbox(label="Patient's Gender: ", options=["Male", "Female"])
		gender = 0
		if gender_obj == "Male":
			gender = 0
		else:
			gender = 1
		age = st.number_input(label="Patient's Age: ")
		blood_sugar = st.number_input(label="Patient's Blood Sugar Level(mg/dL): ")
		hco_3 = st.number_input(label="Patient's Bicarbonate Level(mmol/dL): ")
		pH = st.number_input(label="Patient's pH Level: ")
		uk_obj = st.selectbox(label="Patient's Urinary Ketones Level: ", options=["Positive", "Negative"])
		urinary_ketos = 0
		if uk_obj == "Positive":
			urinary_ketos = 1
		else:
			urinary_ketos = -1
		potasium = st.number_input(label="Patient's Potassium Level: ")
		family = st.selectbox(label="Patient's Family History (DKA): ", options=["Yes", "No"])
		fm=0
		if family == "Yes":
			fm = 1
		else:
			fm = 0
		submit = st.form_submit_button("Submit")
		if submit:
			patient_data = [[blood_sugar, hco_3, pH, urinary_ketos,fm,age,gender,potasium]]
			dia_score = predict(patient_data)
			with st.spinner(text="Diagnosing....."):
				time.sleep(5)
			if dia_score == 0:
				st.success("Negative. DKA not diagnosed.")
			else:
				st.error("Positive. DKA Diagnosed.")
		else:
			patient_data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		return patient_data

def predict(var_name):
	score = loaded_model.predict(var_name)
	prob = loaded_model.predict_proba(var_name)
	return score

# function to run streamlit app
def run():
    st.title("DKA Prediction App")
    st.write("Diabetics Keto Acidosis (DKA) is a serious health problem that requires timely intervention to prevent harmful consequences. This study analyzes various types of patient data, including clinical parameters and historical records such as family history, genetics, age, gender, etc., to identify early indications for developing DKA in children. The proposed AI-based system can support healthcare professionals in the emergency department by providing accurate and timely predictions/decisions.")
    info = inputs()

# running streamlit app.
if __name__ == "__main__":
    run()

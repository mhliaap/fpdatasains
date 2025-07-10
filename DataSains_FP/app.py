import streamlit as st
import numpy as np
import pickle

import os

BASE_DIR = os.path.dirname(__file__)
model = pickle.load(open(os.path.join(BASE_DIR, "model_rf.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))

st.title("Prediksi Penyakit Jantung")

# Input
age = st.number_input("Usia", 20, 100, 50)
sex = st.selectbox("Jenis Kelamin", [0, 1])
cp = st.selectbox("Tipe Nyeri Dada (cp)", [0,1,2,3])
trestbps = st.number_input("Tekanan Darah Istirahat", 80, 200, 120)
chol = st.number_input("Kolesterol", 100, 400, 200)
fbs = st.selectbox("FBS > 120?", [0,1])
restecg = st.selectbox("Rest ECG", [0,1,2])
thalach = st.number_input("Detak Jantung Maks", 70, 210, 150)
exang = st.selectbox("Angina karena Olahraga", [0,1])
oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
slope = st.selectbox("Kemiringan ST", [0,1,2])
ca = st.selectbox("Jumlah Pembuluh Warna", [0,1,2,3])
thal = st.selectbox("Thal", [0,1,2,3])

# Prediksi
if st.button("Prediksi"):
    data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                      exang, oldpeak, slope, ca, thal]])
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    
    if prediction[0] == 1:
        st.error("Pasien berisiko penyakit jantung.")
    else:
        st.success("Pasien tidak berisiko penyakit jantung.")


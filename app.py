import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load Model
model = joblib.load('student_model.sav')

# Konfigurasi Halaman
st.set_page_config(page_title="Prediksi Kelulusan Mahasiswa", page_icon="🎓")

st.title("🎓 Prediksi Status Mahasiswa")
st.write("Menggunakan Algoritma: **Logistic Regression**")
st.markdown("Aplikasi ini memprediksi apakah mahasiswa akan **Lulus (Graduate)**, **Masih Terdaftar (Enrolled)**, atau **Drop Out**.")

st.sidebar.header("Masukkan Data Mahasiswa")

# Fungsi input user
def user_input_features():
    # Input Nilai Semester 1 (Range biasanya 0-20 di dataset ini)
    grade_sem1 = st.sidebar.number_input('Nilai Semester 1 (0-20)', min_value=0.0, max_value=20.0, value=12.0)
    
    # Input Nilai Semester 2
    grade_sem2 = st.sidebar.number_input('Nilai Semester 2 (0-20)', min_value=0.0, max_value=20.0, value=12.0)
    
    # Umur
    age = st.sidebar.slider('Umur saat mendaftar', 17, 70, 20)
    
    # Beasiswa
    scholarship = st.sidebar.selectbox('Penerima Beasiswa?', ('Tidak', 'Ya'))
    scholarship_val = 1 if scholarship == 'Ya' else 0
    
    # Uang Kuliah
    tuition = st.sidebar.selectbox('Pembayaran SPP Lancar?', ('Tidak', 'Ya'))
    tuition_val = 1 if tuition == 'Ya' else 0
    
    # Buat DataFrame sesuai urutan training
    data = {
        'Curricular units 1st sem (grade)': grade_sem1,
        'Curricular units 2nd sem (grade)': grade_sem2,
        'Age at enrollment': age,
        'Scholarship holder': scholarship_val,
        'Tuition fees up to date': tuition_val
    }
    return pd.DataFrame(data, index=[0])

# Tampilkan Input
input_df = user_input_features()

st.subheader("Data yang Anda Masukkan:")
st.write(input_df)

# Tombol Prediksi
if st.button('Prediksi Status'):
    prediction = model.predict(input_df)
    result = prediction[0]
    
    st.subheader("Hasil Prediksi:")
    
    if result == 'Dropout':
        st.error(f"Prediksi: **{result}** (Berisiko Putus Sekolah)")
    elif result == 'Enrolled':
        st.warning(f"Prediksi: **{result}** (Masih Terdaftar)")
    else:
        st.success(f"Prediksi: **{result}** (Kemungkinan Lulus)")

    # Menampilkan probabilitas
    proba = model.predict_proba(input_df)
    st.write(f"Confidence Level: {np.max(proba)*100:.2f}%")

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Sistem Deteksi Dini Dropout",
    page_icon="🎓",
    layout="centered"
)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        return joblib.load('dropout_predictor_model.sav')
    except:
        return None

model = load_model()

# --- FUNGSI POP-UP HASIL (DIALOG) ---
@st.dialog("Hasil Analisis Risiko")
def show_prediction_popup(prediction, probability):
    risk_prob = probability[0][1] * 100  # Probabilitas Dropout
    
    if prediction == 1:
        # Tampilan jika BERISIKO (Merah)
        st.error("⚠️ **PERINGATAN: Mahasiswa ini BERISIKO DROPOUT!**", icon="🚨")
        st.write("Berdasarkan data akademik dan finansial, mahasiswa ini memiliki indikasi kuat untuk putus studi.")
        
        # Tampilkan metrik probabilitas dengan warna merah
        st.metric(label="Tingkat Kemungkinan Dropout", value=f"{risk_prob:.1f}%", delta="Tinggi", delta_color="inverse")
        
        st.markdown("---")
        st.info("Segera jadwalkan sesi konseling akademik.", icon="ℹ️")
        
    else:
        # Tampilan jika AMAN (Hijau)
        st.success("✅ **STATUS AMAN: Mahasiswa Diprediksi Lanjut Studi**", icon="🎉")
        st.write("Mahasiswa memiliki performa yang baik untuk melanjutkan studi.")
        
        # Tampilkan metrik keyakinan (100 - risiko dropout)
        safe_prob = 100 - risk_prob
        st.metric(label="Tingkat Keyakinan Lanjut", value=f"{safe_prob:.1f}%", delta="Aman")
        
        st.markdown("---")
        st.caption("Pertahankan prestasi akademik.")

    if st.button("Tutup"):
        st.rerun()

# --- UI APLIKASI ---
st.title("🎓 Prediksi Risiko Dropout Mahasiswa")
st.markdown("---")

if model is None:
    st.error("Model tidak ditemukan. Jalankan 'train_dropout.py' dulu.")
    st.stop()

# --- FORM INPUT ---
with st.form("input_form"):
    st.subheader("📋 Masukkan Data Mahasiswa")
    
    col1, col2 = st.columns(2)
    
    with col1:
        grade_sem1 = st.number_input('Nilai Rata-rata Sem 1 (0-20)', 0.0, 20.0, 12.0)
        age = st.slider('Umur saat mendaftar', 17, 60, 20)
        scholarship = st.selectbox('Penerima Beasiswa?', ['Tidak', 'Ya'])
    
    with col2:
        grade_sem2 = st.number_input('Nilai Rata-rata Sem 2 (0-20)', 0.0, 20.0, 12.0)
        tuition = st.selectbox('Pembayaran SPP Lancar?', ['Tidak', 'Ya'])

    # Tombol Submit
    submitted = st.form_submit_button("🔍 Analisis Risiko", type="primary")

if submitted:
    # Konversi input ke format model
    scholarship_val = 1 if scholarship == 'Ya' else 0
    tuition_val = 1 if tuition == 'Ya' else 0
    
    input_df = pd.DataFrame({
        'Curricular units 1st sem (grade)': [grade_sem1],
        'Curricular units 2nd sem (grade)': [grade_sem2],
        'Age at enrollment': [age],
        'Scholarship holder': [scholarship_val],
        'Tuition fees up to date': [tuition_val]
    })
    
    # Prediksi
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)
    
    # Tampilkan Pop-up
    show_prediction_popup(pred, proba)

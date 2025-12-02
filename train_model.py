import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# --- 1. LOAD DATA YANG BENAR ---
# Kita langsung baca dengan delimiter koma
df = pd.read_csv('students_dropout_academic_success.csv', delimiter=',')

# --- 2. PREPROCESSING ---
# Target di dataset kamu bernama 'target' (huruf kecil)
# Kita ubah target menjadi Binary: 1 untuk Dropout, 0 untuk lainnya.
df['Binary_Target'] = df['target'].apply(lambda x: 1 if x == 'Dropout' else 0)

# Seleksi Fitur
selected_features = [
    'Curricular units 1st sem (grade)', 
    'Curricular units 2nd sem (grade)', 
    'Age at enrollment', 
    'Scholarship holder', 
    'Tuition fees up to date'
]

X = df[selected_features]
y = df['Binary_Target']

# --- 3. SPLIT DATA ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 4. TRAINING MODEL ---
log_reg = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
log_reg.fit(X_train, y_train)

# --- 5. EVALUASI ---
y_pred = log_reg.predict(X_test)
print("=== Evaluasi Model Prediksi Dropout ===")
print(f"Akurasi: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# --- 6. SIMPAN MODEL ---
model_filename = 'dropout_predictor_model.sav'
joblib.dump(log_reg, model_filename)
print(f"\nModel berhasil disimpan sebagai '{model_filename}'")

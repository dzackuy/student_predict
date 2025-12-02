import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load Data
# Pastikan delimiter sesuai (dataset ini biasanya menggunakan ';')
try:
    df = pd.read_csv('students_dropout_academic_success.csv', delimiter=';')
except:
    df = pd.read_csv('students_dropout_academic_success.csv')

# 2. Seleksi Fitur (Kita ambil fitur yang paling berpengaruh untuk demo)
# Fitur: Nilai Sem 1, Nilai Sem 2, Umur, Status Beasiswa, Pembayaran SPP
selected_features = [
    'Curricular units 1st sem (grade)',
    'Curricular units 2nd sem (grade)',
    'Age at enrollment',
    'Scholarship holder',
    'Tuition fees up to date'
]

target = 'Target'

# Pastikan tidak ada missing value
df = df.dropna()

X = df[selected_features]
y = df[target]

# Encoding Target (Mapping ke angka agar bisa diproses)
# Logistic Regression butuh angka, tapi library sklearn otomatis menangani string di y,
# Namun untuk kerapian deployment nanti, kita map manual outputnya di app.py.

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Training Model dengan LOGISTIC REGRESSION
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# 5. Evaluasi
y_pred = log_reg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi Model Logistic Regression: {accuracy * 100:.2f}%")
print("\nLaporan Klasifikasi:\n", classification_report(y_test, y_pred))

# 6. Simpan Model
joblib.dump(log_reg, 'student_model.sav')
print("Model berhasil disimpan sebagai 'student_model.sav'")

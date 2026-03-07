# Sistem Prediksi Selada 🥬

Aplikasi web berbasis Streamlit untuk memprediksi distribusi penjualan dan jumlah tanam selada menggunakan Machine Learning.

## Fitur

### 1. Prediksi Distribusi Penjualan
- Memprediksi distribusi penjualan ke berbagai saluran: Rumah Sendiri, Rt 1-7, dan Pasar
- Menggunakan fitur: Jumlah Tanam, Data Stok, Data Transaksi
- Mendukung 3 model: Random Forest, SVM, dan Voting Regressor

### 2. Prediksi Jumlah Tanam
- Memprediksi jumlah tanam yang optimal
- Menggunakan fitur: Data Stok, Data Transaksi, Sudah Packing, Stok Belum Panen
- Mendukung 3 model: Random Forest, SVM, dan Voting Regressor

### 3. LIME Explanation
- Penjelasan visual tentang kontribusi setiap fitur terhadap prediksi
- Membantu memahami keputusan model

## Instalasi

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Cara Menjalankan

Jalankan aplikasi dengan perintah:
```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser pada alamat: `http://localhost:8501`

## Format Data

Upload file CSV dengan delimiter `;` (semicolon) yang berisi kolom-kolom berikut:

### Untuk Distribusi Penjualan:
- Periode (opsional)
- Jumlah Tanam
- Data Stok
- Data Transaksi
- Rumah Sendiri
- Rt 1, Rt 2, Rt 3, Rt 4, Rt 5, Rt 6, Rt 7
- Pasar

### Untuk Jumlah Tanam:
- Periode (opsional)
- Data Stok
- Data Transaksi
- Sudah Packing
- Stok Belum Panen
- Jumlah Tanam

## Cara Menggunakan

1. **Pilih Menu**: Pilih antara "Distribusi Penjualan" atau "Jumlah Tanam" di sidebar
2. **Upload Data**: Upload file CSV dengan data historis
3. **Prediksi**: 
   - Masukkan nilai-nilai fitur
   - Pilih model yang ingin digunakan
   - Klik tombol "🚀 Prediksi"
4. **Training Model**: Klik tab "📈 Model Training" untuk melatih dan mengevaluasi semua model
5. **LIME Explanation**: Klik tab "💡 LIME Explanation" untuk melihat penjelasan prediksi

## Model Machine Learning

- **Random Forest**: Ensemble model berbasis decision tree
- **SVM (Support Vector Machine)**: Model regresi dengan kernel RBF
- **Voting Regressor**: Kombinasi dari Random Forest dan SVM

## Metrik Evaluasi

- **MSE (Mean Squared Error)**: Rata-rata kuadrat error
- **MAE (Mean Absolute Error)**: Rata-rata absolut error
- **MAPE (Mean Absolute Percentage Error)**: Rata-rata persentase error

## Teknologi

- Python 3.8+
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib/Seaborn
- LIME

## Catatan

- Pastikan format data CSV sesuai dengan yang dijelaskan di atas
- Model akan dilatih ulang setiap kali melakukan prediksi untuk hasil yang optimal
- Untuk dataset besar, proses training mungkin memakan waktu beberapa menit

# Study Delay Prediction System

A web platform for predicting student study delay risks using Random Forest machine learning algorithm.

## Description

This system helps identify students who are potentially at risk of study delays based on academic data such as Total TAK (Total Credit Points), Number of CO (Cut Off), and GPA per semester. This application is built for private higher education institutions in Indonesia.

## Main Features

- **Manual Input**: Individual prediction for one student.
- **CSV Batch Upload**: Mass prediction from CSV file.
- **Visualization Dashboard**: Interactive charts and tables for data analysis.
- **Machine Learning Models**: 4 separate models for semesters 3-6 with high accuracy.

## Technologies Used

- **Backend**: Python Flask
- **Machine Learning**: Scikit-learn (Random Forest), Joblib for models
- **Frontend**: HTML, CSS (Tailwind CSS), JavaScript, Chart.js
- **Data Processing**: Pandas, NumPy

## Installation

1. Ensure Python 3.7+ is installed.
2. Clone this repository.
3. Install dependencies:
   ```
   pip install flask pandas numpy scikit-learn joblib
   ```
4. Run the application:
   ```
   python app.py
   ```
5. Access in browser: `http://localhost:5000`

## File Structure

- `app.py`: Flask backend with routes and prediction logic.
- `templates/index.html`: Main frontend.
- `static/`: CSS, JS, images.
- `model_*.pkl`: Trained ML models.
- `scaler_*.pkl`: Scalers for preprocessing.

## Usage

### Manual Input
1. Select prediction semester.
2. Fill student data.
3. Click "Predict" for results.

### CSV Upload
1. Download CSV template.
2. Fill student data according to format.
3. Upload file and view results in table/dashboard.

### Dashboard
- View visualizations of status distribution, TAK, CO, GPA.
- Filter by semester and status.

## CSV Format

Required headers: Student Name, NIM, Total TAK, Number of CO, GPA1–GPA6.
Numeric values can use dot or comma.

## License

Created for academic and institutional purposes.

## Contact

For questions, contact the developer.

---

# Sistem Prediksi Keterlambatan Studi

Platform web untuk memprediksi risiko keterlambatan studi mahasiswa menggunakan algoritma machine learning Random Forest.

## Deskripsi

Sistem ini membantu mengidentifikasi mahasiswa yang berpotensi mengalami keterlambatan studi berdasarkan data akademik seperti Total TAK (Total Angka Kredit), Jumlah CO (Cut Off), dan IPK per semester. Aplikasi ini dibangun untuk institusi pendidikan tinggi swasta di Indonesia.

## Fitur Utama

- **Input Manual**: Prediksi individu untuk satu mahasiswa.
- **Upload CSV Batch**: Prediksi massal dari file CSV.
- **Dashboard Visualisasi**: Grafik dan tabel interaktif untuk analisis data.
- **Model Machine Learning**: 4 model terpisah untuk semester 3-6 dengan akurasi tinggi.

## Teknologi yang Digunakan

- **Backend**: Python Flask
- **Machine Learning**: Scikit-learn (Random Forest), Joblib untuk model
- **Frontend**: HTML, CSS (Tailwind CSS), JavaScript, Chart.js
- **Data Processing**: Pandas, NumPy

## Instalasi

1. Pastikan Python 3.7+ terinstall.
2. Clone repository ini.
3. Install dependencies:
   ```
   pip install flask pandas numpy scikit-learn joblib
   ```
4. Jalankan aplikasi:
   ```
   python app.py
   ```
5. Akses di browser: `http://localhost:5000`

## Struktur File

- `app.py`: Backend Flask dengan routes dan logika prediksi.
- `templates/index.html`: Frontend utama.
- `static/`: CSS, JS, gambar.
- `model_*.pkl`: Model ML terlatih.
- `scaler_*.pkl`: Scaler untuk preprocessing.

## Penggunaan

### Input Manual
1. Pilih semester prediksi.
2. Isi data mahasiswa.
3. Klik "Prediksi" untuk hasil.

### Upload CSV
1. Download template CSV.
2. Isi data mahasiswa sesuai format.
3. Upload file dan lihat hasil di tabel/dashboard.

### Dashboard
- Lihat visualisasi distribusi status, TAK, CO, IPK.
- Filter berdasarkan semester dan status.

## Format CSV

Header wajib: Nama Mahasiswa, NIM, Total TAK, Jumlah CO, IPK1–IPK6.
Nilai numerik bisa pakai titik atau koma.

## Lisensi

Dibuat untuk keperluan akademik dan institusional.

## Kontak

Untuk pertanyaan, hubungi pengembang.

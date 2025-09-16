# app.py
from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import pandas as pd
import joblib
import io
import math

app = Flask(__name__)

# ============== Helpers ==============

def to_float(val, default=0.0):
    """Ubah string '3,64' -> 3.64, aman untuk None/NaN; kalau gagal kembalikan default."""
    if val is None:
        return default
    # Pandas NaN check
    try:
        if pd.isna(val):
            return default
    except Exception:
        pass
    # String dengan koma
    if isinstance(val, str):
        val = val.strip().replace(",", ".")
    try:
        return float(val)
    except Exception:
        return default

def series_to_float(s, default=np.nan):
    """Konversi sebuah Series ke float dengan dukungan koma desimal."""
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce").fillna(default)

# ============== Load Semua Model & Scaler ==============

models = {
    'sem3': joblib.load('model_sem3.pkl'),
    'sem4': joblib.load('model_sem4.pkl'),
    'sem5': joblib.load('model_sem5.pkl'),
    'sem6': joblib.load('model_sem6.pkl')
}

scalers_minmax = {
    'sem3': joblib.load('scaler_minmax_sem3.pkl'),
    'sem4': joblib.load('scaler_minmax_sem4.pkl'),
    'sem5': joblib.load('scaler_minmax_sem5.pkl'),
    'sem6': joblib.load('scaler_minmax_sem6.pkl')
}

scalers_standard = {
    'sem3': joblib.load('scaler_standard_sem3.pkl'),
    'sem4': joblib.load('scaler_standard_sem4.pkl'),
    'sem5': joblib.load('scaler_standard_sem5.pkl'),
    'sem6': joblib.load('scaler_standard_sem6.pkl')
}

# ============== Routes ==============

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({'status': False, 'message': 'Invalid input'}), 400

    try:
        sem_int = int(to_float(data.get('semester', 0)))
        semester = f"sem{sem_int}"
        model = models[semester]
        scaler_minmax = scalers_minmax[semester]
        scaler_standard = scalers_standard[semester]

        total_tak = to_float(data.get('total_tak', 0))
        jumlah_co  = to_float(data.get('jumlah_co', 0))

        # ambil ipk1..ipk_sem
        ipk_list = [to_float(data.get(f'ipk{i}', 0)) for i in range(1, sem_int + 1)]
        # antisipasi semua 0 -> min/max tetap 0
        min_ipk = min(ipk_list) if len(ipk_list) else 0.0
        max_ipk = max(ipk_list) if len(ipk_list) else 0.0

        final_data = np.array([[total_tak, jumlah_co] + ipk_list + [min_ipk, max_ipk]])
        part_minmax = final_data[:, :2]
        part_standard = final_data[:, 2:]

        scaled_minmax = scaler_minmax.transform(part_minmax)
        scaled_standard = scaler_standard.transform(part_standard)
        final_scaled = np.hstack([scaled_minmax, scaled_standard])

        pred = model.predict(final_scaled)[0]
        prob = model.predict_proba(final_scaled)[0]

        status = "Tepat Waktu" if pred == 0 else "Terlambat"
        confidence = f"{round(float(prob[pred]) * 100, 2)}%"

        return jsonify({'status': status, 'confidence': confidence})

    except Exception as e:
        return jsonify({'status': False, 'message': str(e)}), 500

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    if 'file' not in request.files:
        return jsonify({'status': False, 'message': 'CSV file missing'}), 400

    try:
        file = request.files['file']
        df = pd.read_csv(file)

        # --- Normalisasi kolom numerik (dukung koma desimal) ---
        if 'Total TAK' in df.columns:
            df['Total TAK'] = series_to_float(df['Total TAK'], default=0.0)
        if 'Jumlah CO' in df.columns:
            df['Jumlah CO'] = series_to_float(df['Jumlah CO'], default=0.0)
        for i in range(1, 7):
            col = f'IPK{i}'
            if col in df.columns:
                df[col] = series_to_float(df[col], default=np.nan)

        # --- Siapkan hasil ---
        results = []

        for _, row in df.iterrows():
            # Hitung jumlah semester dari IPK yang tidak NaN
            ipk_available = []
            for i in range(1, 7):
                col = f'IPK{i}'
                val = row.get(col, np.nan)
                if pd.notna(val):
                    ipk_available.append(i)

            semester_count = len(ipk_available)
            semester_key = f"sem{semester_count}"

            # Ambil fitur utama
            total_tak = to_float(row.get('Total TAK', 0))
            jumlah_co  = to_float(row.get('Jumlah CO', 0))
            ipk_values = [to_float(row.get(f'IPK{i}', 0)) for i in range(1, semester_count + 1)]

            # Kalau semester tidak termasuk model yang tersedia
            if semester_key not in models or semester_count < 3 or semester_count > 6:
                # tetap kirim ipk1..ipk6 ke frontend agar grafik punya data
                result_row = {
                    'nama': row.get('Nama Mahasiswa', ''),
                    'nim':  row.get('NIM', ''),
                    'semester': semester_count,
                    'tak': total_tak,
                    'co': jumlah_co,
                    'status': 'Invalid Semester',
                    'confidence': '-'
                }
                for i in range(1, 7):
                    result_row[f'ipk{i}'] = to_float(row.get(f'IPK{i}', np.nan), default=0.0)
                results.append(result_row)
                continue

            # Siapkan scaler & model
            model = models[semester_key]
            scaler_minmax = scalers_minmax[semester_key]
            scaler_standard = scalers_standard[semester_key]

            # Fitur min/max IPK
            if len(ipk_values) == 0:
                min_ipk = 0.0
                max_ipk = 0.0
            else:
                min_ipk = min(ipk_values)
                max_ipk = max(ipk_values)

            data = np.array([[total_tak, jumlah_co] + ipk_values + [min_ipk, max_ipk]])

            # Scaling bagian sesuai pipeline pelatihan
            scaled_minmax = scaler_minmax.transform(data[:, :2])
            scaled_standard = scaler_standard.transform(data[:, 2:])
            final_scaled = np.hstack([scaled_minmax, scaled_standard])

            # Prediksi
            pred = model.predict(final_scaled)[0]
            prob = model.predict_proba(final_scaled)[0]

            # Susun hasil (lengkap dengan ipk1..ipk6 supaya chart bisa baca)
            result_row = {
                'nama': row.get('Nama Mahasiswa', ''),
                'nim':  row.get('NIM', ''),
                'semester': semester_count,
                'tak': total_tak,
                'co': jumlah_co,
                'status': "Tepat Waktu" if pred == 0 else "Terlambat",
                'confidence': f"{round(float(prob[pred]) * 100, 2)}%"
            }
            for i in range(1, 7):
                result_row[f'ipk{i}'] = to_float(row.get(f'IPK{i}', np.nan), default=0.0)

            results.append(result_row)

        # === Download Mode ===
        if request.args.get('download') == 'true':
            df_result = pd.DataFrame(results)
            output = io.StringIO()
            df_result.to_csv(output, index=False)
            output.seek(0)
            return send_file(
                io.BytesIO(output.getvalue().encode()),
                mimetype='text/csv',
                as_attachment=True,
                download_name='hasil_prediksi_batch.csv'
            )

        # === Tampilkan di UI Mode ===
        return jsonify({'status': True, 'results': results})

    except Exception as e:
        return jsonify({'status': False, 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

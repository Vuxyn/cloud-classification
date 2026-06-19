# Cloud Image Classification System

Sistem klasifikasi citra awan berbasis pemrosesan citra digital tradisional (preprocessing manual, ekstraksi fitur warna dan tekstur, serta klasifikasi menggunakan Random Forest, SVM, dan KNN).

## Contoh Dataset (Dataset Examples)

Dataset Ground-based Cloud Dataset (GCD) memiliki 7 kelas awan. Berikut adalah contoh visualisasi dari masing-masing kelas:

| Kelas | Citra Contoh | Deskripsi |
| :--- | :---: | :--- |
| **1. Cumulus** | ![Cumulus](results/figures/dataset_examples/cumulus.jpg) | Awan rendah yang tebal bergumpal seperti kapas dengan batas yang jelas. |
| **2. Altocumulus** | ![Altocumulus](results/figures/dataset_examples/altocumulus.jpg) | Awan menengah berbentuk bulatan kecil bergumpal atau bergelombang abu-abu/putih. |
| **3. Cirrus** | ![Cirrus](results/figures/dataset_examples/cirrus.jpg) | Awan tinggi yang tipis dan lembut seperti serat atau bulu burung. |
| **4. Clear Sky** | ![Clear Sky](results/figures/dataset_examples/clearsky.jpg) | Langit cerah tanpa awan atau awan sangat minim (didominasi warna biru). |
| **5. Stratocumulus** | ![Stratocumulus](results/figures/dataset_examples/stratocumulus.jpg) | Awan rendah berupa lembaran atau lapisan kasar bergelombang abu-abu/putih. |
| **6. Cumulonimbus** | ![Cumulonimbus](results/figures/dataset_examples/cumulonimbus.jpg) | Awan mendung/badai tebal yang menjulang tinggi, berpotensi hujan dan petir. |
| **7. Mixed** | ![Mixed](results/figures/dataset_examples/mixed.jpg) | Kondisi langit campuran yang memiliki beberapa jenis awan sekaligus. |

## Tools dan Dependensi

### Persyaratan Sistem
- Python 3.10 atau versi lebih baru
- Dataset Ground-based Cloud Dataset (GCD)

### Dependensi Python
Instalasi pustaka yang diperlukan:
```cmd
python -m pip install numpy pandas opencv-python scikit-image scikit-learn scipy matplotlib seaborn tqdm jupyter python-dotenv
```

Untuk akselerasi GPU (opsional, memerlukan NVIDIA GPU dan CUDA Toolkit):
```cmd
python -m pip install cupy-cuda12x
```

## Langkah-Langkah (Steps)

### 1. Konfigurasi Dataset
Buat berkas `.env` pada direktori root proyek dengan menyalin berkas `.env.example`:
```cmd
copy .env.example .env
```
Buka berkas `.env` dan atur path direktori dataset Anda:
```env
DATASET_ROOT=D:/path/ke/GCD
```

### 2. Struktur Direktori Proyek
Pastikan struktur direktori utama adalah sebagai berikut:
```
.
├── dataset/
├── notebooks/
│   ├── 00_baseline.ipynb
│   ├── 01_experiment1.ipynb
│   ├── 02_experiment2.ipynb
│   ├── 03_experiment3.ipynb
│   ├── 04_experiment4.ipynb
│   ├── 05_experiment5.ipynb
│   ├── 06_experiment6.ipynb
│   ├── 07_experiment7.ipynb
│   └── 08_experiment8.ipynb
├── results/
│   ├── figures/
│   └── metrics.csv
├── src/
│   ├── image_processing.py
│   ├── loader.py
│   └── generate_metrics_table.py
├── .env
└── README.md
```

### 3. Menjalankan Notebook Eksperimen
Jalankan berkas notebook di dalam folder `notebooks/` secara berurutan. Setiap notebook merepresentasikan eksperimen dengan konfigurasi preprocessing dan ekstraksi fitur yang berbeda:

- **Baseline (`00_baseline.ipynb`)**: Tanpa preprocessing tambahan (hanya grayscale + resize). Fitur: GLCM.
- **Experiment 1 (`01_experiment1.ipynb`)**: Menambahkan statistik warna HSV (Mean, Std Dev, Skewness, Kurtosis) langsung dari gambar asli dikombinasikan dengan GLCM.
- **Experiment 2 (`02_experiment2.ipynb`)**: Menambahkan fitur warna NRBR (Normalized Red-Blue Ratio) serta histogram & statistik HSV + GLCM.
- **Experiment 3 (`03_experiment3.ipynb`)**: Fitur histogram intensitas keabuan (16 bin) dan statistik orde pertama (mean, std, skewness, kurtosis) + GLCM (tanpa fitur warna).
- **Experiment 4 (`04_experiment4.ipynb`)**: Fitur histogram HSV (32 bin) dan statistik warna orde pertama + GLCM.
- **Experiment 5 (`05_experiment5.ipynb`)**: Preprocessing Enhancement (CLAHE pada channel S dan V ruang warna HSV + Gaussian Blur) + Fitur HSV Histogram & Stats + GLCM.
- **Experiment 6 (`06_experiment6.ipynb`)**: Preprocessing Pipeline: Normalisasi Min-Max + CLAHE pada channel S dan V + Gamma Correction + Fitur HSV Histogram & Stats + GLCM.
- **Experiment 7 (`07_experiment7.ipynb`)**: Preprocessing Pipeline lengkap: Normalisasi Min-Max + CLAHE pada channel S dan V + Gaussian Blur + Gamma Correction + Fitur HSV Histogram & Stats + GLCM.
- **Experiment 8 (`08_experiment8.ipynb`)**: Preprocessing bertahap: Gray World White Balance + Bilateral Filter + CLAHE + Fitur Statistik Warna RGB (Mean, Std, Skewness) & GLCM + Feature Selection (korelasi >= 0.95).

Setiap kali notebook selesai dijalankan, hasil evaluasi model (Random Forest, SVM, KNN) akan disimpan ke `results/metrics.csv` dan script `src/generate_metrics_table.py` akan otomatis dijalankan untuk memperbarui tabel hasil di bawah ini.

### 4. Memperbarui Tabel Metrik Secara Manual
Jika ingin memperbarui tabel metrik dan grafik secara manual dari berkas `results/metrics.csv`, jalankan perintah berikut:
```cmd
python src/generate_metrics_table.py
```

## Hasil Eksperimen (Results)

Berikut adalah ringkasan hasil evaluasi dari setiap eksperimen yang telah dilakukan:

<!-- BEGIN METRICS -->
| Experiment | Classifier | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| Baseline | Random Forest | 0,5139 | 0,5144 | 0,5139 | 0,5133 |
| Baseline | SVM | 0,4985 | 0,5287 | 0,4985 | 0,5049 |
| Baseline | KNN | 0,4796 | 0,4799 | 0,4796 | 0,4771 |
| Experiment 1 | Random Forest | 0,7812 | 0,7807 | 0,7812 | 0,7805 |
| Experiment 1 | SVM | 0,7552 | 0,7601 | 0,7552 | 0,7559 |
| Experiment 1 | KNN | 0,7238 | 0,7228 | 0,7238 | 0,7202 |
| Experiment 2 | Random Forest | 0,7812 | 0,7808 | 0,7812 | 0,7806 |
| Experiment 2 | SVM | 0,7433 | 0,7443 | 0,7433 | 0,7430 |
| Experiment 2 | KNN | 0,6895 | 0,6867 | 0,6895 | 0,6855 |
| Experiment 3 | Random Forest | 0,6558 | 0,6547 | 0,6558 | 0,6531 |
| Experiment 3 | SVM | 0,6092 | 0,6212 | 0,6092 | 0,6035 |
| Experiment 3 | KNN | 0,6211 | 0,6154 | 0,6211 | 0,6132 |
| Experiment 4 | Random Forest | 0,8321 | 0,8312 | 0,8321 | 0,8314 |
| Experiment 4 | SVM | 0,7918 | 0,8008 | 0,7918 | 0,7947 |
| Experiment 4 | KNN | 0,7766 | 0,7696 | 0,7766 | 0,7703 |
| Experiment 5 | Random Forest | 0,8190 | 0,8190 | 0,8190 | 0,8187 |
| Experiment 5 | SVM | 0,8255 | 0,8260 | 0,8255 | 0,8254 |
| Experiment 5 | KNN | 0,7776 | 0,7791 | 0,7776 | 0,7768 |
| Experiment 6 | Random Forest | 0,7818 | 0,7802 | 0,7818 | 0,7795 |
| Experiment 6 | SVM | 0,7877 | 0,7869 | 0,7877 | 0,7868 |
| Experiment 6 | KNN | 0,7345 | 0,7318 | 0,7345 | 0,7315 |
| Experiment 7 | Random Forest | 0,7788 | 0,7756 | 0,7788 | 0,7764 |
| Experiment 7 | SVM | 0,8025 | 0,8017 | 0,8025 | 0,8011 |
| Experiment 7 | KNN | 0,7487 | 0,7465 | 0,7487 | 0,7464 |
| Experiment 8 | Random Forest | 0,7002 | 0,7011 | 0,7002 | 0,6994 |
| Experiment 8 | SVM | 0,6558 | 0,6622 | 0,6558 | 0,6515 |
| Experiment 8 | KNN | 0,6434 | 0,6470 | 0,6434 | 0,6425 |

### Grafik Perbandingan Akurasi

![Comparison Chart](results/figures/metrics_comparison.png)
<!-- END METRICS -->

## Hasil Analisis Eksperimen

- **[Analisis Baseline (Tanpa Preprocessing)](notebooks/00_baseline.ipynb#Analisis)**: Analisis performa awal menggunakan tekstur GLCM dasar sebagai baseline pembanding.
- **[Analisis Experiment 1 (GLCM + HSV Stats)](notebooks/01_experiment1.ipynb#Analisis)**: Analisis pengaruh statistik warna HSV terhadap model klasifikasi.
- **[Analisis Experiment 2 (NRBR + HSV + GLCM)](notebooks/02_experiment2.ipynb#Analisis)**: Analisis performa rasio warna Normalized Red-Blue Ratio (NRBR) dan visualisasinya.
- **[Analisis Experiment 3 (Grayscale Histogram + Stats)](notebooks/03_experiment3.ipynb#Analisis)**: Analisis efektivitas histogram keabuan dan statistik orde pertama tanpa fitur warna.
- **[Analisis Experiment 4 (HSV Histograms + Stats + GLCM)](notebooks/04_experiment4.ipynb#Analisis)**: Analisis performa histogram warna HSV 32 bin yang digabungkan dengan statistik warna.
- **[Analisis Experiment 5 (CLAHE HSV + Gaussian)](notebooks/05_experiment5.ipynb#Analisis)**: Analisis teknik peningkatan kontras lokal (CLAHE) pada channel saturasi/value ruang warna HSV beserta peredaman noise Gaussian.
- **[Analisis Experiment 6 (Min-Max Normalization + CLAHE + Gamma)](notebooks/06_experiment6.ipynb#Analisis)**: Analisis efek normalisasi kontras global dan koreksi gamma (kecerahan) pada performa akurasi.
- **[Analisis Experiment 7 (Min-Max Normalization + CLAHE + Gaussian + Gamma)](notebooks/07_experiment7.ipynb#Analisis)**: Analisis performa dengan penambahan Gaussian Blur setelah CLAHE untuk meredam noise sebelum koreksi gamma.
- **[Analisis Experiment 8 (White Balance + Bilateral + CLAHE + RGB Stats + Feature Selection)](notebooks/08_experiment8.ipynb#Analisis)**: Analisis efek restorasi iluminasi (Gray World), bilateral filtering, statistik warna RGB, dan reduksi dimensi (korelasi >= 0.95).




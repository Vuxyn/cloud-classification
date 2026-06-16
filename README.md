# Cloud Image Classification System

Sistem klasifikasi citra awan berbasis pemrosesan citra digital tradisional (preprocessing manual, ekstraksi fitur warna dan tekstur, serta klasifikasi menggunakan Random Forest, SVM, dan KNN).

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

- **Baseline (`00_baseline.ipynb`)**: Tanpa preprocessing tambahan.
- **Experiment 1 (`01_experiment1.ipynb`)**: Histogram Equalization + Gaussian Filter.
- **Experiment 2 (`02_experiment2.ipynb`)**: CLAHE + Morphological Opening.
- **Experiment 3 (`03_experiment3.ipynb`)**: Haar Wavelet Denoise + Unsharp Mask.
- **Experiment 4 (`04_experiment4.ipynb`)**: Histogram Equalization + Edge + Morfologi.
- **Experiment 5 (`05_experiment5.ipynb`)**: Eksperimen LBP (Local Binary Patterns) pada subset kecil.
- **Experiment 6 (`06_experiment6.ipynb`)**: NRBR + HSV + GLCM (Tanpa LBP).
- **Experiment 7 (`07_experiment7.ipynb`)**: Grayscale GLCM + Grayscale Histogram + Stats (Tanpa LBP).
- **Experiment 8 (`08_experiment8.ipynb`)**: Detailed HSV Histograms + HSV Stats + GLCM (Tanpa LBP).
- **Experiment 9 (`09_experiment9.ipynb`)**: CLAHE pada channel S dan V ruang warna HSV + Gaussian Filter → Grayscale → GLCM.

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
| Baseline | Random Forest | 0,5579 | 0,5503 | 0,5579 | 0,5493 |
| Baseline | SVM | 0,5242 | 0,5194 | 0,5242 | 0,5006 |
| Baseline | KNN | 0,5129 | 0,5037 | 0,5129 | 0,5025 |
| Experiment 1 | Random Forest | 0,5455 | 0,5295 | 0,5455 | 0,5241 |
| Experiment 1 | SVM | 0,5305 | 0,5068 | 0,5305 | 0,4698 |
| Experiment 1 | KNN | 0,4924 | 0,4698 | 0,4924 | 0,4721 |
| Experiment 2 | Random Forest | 0,5924 | 0,5789 | 0,5924 | 0,5807 |
| Experiment 2 | SVM | 0,5845 | 0,5660 | 0,5845 | 0,5553 |
| Experiment 2 | KNN | 0,5511 | 0,5362 | 0,5511 | 0,5389 |
| Experiment 3 | Random Forest | 0,5789 | 0,5656 | 0,5789 | 0,5658 |
| Experiment 3 | SVM | 0,5539 | 0,5388 | 0,5539 | 0,5181 |
| Experiment 3 | KNN | 0,5247 | 0,5106 | 0,5247 | 0,5119 |
| Experiment 4 | Random Forest | 0,5732 | 0,5592 | 0,5732 | 0,5543 |
| Experiment 4 | SVM | 0,5547 | 0,5382 | 0,5547 | 0,5140 |
| Experiment 4 | KNN | 0,5229 | 0,5085 | 0,5229 | 0,5083 |
| Experiment 5 | Random Forest | 0,8242 | 0,8238 | 0,8242 | 0,8237 |
| Experiment 5 | SVM | 0,7882 | 0,8019 | 0,7882 | 0,7923 |
| Experiment 5 | KNN | 0,7945 | 0,7894 | 0,7945 | 0,7889 |
| Experiment 6 | Random Forest | 0,8413 | 0,8405 | 0,8413 | 0,8406 |
| Experiment 6 | SVM | 0,7963 | 0,8066 | 0,7963 | 0,7997 |
| Experiment 6 | KNN | 0,7861 | 0,7803 | 0,7861 | 0,7800 |
| Experiment 7 | Random Forest | 0,6589 | 0,6599 | 0,6589 | 0,6575 |
| Experiment 7 | SVM | 0,6200 | 0,6332 | 0,6200 | 0,6140 |
| Experiment 7 | KNN | 0,6176 | 0,6125 | 0,6176 | 0,6106 |
| Experiment 8 | Random Forest | 0,8413 | 0,8397 | 0,8413 | 0,8400 |
| Experiment 8 | SVM | 0,7955 | 0,8058 | 0,7955 | 0,7989 |
| Experiment 8 | KNN | 0,7839 | 0,7778 | 0,7839 | 0,7782 |
| Experiment9 | Random Forest | 0,5758 | 0,5644 | 0,5758 | 0,5629 |
| Experiment9 | SVM | 0,5632 | 0,5482 | 0,5632 | 0,5319 |
| Experiment9 | KNN | 0,5226 | 0,5062 | 0,5226 | 0,5061 |

### Grafik Perbandingan Akurasi

![Comparison Chart](results/figures/metrics_comparison.png)
<!-- END METRICS -->

## Hasil Analisis Eksperimen

- **[Analisis Baseline (Tanpa Preprocessing)](notebooks/00_baseline.ipynb#Analisis)**: Analisis performa tanpa perbaikan citra, sebagai pembanding dasar.
- **[Analisis Experiment 1 (HE + Gaussian)](notebooks/01_experiment1.ipynb#Analisis)**: Analisis efek Global Histogram Equalization dan Gaussian blur.
- **[Analisis Experiment 2 (CLAHE + Morph Opening)](notebooks/02_experiment2.ipynb#Analisis)**: Analisis peningkatan kontras lokal dan pembersihan noise.
- **[Analisis Experiment 3 (Wavelet Denoise + Unsharp)](notebooks/03_experiment3.ipynb#Analisis)**: Analisis efek restorasi wavelet dan penajaman tepi.
- **[Analisis Experiment 4 (HE + Laplacian + Closing)](notebooks/04_experiment4.ipynb#Analisis)**: Analisis deteksi tepi berbasis Laplacian.
- **[Analisis Experiment 5 (LBP + GLCM + HSV)](notebooks/05_experiment5.ipynb#Analisis)**: Analisis kontribusi Local Binary Patterns (LBP) dan fitur warna HSV.
- **[Analisis Experiment 6 (NRBR + HSV + GLCM)](notebooks/06_experiment6.ipynb#Analisis)**: Analisis performa Normalized Red-Blue Ratio (NRBR).
- **[Analisis Experiment 7 (Grayscale Stats & GLCM)](notebooks/07_experiment7.ipynb#Analisis)**: Analisis performa fitur grayscale komprehensif tanpa fitur warna.
- **[Analisis Experiment 8 (HSV Histograms + Stats + GLCM)](notebooks/08_experiment8.ipynb#Analisis)**: Analisis performa histogram HSV detail dan statistik warna.
- **[Analisis Experiment 9 (CLAHE HSV + Gaussian)](notebooks/09_experiment9.ipynb#Analisis)**: Analisis efek CLAHE pada ruang warna HSV terhadap GLCM.


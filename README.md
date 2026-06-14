# Cloud Image Classification (backup)

Sistem klasifikasi citra awan berbasis preprocessing manual, ekstraksi fitur GLCM, dan klasifikasi dengan Random Forest, SVM, dan KNN.

Project ini self-contained di folder `backup/`. Dataset utama dibaca dari path eksternal (`.env` atau Google Drive), bukan dari folder `dataset/` lokal.

## Struktur Folder

```
backup/
├── dataset/                 # Fallback lokal (opsional, boleh kosong)
│   ├── train/
│   └── test/
├── src/
│   ├── image_processing.py  # Library preprocessing manual
│   └── loader.py            # Loader dataset
├── notebooks/
│   ├── 00_baseline.ipynb
│   ├── 01_experiment1.ipynb
│   ├── 02_experiment2.ipynb
│   ├── 03_experiment3.ipynb
│   └── 04_experiment4.ipynb
├── results/
│   ├── figures/             # Confusion matrix per experiment
│   └── metrics.csv          # Tabel perbandingan experiment (di git)
├── .env.example
└── README.md
```

## Persyaratan

- Python 3.10+
- Conda environment (disarankan)
- Dataset GCD dengan struktur:

```
GCD/
├── train/
│   ├── 1_cumulus/
│   ├── 2_stratus/
│   └── ...
└── test/
    ├── 1_cumulus/
    └── ...
```

Loader akan menggabungkan `train` dan `test`. Split ulang 80/20 dilakukan di notebook.

## Setup
### 1. Install dependensi

```cmd
python -m pip install numpy pandas opencv-python scikit-image scikit-learn scipy matplotlib seaborn tqdm jupyter python-dotenv
```

CuPy (opsional, hanya jika ada NVIDIA GPU):

```cmd
python -m pip install cupy-cuda12x
```
tergantung versi CUDA

Tanpa CuPy, project otomatis fallback ke NumPy (CPU).

### 3. Konfigurasi dataset lokal

Salin `.env.example` menjadi `.env`:

```cmd
copy .env.example .env
```

Isi path dataset:

```env
DATASET_ROOT=D:/path/ke/GCD
```

Gunakan forward slash (`/`) agar aman di Windows.

### 4. Verifikasi modul

Dari root workspace (`Project/`):

```cmd
python backup\src\image_processing.py
```

Import test:

```cmd
python -c "from backup.src import image_processing, loader"
```

## Menjalankan Notebook

1. Buka folder `backup/notebooks/`
2. Pilih kernel Python environment `improc`
3. Jalankan notebook berurutan:

| Notebook | Pipeline |
|---|---|
| `00_baseline` | Tanpa preprocessing tambahan |
| `01_experiment1` | Histogram Equalization + Gaussian Filter |
| `02_experiment2` | CLAHE + Morphological Opening |
| `03_experiment3` | Haar Wavelet Denoise + Unsharp Mask |
| `04_experiment4` | HE + Edge + Morphology |

Jalankan semua cell dari atas ke bawah (Run All).

### Via Jupyter

```cmd
conda activate improc
cd /d "path\ke\backup\notebooks"
jupyter notebook
```

## Prioritas Lokasi Dataset

`get_dataset_root()` memilih path dengan urutan:

1. Google Colab: `/content/drive/MyDrive/GCD`
2. Lokal `.env`: `DATASET_ROOT=...`
3. Fallback: `backup/dataset/`
4. Error jika tidak ditemukan

Jika `.env` sudah diisi, folder `backup/dataset/` tidak dipakai.

## Output

Setelah notebook selesai:

- `notebooks/hasil_ekstraksi_{experiment_name}.csv` - fitur GLCM (lokal, tidak di git)
- `results/figures/{experiment_name}_{classifier}.png` - confusion matrix (lokal)
- `results/metrics.csv` - **tabel perbandingan experiment tim (di git, di-push)**

### Alur kerja tim (tabel experiment)

```text
git pull                          # ambil metrics.csv terbaru
Run All notebook (1 experiment)   # update 3 baris (rf, svm, knn)
git add results/metrics.csv notebooks/XX.ipynb
git commit -m "Experiment 2 selesai"
git push
```

Tiap notebook hanya mengupdate baris experiment-nya sendiri di `metrics.csv`. Setelah 5 experiment selesai = 15 baris (5 x 3 classifier).

### Tabel Perbandingan Performa

Berikut adalah tabel performa model training dari tiap eksperimen yang di-generate secara otomatis:

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
| Experiment 5 | Random Forest | 0,8250 | 0,8238 | 0,8250 | 0,8242 |
| Experiment 5 | SVM | 0,7771 | 0,7886 | 0,7771 | 0,7802 |
| Experiment 5 | KNN | 0,7887 | 0,7834 | 0,7887 | 0,7830 |
<!-- END METRICS -->

## Catatan Penting

- Tidak ada cache `.npz`: data diproses langsung setiap run.
- Preprocessing memakai `ProcessPoolExecutor`. Di Windows, jika error, restart kernel lalu jalankan ulang.
- Dataset besar (~19K gambar) bisa memakan waktu lama, terutama di CPU.

## Troubleshooting

| Masalah | Solusi |
|---|---|
| `pip` tidak dikenali | Pakai `python -m pip install ...` |
| `ModuleNotFoundError` | Pastikan install dan kernel notebook sama (`improc`) |
| Dataset tidak ditemukan | Cek isi `.env` dan struktur folder `GCD/train`, `GCD/test` |
| CuPy gagal install | Skip CuPy, tetap bisa jalan di CPU |
| Multiprocessing error | Restart kernel, Run All ulang |

## CuPy Setup (Opsional)

Syarat:
- NVIDIA GPU
- Driver CUDA terbaru

Cek GPU:

```cmd
nvidia-smi
```

Install (CUDA 12.x):

```cmd
conda activate (envs)
python -m pip install cupy-cuda12x
```

Verifikasi:

```cmd
python -c "import cupy as cp; cp.cuda.Device(0).use(); print('CuPy OK')"
```

Jika berhasil, saat menjalankan `image_processing.py` akan muncul pesan bahwa CuPy aktif.

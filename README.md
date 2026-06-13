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
│   └── metrics.csv          # Akumulasi metrik (append)
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

- `notebooks/hasil_ekstraksi_{experiment_name}.csv` - fitur GLCM
- `results/figures/{experiment_name}_{classifier}.png` - confusion matrix
- `results/metrics.csv` - accuracy, precision, recall, f1 (mode append)

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

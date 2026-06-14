"""
loader.py - Modul untuk memuat dataset citra awan.

Mendukung Google Colab (Google Drive) dan environment lokal (.env).
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .image_processing import _ensure_grayscale

BACKUP_ROOT = Path(__file__).resolve().parent.parent


def _is_colab() -> bool:
    """Deteksi apakah sedang berjalan di Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def get_drive_root() -> str:
    """
    Deteksi environment:
    - Google Colab -> mount Drive ke /content/drive, return "/content/drive/MyDrive"
    - Lokal -> return "" dan print peringatan
    """
    if _is_colab():
        mount_point = "/content/drive"
        if not os.path.ismount(mount_point):
            from google.colab import drive
            drive.mount(mount_point)
            print("Google Drive berhasil di-mount")
        return "/content/drive/MyDrive"

    print("PERINGATAN: Bukan Google Colab. Set DATASET_ROOT di file .env untuk environment lokal.")
    return ""


def download_dataset_from_drive(folder_url: str, output_dir: str = "dataset") -> None:
    """
    Download dataset dari Google Drive menggunakan gdown.
    Skip jika output_dir sudah ada dan tidak kosong.
    folder_url harus berupa shared folder link (Anyone with link).
    """
    import gdown

    out_path = Path(output_dir)
    if out_path.exists() and any(out_path.iterdir()):
        print(f"Skip download: {out_path} sudah ada dan tidak kosong")
        return

    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Mengunduh dataset dari Google Drive ke {out_path} ...")
    gdown.download_folder(url=folder_url, output=str(out_path), quiet=False, use_cookies=False)
    print("Download selesai")


def get_dataset_root() -> Path:
    """
    Tentukan DATASET_ROOT dengan urutan prioritas:
    1. Google Colab  -> Path("/content/drive/MyDrive/GCD")
    2. Lokal .env     -> baca DATASET_ROOT dari .env menggunakan python-dotenv
    3. Lokal default  -> Path("dataset") relatif dari backup/ root
    4. Fallback       -> print instruksi jelas dan raise FileNotFoundError
    """
    if _is_colab():
        colab_options = [
            Path("/content/GCD-zip"),
            Path("/content/GCD-zip/GCD"),
            Path("/content/GCD"),
            Path("/content/drive/MyDrive/GCD"),
            Path("/content/drive/MyDrive/GCD-zip")
        ]
        for opt in colab_options:
            if opt.is_dir():
                if (opt / "train").is_dir() or (opt / "test").is_dir():
                    print(f"Environment: Google Colab | DATASET_ROOT = {opt}")
                    return opt
        raise FileNotFoundError(
            "Dataset tidak ditemukan di Google Colab.\n"
            "Pastikan Anda sudah mengekstrak zip dataset ke /content/GCD-zip atau mengunggah folder GCD ke Google Drive."
        )

    print("Environment: Lokal")

    env_path = BACKUP_ROOT / ".env"
    if env_path.is_file():
        from dotenv import load_dotenv
        load_dotenv(env_path)
        env_root = os.getenv("DATASET_ROOT", "").strip()
        if env_root:
            root = Path(env_root)
            print(f"DATASET_ROOT dari .env: {root}")
            if root.is_dir():
                return root
            raise FileNotFoundError(
                f"DATASET_ROOT di .env tidak ditemukan: {root}\n"
                "Periksa path dataset lokal Anda."
            )

    default_root = BACKUP_ROOT / "dataset"
    print(f"DATASET_ROOT default: {default_root}")
    if default_root.is_dir():
        has_data = any(
            (default_root / split).is_dir() and any((default_root / split).iterdir())
            for split in ("train", "test")
        ) or any(default_root.iterdir())
        if has_data:
            return default_root

    print(
        "Dataset tidak ditemukan.\n"
        "Langkah yang bisa dilakukan:\n"
        "  1. Salin .env.example menjadi .env dan isi DATASET_ROOT\n"
        "  2. Salin dataset ke backup/dataset/{train,test}/label/\n"
        "  3. Gunakan download_dataset_from_drive() di Colab"
    )
    raise FileNotFoundError("Dataset root tidak ditemukan. Lihat instruksi di atas.")


def load_dataset(root_dir: Path, target_size: tuple = (128, 128), color: bool = False) -> tuple:
    """
    Load semua gambar dari struktur: root_dir/{train,test}/label_name/image.jpg

    - Gabungkan train dan test (klasifikasi ulang dengan split manual di notebook)
    - Konversi ke grayscale jika color=False menggunakan _ensure_grayscale dari image_processing.py
    - Resize dengan cv2.resize ke target_size
    - Tampilkan tqdm progress bar per label
    - Skip gambar yang gagal dibaca (cv2.imread return None) + print warning
    - Return: (images: np.ndarray shape NxHxW[xC] uint8, labels: np.ndarray str, filenames: list[str])
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Direktori dataset tidak ditemukan: {root}")

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images_list: list[np.ndarray] = []
    labels_list: list[str] = []
    filenames_list: list[str] = []

    splits = []
    for split_name in ("train", "test"):
        split_dir = root / split_name
        if split_dir.is_dir():
            splits.append(split_dir)

    if not splits:
        label_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
        if not label_dirs:
            raise ValueError(f"Tidak ada subfolder train/test atau label di: {root}")
        splits = [root]

    for split_dir in splits:
        label_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        for label_dir in label_dirs:
            label_name = label_dir.name
            parts = label_name.split("_", 1)
            class_name = parts[1] if len(parts) > 1 else label_name

            img_files = sorted([
                f for f in label_dir.iterdir()
                if f.is_file() and f.suffix.lower() in image_exts
            ])

            for img_path in tqdm(img_files, desc=f"Loading {class_name}", leave=False):
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"PERINGATAN: Gagal membaca gambar, dilewati: {img_path}")
                    continue

                if not color:
                    img = _ensure_grayscale(img)
                img = cv2.resize(
                    img,
                    (target_size[1], target_size[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

                images_list.append(img)
                labels_list.append(class_name)
                filenames_list.append(img_path.name)

    if not images_list:
        raise ValueError(f"Tidak ada gambar yang berhasil dimuat dari: {root}")

    images = np.array(images_list, dtype=np.uint8)
    labels = np.array(labels_list, dtype=object)

    print(
        f"Dataset loaded: {len(images)} gambar, "
        f"{len(set(labels_list))} kelas, ukuran={target_size}"
    )

    return images, labels, filenames_list

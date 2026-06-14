from __future__ import annotations

import numpy as np
import cv2

import sys
import os

if 'google.colab' in sys.modules or os.path.exists('/usr/local/cuda'):
    os.environ['CUDA_PATH'] = '/usr/local/cuda'

try:
    import cupy as cp
    cp.cuda.Device(0).use()
    xp = cp
    USING_GPU = True
    print("CuPy aktif - komputasi berjalan di GPU")
except Exception:
    xp = np
    USING_GPU = False
    print("CuPy tidak tersedia - fallback ke NumPy (CPU)")


def _to_xp(image: np.ndarray):
    """Pindahkan array ke device yang aktif (GPU atau CPU)."""
    return xp.asarray(image)


def _to_np(image) -> np.ndarray:
    """Kembalikan array ke CPU sebagai np.ndarray."""
    return xp.asnumpy(image) if USING_GPU else np.asarray(image)


def _pad_image(image, pad: int, mode: str = 'reflect') -> np.ndarray:
    """Zero-pad atau reflect-pad image sebesar `pad` piksel di semua sisi."""
    if pad <= 0:
        return _to_np(image)

    pad_mode = 'reflect' if mode == 'reflect' else 'constant'
    constant_values = 0 if mode != 'reflect' else None

    if pad_mode == 'constant':
        padded = xp.pad(
            image,
            ((pad, pad), (pad, pad)),
            mode=pad_mode,
            constant_values=constant_values,
        )
    else:
        padded = xp.pad(image, ((pad, pad), (pad, pad)), mode=pad_mode)

    return _to_np(padded)


def _convolve2d_xp(image, kernel):
    """
    Konvolusi 2D internal — input/output tetap di xp (GPU/CPU).
    Reflect-padding sebesar kernel_size//2.
    """
    img = image.astype(xp.float64)
    kernel = kernel.astype(xp.float64)

    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = xp.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

    h, w = img.shape
    output = xp.zeros((h, w), dtype=xp.float64)

    for i in range(kh):
        for j in range(kw):
            output += padded[i:i + h, j:j + w] * kernel[i, j]

    return output


def _convolve2d(image, kernel) -> np.ndarray:
    """Wrapper konvolusi 2D — kembalikan np.ndarray untuk API publik."""
    img = _to_xp(image)
    k = _to_xp(kernel)
    return _to_np(_convolve2d_xp(img, k))


def _histogram_256(image) -> np.ndarray:
    """Hitung histogram 256-bin secara vectorized."""
    flat = image.astype(xp.int32).ravel()
    if USING_GPU:
        hist = xp.bincount(flat, minlength=256).astype(xp.float64)
    else:
        hist, _ = xp.histogram(flat, bins=256, range=(0, 256))
        hist = hist.astype(xp.float64)
    return hist


def _ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Pastikan image adalah grayscale uint8.
    Jika RGB/BGR (3 channel) -> konversi dengan cv2.cvtColor ke grayscale.
    Jika sudah grayscale -> return as-is.
    """
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


# -- NORMALISASI ----------------------------------------------------------------

def normalize_minmax(image: np.ndarray) -> np.ndarray:
    """Rescale intensitas ke [0, 255]. Handle edge case semua pixel sama."""
    img = _to_xp(image).astype(xp.float64)
    mi, ma = img.min(), img.max()

    if mi == ma:
        return _to_np(xp.zeros_like(img, dtype=xp.uint8))

    norm = (img - mi) / (ma - mi) * 255.0
    result = xp.clip(norm, 0, 255).astype(xp.uint8)
    return _to_np(result)


def normalize_zscore(image: np.ndarray, clip: bool = True) -> np.ndarray:
    """Z-score normalization, clip hasil ke [0, 255] jika clip=True."""
    img = _to_xp(image).astype(xp.float64)
    mean = img.mean()
    std = img.std()
    z = (img - mean) / (std + 1e-8)

    if clip:
        z_min, z_max = z.min(), z.max()
        if z_max - z_min == 0:
            result = xp.zeros_like(img, dtype=xp.uint8)
        else:
            norm = (z - z_min) / (z_max - z_min) * 255.0
            result = xp.clip(norm, 0, 255).astype(xp.uint8)
    else:
        result = xp.clip(z * 127.5 + 127.5, 0, 255).astype(xp.uint8)

    return _to_np(result)


# -- HISTOGRAM EQUALIZATION -----------------------------------------------------

def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Global HE menggunakan CDF. Hitung histogram -> CDF -> LUT -> apply."""
    img = _to_xp(image).astype(xp.uint8)
    h, w = img.shape

    hist = _histogram_256(img)
    cdf = xp.cumsum(hist)
    total_pixels = h * w
    lut = xp.round(cdf * 255.0 / total_pixels).astype(xp.uint8)

    result = lut[img.ravel()].reshape(h, w)
    return _to_np(result)


def clahe(image: np.ndarray, tile_size: int = 8, clip_limit: float = 2.0) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization.
    Bagi gambar jadi grid tile_size x tile_size.
    Clip histogram tiap tile di clip_limit x (tile_area / 256).
    HE per tile, interpolasi bilinear antar tile.
    """
    img = _to_xp(image).astype(xp.float64)
    h, w = img.shape

    tile_h = max(1, h // tile_size)
    tile_w = max(1, w // tile_size)
    padded_h = tile_h * tile_size
    padded_w = tile_w * tile_size
    img_crop = img[:padded_h, :padded_w]

    luts = xp.zeros((tile_size, tile_size, 256), dtype=xp.float64)

    for ti in range(tile_size):
        for tj in range(tile_size):
            y0 = ti * tile_h
            x0 = tj * tile_w
            tile = img_crop[y0:y0 + tile_h, x0:x0 + tile_w]

            tile_uint8 = xp.clip(tile, 0, 255).astype(xp.int32)
            hist = _histogram_256(tile_uint8)

            tile_area = tile_h * tile_w
            clip_val = clip_limit * (tile_area / 256.0)
            excess = xp.sum(xp.maximum(hist - clip_val, 0))
            hist = xp.minimum(hist, clip_val)
            hist += excess / 256.0

            cdf = xp.cumsum(hist)
            cdf_min = cdf[cdf > 0].min() if xp.any(cdf > 0) else 0
            denom = float(tile_area) - float(cdf_min)
            if denom > 0:
                luts[ti, tj] = xp.clip(
                    (cdf - cdf_min) / denom * 255.0, 0, 255
                )
            else:
                luts[ti, tj] = xp.arange(256, dtype=xp.float64)

    ys = xp.arange(int(padded_h), dtype=xp.float64)
    xs = xp.arange(int(padded_w), dtype=xp.float64)
    yy, xx = xp.meshgrid(ys, xs, indexing='ij')

    fy = (yy - tile_h / 2.0) / tile_h
    fx = (xx - tile_w / 2.0) / tile_w

    ti0 = xp.floor(fy).astype(xp.int32).clip(0, tile_size - 1)
    tj0 = xp.floor(fx).astype(xp.int32).clip(0, tile_size - 1)
    ti1 = xp.minimum(ti0 + 1, tile_size - 1)
    tj1 = xp.minimum(tj0 + 1, tile_size - 1)

    dy = (fy - ti0).clip(0.0, 1.0)
    dx = (fx - tj0).clip(0.0, 1.0)

    pixels = img_crop.astype(xp.int32).clip(0, 255)
    lut00 = luts[ti0, tj0, pixels]
    lut01 = luts[ti0, tj1, pixels]
    lut10 = luts[ti1, tj0, pixels]
    lut11 = luts[ti1, tj1, pixels]

    result = (
        (1 - dy) * (1 - dx) * lut00
        + (1 - dy) * dx * lut01
        + dy * (1 - dx) * lut10
        + dy * dx * lut11
    )

    final = xp.zeros_like(_to_xp(image), dtype=xp.uint8)
    final[:padded_h, :padded_w] = xp.clip(result, 0, 255).astype(xp.uint8)

    if padded_h < h:
        final[padded_h:, :padded_w] = xp.clip(
            img[padded_h:, :padded_w], 0, 255
        ).astype(xp.uint8)
    if padded_w < w:
        final[:padded_h, padded_w:] = xp.clip(
            img[:padded_h, padded_w:], 0, 255
        ).astype(xp.uint8)
    if padded_h < h and padded_w < w:
        final[padded_h:, padded_w:] = xp.clip(
            img[padded_h:, padded_w:], 0, 255
        ).astype(xp.uint8)

    return _to_np(final)


# -- FILTERING ------------------------------------------------------------------

def gaussian_filter(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Gaussian blur manual. Buat kernel 2D: G(x,y)=exp(-(x^2+y^2)/(2*sigma^2)), konvolusi manual."""
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size harus ganjil")

    img = _to_xp(image).astype(xp.float64)
    half = kernel_size // 2
    ax = xp.arange(-half, half + 1, dtype=xp.float64)
    xx, yy = xp.meshgrid(ax, ax)
    kernel = xp.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / kernel.sum()

    result = _convolve2d_xp(img, kernel)
    result = xp.clip(result, 0, 255).astype(xp.uint8)
    return _to_np(result)


def mean_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Box blur manual dengan sliding window."""
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size harus ganjil")

    img = _to_xp(image).astype(xp.float64)
    kernel = xp.ones((kernel_size, kernel_size), dtype=xp.float64)
    kernel /= (kernel_size * kernel_size)

    result = _convolve2d_xp(img, kernel)
    result = xp.clip(result, 0, 255).astype(xp.uint8)
    return _to_np(result)


def median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Median filter manual menggunakan np.median per patch."""
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size harus ganjil")

    img = _to_xp(image).astype(xp.float64)
    pad = kernel_size // 2
    padded = xp.pad(img, ((pad, pad), (pad, pad)), mode='reflect')

    h, w = img.shape
    result = xp.zeros((h, w), dtype=xp.float64)

    for i in range(h):
        for j in range(w):
            patch = padded[i:i + kernel_size, j:j + kernel_size]
            result[i, j] = xp.median(patch)

    result = xp.clip(result, 0, 255).astype(xp.uint8)
    return _to_np(result)


# -- SHARPENING -----------------------------------------------------------------

def laplacian_sharpen(image: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Laplacian kernel [[0,-1,0],[-1,4,-1],[0,-1,0]], sharpened = image + alpha * laplacian."""
    img = _to_xp(image).astype(xp.float64)
    laplacian_kernel = xp.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0],
    ], dtype=xp.float64)

    laplacian = _convolve2d_xp(img, laplacian_kernel)
    sharpened = img + alpha * laplacian
    result = xp.clip(sharpened, 0, 255).astype(xp.uint8)
    return _to_np(result)


def unsharp_mask(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0,
                 amount: float = 1.5) -> np.ndarray:
    """mask = image - gaussian_filter(image), output = image + amount * mask."""
    img = _to_xp(image).astype(xp.float64)
    blurred = _to_xp(gaussian_filter(image, kernel_size, sigma)).astype(xp.float64)

    mask = img - blurred
    output = img + amount * mask
    result = xp.clip(output, 0, 255).astype(xp.uint8)
    return _to_np(result)


# -- THRESHOLDING ---------------------------------------------------------------

def binary_threshold(image: np.ndarray, threshold: int = 127) -> np.ndarray:
    """Pixel >= threshold -> 255, else -> 0."""
    img = _to_xp(image)
    result = xp.where(img >= threshold, xp.uint8(255), xp.uint8(0)).astype(xp.uint8)
    return _to_np(result)


def otsu_threshold(image: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Otsu thresholding menggunakan within-class variance.
    Iterasi semua t (1-254), pilih t dengan variance terkecil.
    Return: (thresholded_image, optimal_t)
    """
    img = _to_xp(image).astype(xp.float64)
    h, w = img.shape
    total_pixels = h * w

    img_int = xp.clip(img, 0, 255).astype(xp.int32)
    hist = _histogram_256(img_int)

    sum_total = 0.0
    for i in range(256):
        sum_total += i * float(hist[i])

    best_var = -1.0
    best_t = 0
    weight_bg = 0.0
    sum_bg = 0.0

    for t in range(256):
        weight_bg += float(hist[t])
        if weight_bg == 0:
            continue

        weight_fg = total_pixels - weight_bg
        if weight_fg == 0:
            break

        sum_bg += t * float(hist[t])
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg

        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

        if var_between > best_var:
            best_var = var_between
            best_t = t

    result = xp.where(img_int >= best_t, xp.uint8(255), xp.uint8(0)).astype(xp.uint8)
    return _to_np(result), best_t


def adaptive_threshold(image: np.ndarray, block_size: int = 11, C: int = 2) -> np.ndarray:
    """Threshold per region: t = mean(region) - C. block_size harus ganjil."""
    if block_size % 2 == 0:
        raise ValueError("block_size harus ganjil")

    img = _to_xp(image).astype(xp.float64)
    local_mean = _to_xp(mean_filter(image, block_size)).astype(xp.float64)
    threshold_map = local_mean - C

    result = xp.where(img >= threshold_map, xp.uint8(255), xp.uint8(0)).astype(xp.uint8)
    return _to_np(result)


# -- MORFOLOGI ------------------------------------------------------------------

def dilate(image: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """Dilation: output pixel = max(patch). Structuring element kotak."""
    img = _to_xp(image).astype(xp.float64)
    pad = kernel_size // 2

    for _ in range(iterations):
        padded = xp.pad(img, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)
        h, w = img.shape
        result = xp.zeros((h, w), dtype=xp.float64)

        for i in range(kernel_size):
            for j in range(kernel_size):
                result = xp.maximum(result, padded[i:i + h, j:j + w])

        img = result

    result = xp.clip(img, 0, 255).astype(xp.uint8)
    return _to_np(result)


def erode(image: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """Erosion: output pixel = min(patch)."""
    img = _to_xp(image).astype(xp.float64)
    pad = kernel_size // 2

    for _ in range(iterations):
        padded = xp.pad(img, ((pad, pad), (pad, pad)), mode='constant', constant_values=255)
        h, w = img.shape
        result = xp.full((h, w), 255.0, dtype=xp.float64)

        for i in range(kernel_size):
            for j in range(kernel_size):
                result = xp.minimum(result, padded[i:i + h, j:j + w])

        img = result

    result = xp.clip(img, 0, 255).astype(xp.uint8)
    return _to_np(result)


def morphological_opening(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """opening = dilate(erode(image)). Hilangkan noise kecil."""
    eroded = erode(image, kernel_size)
    return dilate(eroded, kernel_size)


def morphological_closing(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """closing = erode(dilate(image)). Tutup lubang kecil."""
    dilated = dilate(image, kernel_size)
    return erode(dilated, kernel_size)


def top_hat(image: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """White top-hat: image - opening(image)."""
    img = _to_xp(image).astype(xp.float64)
    opened = _to_xp(morphological_opening(image, kernel_size)).astype(xp.float64)

    result = img - opened
    result = xp.clip(result, 0, 255).astype(xp.uint8)
    return _to_np(result)


# -- WAVELET --------------------------------------------------------------------

def _haar_decompose_1level(image):
    """Dekomposisi Haar 1-level menjadi LL, LH, HL, HH."""
    h, w = image.shape
    h = h - (h % 2)
    w = w - (w % 2)
    img = image[:h, :w]

    inv_sqrt2 = 1.0 / (2.0 ** 0.5)

    low_h = (img[:, 0::2] + img[:, 1::2]) * inv_sqrt2
    high_h = (img[:, 0::2] - img[:, 1::2]) * inv_sqrt2

    LL = (low_h[0::2, :] + low_h[1::2, :]) * inv_sqrt2
    LH = (low_h[0::2, :] - low_h[1::2, :]) * inv_sqrt2
    HL = (high_h[0::2, :] + high_h[1::2, :]) * inv_sqrt2
    HH = (high_h[0::2, :] - high_h[1::2, :]) * inv_sqrt2

    return LL, LH, HL, HH


def _haar_reconstruct_1level(LL, LH, HL, HH):
    """Rekonstruksi inverse Haar 1-level dari 4 subband."""
    inv_sqrt2 = 1.0 / (2.0 ** 0.5)

    h2, w2 = LL.shape
    h, w = h2 * 2, w2 * 2

    low_h = xp.zeros((h, w2), dtype=xp.float64)
    high_h = xp.zeros((h, w2), dtype=xp.float64)

    low_h[0::2, :] = (LL + LH) * inv_sqrt2
    low_h[1::2, :] = (LL - LH) * inv_sqrt2
    high_h[0::2, :] = (HL + HH) * inv_sqrt2
    high_h[1::2, :] = (HL - HH) * inv_sqrt2

    result = xp.zeros((h, w), dtype=xp.float64)
    result[:, 0::2] = (low_h + high_h) * inv_sqrt2
    result[:, 1::2] = (low_h - high_h) * inv_sqrt2

    return result


def _soft_threshold(coeffs, threshold: float):
    """Soft thresholding: sign(x) * max(|x| - threshold, 0)."""
    return xp.sign(coeffs) * xp.maximum(xp.abs(coeffs) - threshold, 0)


def haar_wavelet_denoise(image: np.ndarray, level: int = 1, threshold: float = 20.0) -> np.ndarray:
    """
    Haar wavelet decomposition manual (tanpa PyWavelets).
    Per level: pisahkan LL, LH, HL, HH subband.
    Soft thresholding pada detail coefficients: sign(x) * max(|x|-threshold, 0)
    Rekonstruksi dengan inverse Haar.
    """
    img = _to_xp(image).astype(xp.float64)
    orig_shape = img.shape

    coeffs_list = []
    current = img.copy()
    for _ in range(level):
        LL, LH, HL, HH = _haar_decompose_1level(current)
        LH = _soft_threshold(LH, threshold)
        HL = _soft_threshold(HL, threshold)
        HH = _soft_threshold(HH, threshold)
        coeffs_list.append((LH, HL, HH))
        current = LL

    reconstructed = current
    for LH, HL, HH in reversed(coeffs_list):
        reconstructed = _haar_reconstruct_1level(reconstructed, LH, HL, HH)

    result = reconstructed[:orig_shape[0], :orig_shape[1]]
    result = xp.clip(result, 0, 255).astype(xp.uint8)
    return _to_np(result)


def wavelet_enhance(image: np.ndarray, level: int = 1, boost: float = 1.5) -> np.ndarray:
    """Seperti haar_wavelet_denoise tapi detail coefficients di-boost (x boost), bukan di-threshold."""
    img = _to_xp(image).astype(xp.float64)
    orig_shape = img.shape

    coeffs_list = []
    current = img.copy()
    for _ in range(level):
        LL, LH, HL, HH = _haar_decompose_1level(current)
        LH = LH * boost
        HL = HL * boost
        HH = HH * boost
        coeffs_list.append((LH, HL, HH))
        current = LL

    reconstructed = current
    for LH, HL, HH in reversed(coeffs_list):
        reconstructed = _haar_reconstruct_1level(reconstructed, LH, HL, HH)

    result = reconstructed[:orig_shape[0], :orig_shape[1]]
    result = xp.clip(result, 0, 255).astype(xp.uint8)
    return _to_np(result)


if __name__ == "__main__":
    dummy = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
    dummy_rgb = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)

    tests = {
        "_ensure_grayscale": lambda: _ensure_grayscale(dummy_rgb),
        "normalize_minmax": lambda: normalize_minmax(dummy),
        "normalize_zscore": lambda: normalize_zscore(dummy),
        "histogram_equalization": lambda: histogram_equalization(dummy),
        "clahe": lambda: clahe(dummy, tile_size=8, clip_limit=2.0),
        "gaussian_filter": lambda: gaussian_filter(dummy, 5, 1.0),
        "mean_filter": lambda: mean_filter(dummy, 3),
        "median_filter": lambda: median_filter(dummy, 3),
        "laplacian_sharpen": lambda: laplacian_sharpen(dummy, 0.5),
        "unsharp_mask": lambda: unsharp_mask(dummy, 5, 1.0, 1.5),
        "binary_threshold": lambda: binary_threshold(dummy, 127),
        "otsu_threshold": lambda: otsu_threshold(dummy),
        "adaptive_threshold": lambda: adaptive_threshold(dummy, 11, 2),
        "dilate": lambda: dilate(dummy, 3, 1),
        "erode": lambda: erode(dummy, 3, 1),
        "morphological_opening": lambda: morphological_opening(dummy, 3),
        "morphological_closing": lambda: morphological_closing(dummy, 3),
        "top_hat": lambda: top_hat(dummy, 15),
        "haar_wavelet_denoise": lambda: haar_wavelet_denoise(dummy, 1, 20.0),
        "wavelet_enhance": lambda: wavelet_enhance(dummy, 1, 1.5),
    }

    for name, fn in tests.items():
        try:
            result = fn()
            if isinstance(result, tuple):
                img_result, _ = result
                assert img_result.shape == dummy.shape or name == "_ensure_grayscale"
            elif name == "_ensure_grayscale":
                assert result.shape == (128, 128)
            else:
                assert result.shape == dummy.shape
            print(f"[OK] {name} OK")
        except Exception as exc:
            print(f"[FAIL] {name} FAILED: {exc}")

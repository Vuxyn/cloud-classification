from __future__ import annotations

import numpy as np
import cv2
from scipy.stats import skew, kurtosis

import sys
import os

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

if 'google.colab' in sys.modules or os.path.exists('/usr/local/cuda'):
    os.environ['CUDA_PATH'] = '/usr/local/cuda'

try:
    import cupy as cp
    cp.cuda.Device(0).use()
    # Test compiled operation to ensure CUDA toolkit headers are present and working
    dummy = cp.array([0, 1], dtype=cp.int32)
    cp.bincount(dummy)
    xp = cp
    USING_GPU = True
    print("CuPy aktif - komputasi berjalan di GPU")
except Exception as e:
    xp = np
    USING_GPU = False
    print(f"CuPy tidak aktif atau tidak berfungsi (fallback ke CPU/NumPy): {e}")


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

def histogram_spesifcation(image: np.ndarray) -> np.ndarray:
    """Global HE menggunakan CDF. Hitung histogram -> CDF -> LUT -> apply."""
    img = _to_xp(image).astype(xp.uint8)
    h, w = img.shape

    hist = _histogram_256(img)
    cdf = xp.cumsum(hist)
    total_pixels = h * w
    lut = xp.round(cdf * 255.0 / total_pixels).astype(xp.uint8)

    result = lut[img.ravel()].reshape(h, w)
    return _to_np(result)

if HAS_NUMBA:
    @njit(parallel=True, fastmath=True)
    def _clahe_numba(img, tile_h, tile_w, tile_size, clip_limit):
        h, w = img.shape
        luts = np.zeros((tile_size, tile_size, 256), dtype=np.float64)
        tile_area = tile_h * tile_w
        clip_val = clip_limit * (tile_area / 256.0)

        # 1. Compute LUTs for each tile
        for ti in range(tile_size):
            for tj in range(tile_size):
                y0 = ti * tile_h
                x0 = tj * tile_w
                
                # Compute histogram
                hist = np.zeros(256, dtype=np.float64)
                for y in range(y0, y0 + tile_h):
                    for x in range(x0, x0 + tile_w):
                        val = img[y, x]
                        hist[val] += 1.0
                
                # Clip histogram
                excess = 0.0
                for k in range(256):
                    if hist[k] > clip_val:
                        excess += hist[k] - clip_val
                        hist[k] = clip_val
                
                # Redistribute excess
                val_inc = excess / 256.0
                for k in range(256):
                    hist[k] += val_inc
                    
                # CDF
                cdf = np.zeros(256, dtype=np.float64)
                running_sum = 0.0
                for k in range(256):
                    running_sum += hist[k]
                    cdf[k] = running_sum
                    
                # Find min cdf value > 0
                cdf_min = 0.0
                for k in range(256):
                    if cdf[k] > 0:
                        cdf_min = cdf[k]
                        break
                        
                denom = float(tile_area) - float(cdf_min)
                if denom > 0:
                    for k in range(256):
                        luts[ti, tj, k] = min(max((cdf[k] - cdf_min) / denom * 255.0, 0.0), 255.0)
                else:
                    for k in range(256):
                        luts[ti, tj, k] = float(k)
                        
        # 2. Interpolate for each pixel
        out = np.zeros((h, w), dtype=np.uint8)
        for y in prange(h):
            for x in range(w):
                fy = (float(y) - float(tile_h) / 2.0) / float(tile_h)
                fx = (float(x) - float(tile_w) / 2.0) / float(tile_w)
                
                ti0 = int(np.floor(fy))
                tj0 = int(np.floor(fx))
                
                ti0 = max(0, min(ti0, tile_size - 1))
                tj0 = max(0, min(tj0, tile_size - 1))
                ti1 = min(ti0 + 1, tile_size - 1)
                tj1 = min(tj0 + 1, tile_size - 1)
                
                dy = fy - float(ti0)
                dx = fx - float(tj0)
                
                dy = max(0.0, min(dy, 1.0))
                dx = max(0.0, min(dx, 1.0))
                
                val = img[y, x]
                
                lut00 = luts[ti0, tj0, val]
                lut01 = luts[ti0, tj1, val]
                lut10 = luts[ti1, tj0, val]
                lut11 = luts[ti1, tj1, val]
                
                res = (1.0 - dy) * (1.0 - dx) * lut00 + \
                      (1.0 - dy) * dx * lut01 + \
                      dy * (1.0 - dx) * lut10 + \
                      dy * dx * lut11
                      
                out[y, x] = np.uint8(min(max(res, 0.0), 255.0))
                
        return out


def clahe(image: np.ndarray, tile_size: int = 8, clip_limit: float = 2.0) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization.
    Bagi gambar jadi grid tile_size x tile_size.
    Clip histogram tiap tile di clip_limit x (tile_area / 256).
    HE per tile, interpolasi bilinear antar tile.
    """
    h, w = image.shape
    tile_h = max(1, h // tile_size)
    tile_w = max(1, w // tile_size)
    padded_h = tile_h * tile_size
    padded_w = tile_w * tile_size

    if not USING_GPU and HAS_NUMBA:
        img_uint8 = np.asarray(image, dtype=np.uint8)
        img_crop = img_uint8[:padded_h, :padded_w]
        result = _clahe_numba(img_crop, tile_h, tile_w, tile_size, clip_limit)
        
        final = np.zeros_like(image, dtype=np.uint8)
        final[:padded_h, :padded_w] = result
        if padded_h < h:
            final[padded_h:, :padded_w] = image[padded_h:, :padded_w]
        if padded_w < w:
            final[:padded_h, padded_w:] = image[:padded_h, padded_w:]
        if padded_h < h and padded_w < w:
            final[padded_h:, padded_w:] = image[padded_h:, padded_w:]
        return final

    img = _to_xp(image).astype(xp.float64)
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
    """Gaussian blur manual. Buat kernel 2D: G(x,y)=exp(-(x^2+y^2)/(2*sigma^2)), konvolusi manual.
    Mendukung gambar grayscale (H, W) maupun warna BGR (H, W, 3)."""
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size harus ganjil")

    half = kernel_size // 2
    ax = xp.arange(-half, half + 1, dtype=xp.float64)
    xx, yy = xp.meshgrid(ax, ax)
    kernel = xp.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / kernel.sum()

    if image.ndim == 3:
        channels = []
        for c in range(image.shape[2]):
            ch = _to_xp(image[:, :, c]).astype(xp.float64)
            ch = _convolve2d_xp(ch, kernel)
            ch = xp.clip(ch, 0, 255).astype(xp.uint8)
            channels.append(_to_np(ch))
        return np.stack(channels, axis=2)

    img = _to_xp(image).astype(xp.float64)
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


if HAS_NUMBA:
    @njit(parallel=True, fastmath=True)
    def _median_filter_numba(padded: np.ndarray, h: int, w: int, kernel_size: int) -> np.ndarray:
        result = np.zeros((h, w), dtype=np.float64)
        for i in prange(h):
            for j in range(w):
                flat_patch = padded[i:i + kernel_size, j:j + kernel_size].copy().flatten()
                flat_patch.sort()
                mid = len(flat_patch) // 2
                result[i, j] = flat_patch[mid]
        return result

def median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Median filter manual menggunakan np.median per patch."""
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size harus ganjil")

    if USING_GPU:
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

    if HAS_NUMBA:
        img = np.asarray(image).astype(np.float64)
        pad = kernel_size // 2
        padded = np.pad(img, ((pad, pad), (pad, pad)), mode='reflect')
        h, w = img.shape
        result = _median_filter_numba(padded, h, w, kernel_size)
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result

    img = np.asarray(image).astype(np.float64)
    pad = kernel_size // 2
    padded = np.pad(img, ((pad, pad), (pad, pad)), mode='reflect')
    h, w = img.shape
    result = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            patch = padded[i:i + kernel_size, j:j + kernel_size]
            result[i, j] = np.median(patch)
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


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



def to_grayscale(image: np.ndarray) -> np.ndarray:
    return _ensure_grayscale(image)


def to_hsv(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def clahe_hsv(image: np.ndarray, tile_size: int = 8, clip_limit: float = 2.0) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = clahe(s, tile_size=tile_size, clip_limit=clip_limit)
    v = clahe(v, tile_size=tile_size, clip_limit=clip_limit)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)


def equalize_hsv(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = histogram_equalization(v)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)


def equalize_per_channel(image: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(image)
    b = histogram_equalization(b)
    g = histogram_equalization(g)
    r = histogram_equalization(r)
    return cv2.merge([b, g, r])


def clahe_per_channel(image: np.ndarray, tile_size: int = 8, clip_limit: float = 2.0) -> np.ndarray:
    b, g, r = cv2.split(image)
    b = clahe(b, tile_size=tile_size, clip_limit=clip_limit)
    g = clahe(g, tile_size=tile_size, clip_limit=clip_limit)
    r = clahe(r, tile_size=tile_size, clip_limit=clip_limit)
    return cv2.merge([b, g, r])


def calc_nrbr(image: np.ndarray) -> np.ndarray:
    b = image[:, :, 0].astype(np.float64)
    r = image[:, :, 2].astype(np.float64)
    return (r - b) / (r + b + 1e-8)


def local_binary_pattern(image: np.ndarray, P: int = 8, R: int = 1) -> np.ndarray:
    img = _ensure_grayscale(image)
    img_xp = _to_xp(img).astype(xp.float64)
    h, w = img_xp.shape
    padded = xp.pad(img_xp, ((R, R), ((R, R))), mode='reflect')
    
    offsets = [
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
        (1, 0),
        (1, 1)
    ]
    
    lbp_img = xp.zeros((h, w), dtype=xp.uint8)
    center = padded[R:R+h, R:R+w]
    
    for i, (dy, dx) in enumerate(offsets):
        neighbor = padded[R+dy : R+dy+h, R+dx : R+dx+w]
        mask = (neighbor >= center).astype(xp.uint8)
        lbp_img += mask * (1 << i)
        
    return _to_np(lbp_img)


def calc_histogram(image: np.ndarray, bins: int, range_val: tuple[float, float], density: bool = False) -> np.ndarray:
    img = _to_xp(image).astype(xp.float64)
    min_val, max_val = range_val
    
    if min_val == max_val:
        hist = xp.zeros(bins, dtype=xp.float64)
        hist[0] = float(img.size)
    else:
        flat = img.ravel()
        bin_indices = ((flat - min_val) / (max_val - min_val) * bins).astype(xp.int32)
        bin_indices = xp.clip(bin_indices, 0, bins - 1)
        hist = xp.bincount(bin_indices, minlength=bins).astype(xp.float64)
        
    if density:
        bin_width = (max_val - min_val) / bins
        total = hist.sum()
        if total > 0 and bin_width > 0:
            hist = hist / (total * bin_width)
            
    return _to_np(hist)


def extract_hsv_features(images: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float], list[float], list[float], list[float], list[float], list[float], list[float], list[float], list[float], list[float], list[float], list[float]]:
    h_hists, s_hists, v_hists = [], [], []
    h_means, h_stds, h_skews, h_kurts = [], [], [], []
    s_means, s_stds, s_skews, s_kurts = [], [], [], []
    v_means, v_stds, v_skews, v_kurts = [], [], [], []
    
    for img in images:
        hsv = to_hsv(img)
        
        h_hist = calc_histogram(hsv[:, :, 0], bins=16, range_val=(0, 180), density=True)
        s_hist = calc_histogram(hsv[:, :, 1], bins=8, range_val=(0, 256), density=True)
        v_hist = calc_histogram(hsv[:, :, 2], bins=8, range_val=(0, 256), density=True)
        
        h_hists.append(h_hist)
        s_hists.append(s_hist)
        v_hists.append(v_hist)
        
        h_flat = hsv[:, :, 0].ravel()
        h_means.append(float(np.mean(h_flat)))
        h_stds.append(float(np.std(h_flat)))
        h_skews.append(float(skew(h_flat)))
        h_kurts.append(float(kurtosis(h_flat)))
        
        s_flat = hsv[:, :, 1].ravel()
        s_means.append(float(np.mean(s_flat)))
        s_stds.append(float(np.std(s_flat)))
        s_skews.append(float(skew(s_flat)))
        s_kurts.append(float(kurtosis(s_flat)))
        
        v_flat = hsv[:, :, 2].ravel()
        v_means.append(float(np.mean(v_flat)))
        v_stds.append(float(np.std(v_flat)))
        v_skews.append(float(skew(v_flat)))
        v_kurts.append(float(kurtosis(v_flat)))
        
    return (
        np.array(h_hists), np.array(s_hists), np.array(v_hists),
        h_means, h_stds, h_skews, h_kurts,
        s_means, s_stds, s_skews, s_kurts,
        v_means, v_stds, v_skews, v_kurts
    )


def extract_nrbr_features(images: np.ndarray) -> tuple[np.ndarray, list[float], list[float], list[float], list[float]]:
    nrbr_hists = []
    nrbr_means, nrbr_stds, nrbr_skews, nrbr_kurts = [], [], [], []
    
    for img in images:
        nrbr = calc_nrbr(img)
        
        nrbr_hist = calc_histogram(nrbr, bins=16, range_val=(-1.0, 1.0), density=True)
        nrbr_hists.append(nrbr_hist)
        
        nrbr_flat = nrbr.ravel()
        nrbr_means.append(float(np.mean(nrbr_flat)))
        nrbr_stds.append(float(np.std(nrbr_flat)))
        nrbr_skews.append(float(skew(nrbr_flat)))
        nrbr_kurts.append(float(kurtosis(nrbr_flat)))
        
    return np.array(nrbr_hists), nrbr_means, nrbr_stds, nrbr_skews, nrbr_kurts


def extract_lbp_features(images: np.ndarray, bins: int = 8) -> np.ndarray:
    lbp_hists = []
    for img in images:
        lbp = local_binary_pattern(img)
        lbp_hist = calc_histogram(lbp, bins=bins, range_val=(0, 256), density=True)
        lbp_hists.append(lbp_hist)
    return np.array(lbp_hists)


def extract_grayscale_stats(images_gray: np.ndarray) -> tuple[np.ndarray, list[float], list[float], list[float], list[float]]:
    gray_hists = []
    gray_means, gray_stds, gray_skews, gray_kurts = [], [], [], []
    
    for img in images_gray:
        gray_hist = calc_histogram(img, bins=16, range_val=(0, 256), density=True)
        gray_hists.append(gray_hist)
        
        flat = img.ravel()
        gray_means.append(float(np.mean(flat)))
        gray_stds.append(float(np.std(flat)))
        gray_skews.append(float(skew(flat)))
        gray_kurts.append(float(kurtosis(flat)))
        
    return np.array(gray_hists), gray_means, gray_stds, gray_skews, gray_kurts


def gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Gamma correction untuk citra grayscale maupun multi-channel (BGR/RGB).
    Menggunakan lookup table (LUT) untuk kecepatan maksimal.
    """
    if gamma <= 0:
        gamma = 1.0
        
    inv_gamma = 1.0 / gamma
    
    # Buat lookup table di CPU
    lut_cpu = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    
    if USING_GPU:
        img = _to_xp(image)
        lut_gpu = xp.asarray(lut_cpu)
        result = lut_gpu[img]
        return _to_np(result)
    else:
        # cv2.LUT sangat cepat di CPU dan mendukung citra multi-channel BGR secara as-is
        return cv2.LUT(image, lut_cpu)


def gray_world_white_balance(image: np.ndarray) -> np.ndarray:
    """
    Keseimbangan warna (White Balance) menggunakan asumsi Gray World.
    Rata-rata intensitas masing-masing saluran R, G, B disamakan dengan
    rata-rata keseluruhan dari ketiga saluran tersebut.
    Mendukung input citra berwarna (H, W, 3) baik RGB maupun BGR.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        return image
        
    img = _to_xp(image).astype(xp.float64)
    
    # Hitung rata-rata untuk masing-masing saluran
    mean_0 = img[:, :, 0].mean()
    mean_1 = img[:, :, 1].mean()
    mean_2 = img[:, :, 2].mean()
    
    # Rata-rata target abu-abu
    mean_gray = (mean_0 + mean_1 + mean_2) / 3.0
    
    if mean_gray == 0:
        return image
        
    # Hitung faktor pengali untuk setiap saluran
    scale_0 = mean_gray / (mean_0 + 1e-8)
    scale_1 = mean_gray / (mean_1 + 1e-8)
    scale_2 = mean_gray / (mean_2 + 1e-8)
    
    # Lakukan scaling dan kliping ke rentang [0, 255]
    img_wb = xp.zeros_like(img)
    img_wb[:, :, 0] = xp.clip(img[:, :, 0] * scale_0, 0, 255)
    img_wb[:, :, 1] = xp.clip(img[:, :, 1] * scale_1, 0, 255)
    img_wb[:, :, 2] = xp.clip(img[:, :, 2] * scale_2, 0, 255)
    
    return _to_np(img_wb.astype(xp.uint8))


def bilateral_filter(image: np.ndarray, d: int = 9, sigma_color: float = 75.0, sigma_space: float = 75.0) -> np.ndarray:
    """
    Bilateral filter menggunakan OpenCV untuk smoothing dengan mempertahankan tepi (edge-preserving).
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def extract_color_features(img_bgr: np.ndarray) -> dict:
    """
    Ekstrak fitur warna statistik dari image BGR.
    Returns: dict dengan 9 fitur (mean, std, skewness untuk R, G, B)
    
    Justifikasi:
    - Mean: rata-rata warna dominan per channel
    - Std: variasi warna (tekstur warna)
    - Skewness: distribusi asimetri warna (berguna untuk awan putih vs abu-abu)
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    features = {}
    
    channel_names = ['R', 'G', 'B']
    for ch_idx, ch_name in enumerate(channel_names):
        channel = img_rgb[:, :, ch_idx].astype(np.float64)
        features[f'Color_Mean_{ch_name}'] = float(np.mean(channel))
        features[f'Color_Std_{ch_name}']  = float(np.std(channel))
        features[f'Color_Skew_{ch_name}'] = float(skew(channel.ravel()))
    
    return features


def visualize_pipeline_steps(images: np.ndarray, labels: np.ndarray, pipeline: list, sample_indices: list | None = None) -> None:
    """
    Menampilkan visualisasi hasil preprocessing untuk setiap tahapan di PIPELINE secara otomatis.
    Mendukung visualisasi dinamis untuk N-tahap pipeline.
    """
    import matplotlib.pyplot as plt
    import cv2
    
    class_names = sorted(list(set(labels)))
    
    # 1. Cari 1 sampel per kelas jika belum disediakan
    if sample_indices is None:
        sample_indices = []
        for cls in class_names:
            idx = np.where(labels == cls)[0][0]
            sample_indices.append(idx)
            
    num_classes = len(class_names)
    num_steps = len(pipeline)
    
    # Simpan state gambar untuk setiap tahap
    # Tahap 0: Gambar asli
    current_images = [images[idx].copy() for idx in sample_indices]
    
    # Grid subplot: (num_steps + 1) baris, num_classes kolom
    fig, axes = plt.subplots(num_steps + 1, num_classes, figsize=(num_classes * 2.5, (num_steps + 1) * 2.5))
    
    # Pastikan axes berbentuk 2D array jika num_steps=0 atau num_classes=1
    if num_steps == 0:
        axes = np.expand_dims(axes, axis=0)
    if num_classes == 1:
        axes = np.expand_dims(axes, axis=1)
    if axes.ndim == 1:
        if num_steps + 1 == 1:
            axes = np.expand_dims(axes, axis=0)
        else:
            axes = np.expand_dims(axes, axis=1)
        
    # Helper untuk extract nama fungsi
    def get_step_name(fn, idx):
        co = getattr(fn, '__code__', None)
        if co and co.co_names:
            # Cari nama fungsi yang dipanggil (selain cv2 atau np jika ada)
            names = [n for n in co.co_names if n not in ('cv2', 'np', 'cvtColor', 'COLOR_BGR2GRAY', 'COLOR_BGR2RGB')]
            if names:
                return f"Step {idx}: {names[0]}"
        name = getattr(fn, '__name__', '<lambda>')
        if name == '<lambda>':
            return f"Step {idx}: Preprocessed"
        return f"Step {idx}: {name}"

    # Row 0: Original images
    for i, img in enumerate(current_images):
        if img.ndim == 3:
            img_disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_disp = img
            
        cmap = 'gray' if img.ndim == 2 else None
        axes[0, i].imshow(img_disp, cmap=cmap)
        axes[0, i].set_title(f"{class_names[i]}\nOriginal", fontsize=8)
        axes[0, i].axis('off')
        
    # Row 1 to N: Pipeline steps
    for step_idx, fn in enumerate(pipeline):
        step_name = get_step_name(fn, step_idx + 1)
        
        for i in range(num_classes):
            # Terapkan fungsi dari pipeline ke gambar sampel secara bertahap
            current_images[i] = fn(current_images[i])
            img = current_images[i]
            
            # Konversi ke RGB jika berwarna
            if img.ndim == 3:
                img_disp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_disp = img
                
            cmap = 'gray' if img.ndim == 2 else None
            axes[step_idx + 1, i].imshow(img_disp, cmap=cmap)
            axes[step_idx + 1, i].set_title(f"{step_name}", fontsize=8)
            axes[step_idx + 1, i].axis('off')
            
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dummy = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
    dummy_rgb = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)

    tests = {
        "extract_color_features": lambda: extract_color_features(dummy_rgb),
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
        "to_grayscale": lambda: to_grayscale(dummy_rgb),
        "to_hsv": lambda: to_hsv(dummy_rgb),
        "clahe_hsv": lambda: clahe_hsv(dummy_rgb),
        "equalize_hsv": lambda: equalize_hsv(dummy_rgb),
        "equalize_per_channel": lambda: equalize_per_channel(dummy_rgb),
        "clahe_per_channel": lambda: clahe_per_channel(dummy_rgb),
        "calc_nrbr": lambda: calc_nrbr(dummy_rgb),
        "gamma_correction": lambda: gamma_correction(dummy, 1.2),
        "local_binary_pattern": lambda: local_binary_pattern(dummy),
        "calc_histogram": lambda: calc_histogram(dummy, 16, (0, 256)),
        "extract_hsv_features": lambda: extract_hsv_features(np.array([dummy_rgb])),
        "extract_nrbr_features": lambda: extract_nrbr_features(np.array([dummy_rgb])),
        "extract_lbp_features": lambda: extract_lbp_features(np.array([dummy])),
        "extract_grayscale_stats": lambda: extract_grayscale_stats(np.array([dummy])),
    }

    for name, fn in tests.items():
        try:
            result = fn()
            if name == "extract_color_features":
                assert isinstance(result, dict) and len(result) == 9
            elif name == "to_hsv":
                assert result.shape == (128, 128, 3)
            elif name == "calc_histogram":
                assert result.shape == (16,)
            elif name == "extract_lbp_features":
                assert result.shape == (1, 8)
            elif name in ("clahe_hsv", "equalize_hsv", "equalize_per_channel", "clahe_per_channel"):
                assert result.shape == (128, 128, 3)
            elif isinstance(result, tuple):
                img_result = result[0]
                assert img_result.shape == dummy.shape or name == "_ensure_grayscale" or len(result) > 2
            elif name == "_ensure_grayscale" or name == "to_grayscale" or name == "local_binary_pattern":
                assert result.shape == (128, 128)
            else:
                assert result.shape == dummy.shape
            print(f"[OK] {name} OK")
        except Exception as exc:
            import traceback
            print(f"[FAIL] {name} FAILED: {exc}")
            traceback.print_exc()

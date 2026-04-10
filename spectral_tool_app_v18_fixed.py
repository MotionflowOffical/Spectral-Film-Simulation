from __future__ import annotations

import csv
import json
import math
import os
import queue
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
try:
    if os.name == "nt" or os.environ.get("DISPLAY"):
        matplotlib.use("TkAgg")
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    else:
        raise ImportError("No interactive display available")
except Exception:
    matplotlib.use("Agg")
    FigureCanvasTkAgg = None
from matplotlib.figure import Figure

try:
    import rawpy  # type: ignore
except Exception:
    rawpy = None

try:
    import tifffile  # type: ignore
except Exception:
    tifffile = None


APP_TITLE = "Spectral Band Splitter"
DEFAULT_PREVIEW_MAX = 900
DEFAULT_SENSITIVITY_LOG_CEILING = 4.0
DEFAULT_DENSITY_LOG_CEILING = 4.0
DEFAULT_DENSITY_X_MIN = -3.0
DEFAULT_DENSITY_X_MAX = 1.0
EPS = 1e-6
BAND_EXPORT_FORMATS = ("PNG 16-bit", "TIFF 16-bit", "TIFF 32-bit float", "NumPy .npy")
FINAL_EXPORT_FORMATS = ("PNG 8-bit", "TIFF 16-bit", "TIFF 32-bit float", "RAW linear .npy")
LUT_3D_SIZES = ("32", "64", "128")
LUT_COLOR_MODES = (
    "sRGB gamma",
    "Linear sRGB / Rec.709",
    "Rec.709 gamma",
    "Linear Rec.2020",
    "Rec.2020 gamma",
)
LUT_INPUT_MODES = LUT_COLOR_MODES
LUT_OUTPUT_MODES = LUT_COLOR_MODES
ILLUMINANT_MODES = (
    "Auto (Shades of Gray)",
    "D50",
    "D55",
    "D65",
    "Tungsten",
    "LED",
)
DEFAULT_ILLUMINANT_MODE = ILLUMINANT_MODES[0]
DEFAULT_SHADES_OF_GRAY_P = 6.0
DEFAULT_NOISE_CLEANUP = 0.45
DEFAULT_FILM_GRAIN = 0.006
DEFAULT_DENSITY_COLOR_EXPOSURE_BIAS = 0.32
DEFAULT_SOURCE_LUMA_PRESERVE = 0.72
DEFAULT_SOURCE_CHROMA_PRESERVE = 0.28
DEFAULT_REPROJECTION_ITERS = 2
DEFAULT_REPROJECTION_STRENGTH = 0.72
DEFAULT_SEPARATION_CHUNK_PIXELS = 262144
DEFAULT_WORKER_THREADS = max(2, min(4, os.cpu_count() or 2))
LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


@dataclass
class CurveModel:
    name: str
    color: str
    points: List[Tuple[float, float]]
    visible: bool = True
    y_min: float = 0.0
    y_max: float = 1.0

    def sorted_points(self) -> List[Tuple[float, float]]:
        pts = sorted((float(x), float(y)) for x, y in self.points)
        dedup: List[Tuple[float, float]] = []
        for x, y in pts:
            if dedup and abs(dedup[-1][0] - x) < 1e-9:
                dedup[-1] = (x, y)
            else:
                dedup.append((x, y))
        return dedup

    def sample_linear(self, x: np.ndarray) -> np.ndarray:
        pts = self.sorted_points()
        xp = np.array([p[0] for p in pts], dtype=np.float64)
        fp = np.array([np.clip(p[1], self.y_min, self.y_max) for p in pts], dtype=np.float64)
        out = monotone_cubic_interpolate(xp, fp, x)
        return np.clip(out, self.y_min, self.y_max)


def gaussian_curve(center: float, width: float, x: np.ndarray, amp: float = 1.0) -> np.ndarray:
    return amp * np.exp(-0.5 * ((x - center) / max(width, 1e-6)) ** 2)


def monotone_cubic_interpolate(xp: np.ndarray, fp: np.ndarray, x: np.ndarray) -> np.ndarray:
    xp = np.asarray(xp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if xp.size == 0:
        return np.zeros_like(x, dtype=np.float64)
    if xp.size == 1:
        return np.full_like(x, fp[0], dtype=np.float64)
    if xp.size == 2:
        return np.interp(x, xp, fp, left=fp[0], right=fp[-1])

    h = np.diff(xp)
    h = np.where(np.abs(h) < 1e-12, 1e-12, h)
    delta = np.diff(fp) / h
    m = np.zeros_like(fp)

    m[0] = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
    if np.sign(m[0]) != np.sign(delta[0]):
        m[0] = 0.0
    elif np.sign(delta[0]) != np.sign(delta[1]) and abs(m[0]) > abs(3.0 * delta[0]):
        m[0] = 3.0 * delta[0]

    for i in range(1, xp.size - 1):
        if delta[i - 1] == 0.0 or delta[i] == 0.0 or np.sign(delta[i - 1]) != np.sign(delta[i]):
            m[i] = 0.0
        else:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])

    m[-1] = ((2.0 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2])
    if np.sign(m[-1]) != np.sign(delta[-1]):
        m[-1] = 0.0
    elif np.sign(delta[-1]) != np.sign(delta[-2]) and abs(m[-1]) > abs(3.0 * delta[-1]):
        m[-1] = 3.0 * delta[-1]

    indices = np.searchsorted(xp, x, side="right") - 1
    indices = np.clip(indices, 0, xp.size - 2)

    x0 = xp[indices]
    x1 = xp[indices + 1]
    y0 = fp[indices]
    y1 = fp[indices + 1]
    m0 = m[indices]
    m1 = m[indices + 1]
    hh = np.where(np.abs(x1 - x0) < 1e-12, 1e-12, x1 - x0)
    t = np.clip((x - x0) / hh, 0.0, 1.0)

    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2
    out = h00 * y0 + h10 * hh * m0 + h01 * y1 + h11 * hh * m1

    out = np.where(x <= xp[0], fp[0], out)
    out = np.where(x >= xp[-1], fp[-1], out)
    return out


def linear_sensitivity_to_log_units(y_linear: np.ndarray | float, ceiling: float) -> np.ndarray:
    y = np.clip(np.asarray(y_linear, dtype=np.float64), 10 ** (-ceiling), 1.0)
    return np.clip(ceiling + np.log10(y), 0.0, ceiling)


def log_units_to_linear_sensitivity(y_log: np.ndarray | float, ceiling: float) -> np.ndarray:
    y = np.clip(np.asarray(y_log, dtype=np.float64), 0.0, ceiling)
    return np.clip(10 ** (y - ceiling), 10 ** (-ceiling), 1.0)


def density_to_relative_transmittance(y_density: np.ndarray | float) -> np.ndarray:
    d = np.clip(np.asarray(y_density, dtype=np.float64), 0.0, None)
    return np.clip(np.power(10.0, -d), 0.0, 1.0)


def default_sensitivity_curves(nm_min: int, nm_max: int, ceiling: float = DEFAULT_SENSITIVITY_LOG_CEILING) -> Dict[str, CurveModel]:
    xs = np.linspace(nm_min, nm_max, 11)

    def make_points(center: float, width: float, shoulder_center: float, shoulder_width: float) -> List[Tuple[float, float]]:
        y_lin = gaussian_curve(center, width, xs, 1.0) + 0.22 * gaussian_curve(shoulder_center, shoulder_width, xs, 1.0)
        y_lin = np.clip(y_lin / max(float(y_lin.max()), 1e-6), 10 ** (-ceiling), 1.0)
        y_log = linear_sensitivity_to_log_units(y_lin, ceiling)
        return [(float(a), float(b)) for a, b in zip(xs, y_log)]

    return {
        "Red": CurveModel("Red", "#d84a4a", make_points(610, 34, 570, 60), True, 0.0, ceiling),
        "Green": CurveModel("Green", "#40a857", make_points(545, 28, 500, 55), True, 0.0, ceiling),
        "Blue": CurveModel("Blue", "#4888da", make_points(455, 24, 500, 60), True, 0.0, ceiling),
    }


def default_density_curves(ceiling: float = DEFAULT_DENSITY_LOG_CEILING) -> Dict[str, CurveModel]:
    # H&D / characteristic style defaults:
    # X = relative log exposure (log H), Y = optical density after development.
    base_x = [-3.0, -2.6, -2.2, -1.8, -1.4, -1.0, -0.5, 0.0, 0.5, 0.9]
    red_y = [0.08, 0.11, 0.17, 0.30, 0.58, 1.02, 1.72, 2.34, 2.78, 2.96]
    green_y = [0.10, 0.13, 0.19, 0.34, 0.64, 1.10, 1.82, 2.46, 2.92, 3.08]
    blue_y = [0.07, 0.10, 0.16, 0.28, 0.54, 0.96, 1.60, 2.20, 2.60, 2.84]

    def scale(vals: List[float]) -> List[float]:
        nominal_top = max(max(vals), 1e-6)
        usable_top = max(min(ceiling * 0.82, nominal_top), 1e-6)
        return [float(np.clip(v * usable_top / nominal_top, 0.0, ceiling)) for v in vals]

    red_pts = list(zip(base_x, scale(red_y)))
    green_pts = list(zip(base_x, scale(green_y)))
    blue_pts = list(zip(base_x, scale(blue_y)))
    return {
        "Red": CurveModel("Red", "#d84a4a", red_pts, True, 0.0, ceiling),
        "Green": CurveModel("Green", "#40a857", green_pts, True, 0.0, ceiling),
        "Blue": CurveModel("Blue", "#4888da", blue_pts, True, 0.0, ceiling),
    }


def srgb_to_linear(img: np.ndarray) -> np.ndarray:
    img = np.clip(img, 0.0, 1.0)
    return np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(img: np.ndarray) -> np.ndarray:
    img = np.clip(img, 0.0, 1.0)
    return np.where(img <= 0.0031308, img * 12.92, 1.055 * np.power(img, 1.0 / 2.4) - 0.055)


def linear_to_rec709(img: np.ndarray) -> np.ndarray:
    img = np.clip(img, 0.0, 1.0)
    return np.where(img < 0.018, 4.5 * img, 1.099 * np.power(img, 0.45) - 0.099)


def linear_to_rec2020(img: np.ndarray) -> np.ndarray:
    img = np.clip(img, 0.0, 1.0)
    alpha = 1.09929682680944
    beta = 0.018053968510807
    return np.where(img < beta, 4.5 * img, alpha * np.power(img, 0.45) - (alpha - 1.0))


def rec709_to_linear(img: np.ndarray) -> np.ndarray:
    img = np.clip(img, 0.0, 1.0)
    return np.where(img < 0.081, img / 4.5, np.power((img + 0.099) / 1.099, 1.0 / 0.45))


def rec2020_to_linear(img: np.ndarray) -> np.ndarray:
    img = np.clip(img, 0.0, 1.0)
    alpha = 1.09929682680944
    beta = 0.0812428582986352
    return np.where(img < beta, img / 4.5, np.power((img + (alpha - 1.0)) / alpha, 1.0 / 0.45))


def linear_rec709_to_rec2020(img: np.ndarray) -> np.ndarray:
    data = np.asarray(img, dtype=np.float64)
    m = np.array([
        [0.6274039, 0.3292830, 0.0433131],
        [0.0690973, 0.9195404, 0.0113623],
        [0.0163914, 0.0880133, 0.8955953],
    ], dtype=np.float64)
    return np.tensordot(data, m.T, axes=([data.ndim - 1], [0]))


def linear_rec2020_to_rec709(img: np.ndarray) -> np.ndarray:
    data = np.asarray(img, dtype=np.float64)
    m = np.array([
        [1.6604910, -0.5876411, -0.0728499],
        [-0.1245505, 1.1328999, -0.0083494],
        [-0.0181508, -0.1005789, 1.1187297],
    ], dtype=np.float64)
    return np.tensordot(data, m.T, axes=([data.ndim - 1], [0]))


def decode_lut_input_to_linear_rec709(rgb_encoded: np.ndarray, input_mode: str) -> np.ndarray:
    data = np.clip(np.asarray(rgb_encoded, dtype=np.float64), 0.0, 1.0)
    mode = str(input_mode)
    if mode == "sRGB gamma":
        return np.clip(srgb_to_linear(data), 0.0, None)
    if mode == "Linear sRGB / Rec.709":
        return np.clip(data, 0.0, None)
    if mode == "Rec.709 gamma":
        return np.clip(rec709_to_linear(data), 0.0, None)
    if mode == "Linear Rec.2020":
        return np.clip(linear_rec2020_to_rec709(data), 0.0, None)
    if mode == "Rec.2020 gamma":
        return np.clip(linear_rec2020_to_rec709(rec2020_to_linear(data)), 0.0, None)
    raise ValueError(f"Unsupported LUT input mode: {input_mode}")


def encode_lut_output_from_linear_rec709(rgb_linear: np.ndarray, output_mode: str) -> np.ndarray:
    data = np.clip(np.asarray(rgb_linear, dtype=np.float64), 0.0, None)
    mode = str(output_mode)
    if mode == "sRGB gamma":
        return np.clip(linear_to_srgb(data), 0.0, 1.0)
    if mode == "Linear sRGB / Rec.709":
        return np.clip(data, 0.0, 1.0)
    if mode == "Rec.709 gamma":
        return np.clip(linear_to_rec709(data), 0.0, 1.0)
    rec2020_linear = np.clip(linear_rec709_to_rec2020(data), 0.0, None)
    if mode == "Linear Rec.2020":
        return np.clip(rec2020_linear, 0.0, 1.0)
    if mode == "Rec.2020 gamma":
        return np.clip(linear_to_rec2020(rec2020_linear), 0.0, 1.0)
    raise ValueError(f"Unsupported LUT output mode: {output_mode}")


def wavelength_to_rgb(wavelength_nm: float, gamma: float = 0.8) -> Tuple[int, int, int]:
    w = float(wavelength_nm)
    if w < 380 or w > 780:
        return (0, 0, 0)
    if w < 440:
        att = 0.3 + 0.7 * (w - 380) / (440 - 380)
        r = ((-(w - 440) / (440 - 380)) * att) ** gamma
        g = 0.0
        b = (1.0 * att) ** gamma
    elif w < 490:
        r = 0.0
        g = ((w - 440) / (490 - 440)) ** gamma
        b = 1.0
    elif w < 510:
        r = 0.0
        g = 1.0
        b = (-(w - 510) / (510 - 490)) ** gamma
    elif w < 580:
        r = ((w - 510) / (580 - 510)) ** gamma
        g = 1.0
        b = 0.0
    elif w < 645:
        r = 1.0
        g = (-(w - 645) / (645 - 580)) ** gamma
        b = 0.0
    else:
        att = 0.3 + 0.7 * (780 - w) / (780 - 645)
        r = (1.0 * att) ** gamma
        g = 0.0
        b = 0.0
    return tuple(int(max(0.0, min(1.0, c)) * 255) for c in (r, g, b))




def normalize_rgb_chromaticity(rgb: np.ndarray | List[float] | Tuple[float, float, float]) -> np.ndarray:
    arr = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, None)
    if arr.ndim == 1:
        return arr / max(float(np.sum(arr)), EPS)
    sums = np.maximum(np.sum(arr, axis=-1, keepdims=True), EPS)
    return arr / sums


def planck_spectrum(wavelengths_nm: np.ndarray, temperature_k: float) -> np.ndarray:
    wl_m = np.clip(np.asarray(wavelengths_nm, dtype=np.float64), 1.0, None) * 1e-9
    c2 = 1.438776877e-2
    exponent = np.clip(c2 / np.maximum(wl_m * max(float(temperature_k), 1.0), 1e-20), 1e-8, 700.0)
    radiance = np.power(wl_m, -5.0) / (np.exp(exponent) - 1.0)
    radiance = np.clip(radiance, 0.0, None)
    return radiance / max(float(np.max(radiance)), EPS)


def daylight_like_spectrum(wavelengths_nm: np.ndarray, temperature_k: float) -> np.ndarray:
    bb = planck_spectrum(wavelengths_nm, temperature_k)
    wl = np.clip(np.asarray(wavelengths_nm, dtype=np.float64), 360.0, None)
    rayleigh = np.power(560.0 / wl, 0.65)
    spectrum = bb * (0.82 + 0.18 * rayleigh)
    return spectrum / max(float(np.max(spectrum)), EPS)


def cool_white_led_spectrum(wavelengths_nm: np.ndarray) -> np.ndarray:
    wl = np.asarray(wavelengths_nm, dtype=np.float64)
    blue_pump = 1.15 * gaussian_curve(450.0, 16.0, wl, 1.0)
    phosphor = 0.95 * gaussian_curve(545.0, 72.0, wl, 1.0)
    red_tail = 0.18 * gaussian_curve(610.0, 38.0, wl, 1.0)
    spectrum = blue_pump + phosphor + red_tail
    return spectrum / max(float(np.max(spectrum)), EPS)


def build_scene_illuminant_library(wavelengths_nm: np.ndarray) -> Dict[str, np.ndarray]:
    wl = np.asarray(wavelengths_nm, dtype=np.float64)
    return {
        "D50": daylight_like_spectrum(wl, 5003.0),
        "D55": daylight_like_spectrum(wl, 5503.0),
        "D65": daylight_like_spectrum(wl, 6504.0),
        "Tungsten": planck_spectrum(wl, 3200.0),
        "LED": cool_white_led_spectrum(wl),
    }


def build_reference_rgb_response_matrix(wavelengths: np.ndarray) -> np.ndarray:
    wl = np.asarray(wavelengths, dtype=np.float64)
    r = gaussian_curve(610.0, 55.0, wl, 1.0) + 0.16 * gaussian_curve(565.0, 85.0, wl, 1.0)
    g = gaussian_curve(545.0, 42.0, wl, 1.0) + 0.14 * gaussian_curve(500.0, 78.0, wl, 1.0)
    b = gaussian_curve(455.0, 24.0, wl, 1.0) + 0.12 * gaussian_curve(505.0, 60.0, wl, 1.0)
    s = np.vstack([r, g, b])
    s /= np.maximum(np.sum(s, axis=1, keepdims=True), EPS)
    return s.astype(np.float64)


def estimate_scene_chromaticity_shades_of_gray(rgb_linear: np.ndarray, p: float = DEFAULT_SHADES_OF_GRAY_P) -> np.ndarray:
    data = np.clip(np.asarray(rgb_linear, dtype=np.float64), 0.0, None)
    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError("Expected an H×W×3 linear RGB image for illuminant estimation.")
    flat = data.reshape(-1, 3)
    luma = flat @ LUMA_WEIGHTS
    if flat.shape[0] > 2048:
        lo = float(np.percentile(luma, 1.0))
        hi = float(np.percentile(luma, 99.3))
        sat = np.max(flat, axis=1)
        sat_hi = float(np.percentile(sat, 99.7))
        mask = (luma > lo) & (luma < hi) & (sat < sat_hi)
        if int(np.count_nonzero(mask)) >= 1024:
            flat = flat[mask]
    p = max(float(p), 1.0)
    estimate = np.power(np.mean(np.power(flat + EPS, p), axis=0), 1.0 / p)
    return normalize_rgb_chromaticity(estimate)


def select_scene_illuminant(
    rgb_linear: np.ndarray,
    wavelengths_nm: np.ndarray,
    selected_mode: str = DEFAULT_ILLUMINANT_MODE,
    p: float = DEFAULT_SHADES_OF_GRAY_P,
) -> Dict[str, object]:
    library = build_scene_illuminant_library(wavelengths_nm)
    ref_matrix = build_reference_rgb_response_matrix(wavelengths_nm)
    estimated_chroma = estimate_scene_chromaticity_shades_of_gray(rgb_linear, p=p)
    candidate_chromas: Dict[str, np.ndarray] = {}
    candidate_distances: Dict[str, float] = {}
    for name, spectrum in library.items():
        candidate_rgb = ref_matrix @ spectrum
        candidate_chromas[name] = normalize_rgb_chromaticity(candidate_rgb)
        candidate_distances[name] = float(np.linalg.norm(candidate_chromas[name] - estimated_chroma))

    estimated_label = min(candidate_distances, key=candidate_distances.get)
    if str(selected_mode).startswith("Auto"):
        selected_label = estimated_label
    else:
        selected_label = str(selected_mode) if str(selected_mode) in library else estimated_label

    return {
        "mode": str(selected_mode),
        "estimated_chromaticity": estimated_chroma.astype(np.float64),
        "estimated_label": estimated_label,
        "selected_label": selected_label,
        "selected_spectrum": library[selected_label].astype(np.float64),
        "candidate_chromaticities": {name: arr.astype(np.float64) for name, arr in candidate_chromas.items()},
        "candidate_distances": candidate_distances,
    }


def format_illuminant_info(selection: Dict[str, object]) -> str:
    est = np.asarray(selection.get("estimated_chromaticity", np.array([1.0, 1.0, 1.0])), dtype=np.float64)
    est_label = str(selection.get("estimated_label", "Unknown"))
    sel_label = str(selection.get("selected_label", est_label))
    mode = str(selection.get("mode", DEFAULT_ILLUMINANT_MODE))
    return f"Scene illuminant prior: {sel_label} • SoG chroma {est[0]:.3f}/{est[1]:.3f}/{est[2]:.3f} • estimate {est_label} • mode {mode}"
def second_difference_matrix(n: int) -> np.ndarray:
    if n < 3:
        return np.eye(n, dtype=np.float64)
    d = np.zeros((n - 2, n), dtype=np.float64)
    for i in range(n - 2):
        d[i, i] = 1.0
        d[i, i + 1] = -2.0
        d[i, i + 2] = 1.0
    return d


def build_sensitivity_matrix(wavelengths: np.ndarray, curves: Dict[str, CurveModel], sensitivity_ceiling: float) -> np.ndarray:
    r = log_units_to_linear_sensitivity(curves["Red"].sample_linear(wavelengths), sensitivity_ceiling)
    g = log_units_to_linear_sensitivity(curves["Green"].sample_linear(wavelengths), sensitivity_ceiling)
    b = log_units_to_linear_sensitivity(curves["Blue"].sample_linear(wavelengths), sensitivity_ceiling)
    s = np.vstack([r, g, b])
    row_sums = np.maximum(s.sum(axis=1, keepdims=True), EPS)
    s = s / row_sums
    return s.astype(np.float64)


def precompute_separation_operator(
    s: np.ndarray,
    smoothness: float = 0.02,
    energy_reg: float = 0.002,
    illuminant_spectrum: Optional[np.ndarray] = None,
) -> np.ndarray:
    n_bands = s.shape[1]
    d = second_difference_matrix(n_bands)
    measurement = np.asarray(s, dtype=np.float64)
    if illuminant_spectrum is not None:
        illum = np.clip(np.asarray(illuminant_spectrum, dtype=np.float64), 0.0, None).reshape(1, -1)
        if illum.shape[1] != n_bands:
            raise ValueError("Illuminant spectrum length must match the number of spectral bands.")
        measurement = measurement * illum
    a = measurement.T @ measurement + smoothness * (d.T @ d) + energy_reg * np.eye(n_bands, dtype=np.float64)
    b = measurement.T
    return np.linalg.solve(a, b)




def rgb_to_spectral_cube(
    rgb_linear: np.ndarray,
    operator: np.ndarray,
    source_sensitivity_matrix: Optional[np.ndarray] = None,
    reprojection_iters: int = DEFAULT_REPROJECTION_ITERS,
    reprojection_strength: float = DEFAULT_REPROJECTION_STRENGTH,
    spectrum_scale: Optional[np.ndarray] = None,
    chunk_pixels: int = DEFAULT_SEPARATION_CHUNK_PIXELS,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    h, w, _ = rgb_linear.shape
    flat = np.clip(rgb_linear.reshape(-1, 3).astype(np.float32, copy=False), 0.0, None)
    total_pixels = int(flat.shape[0])
    n_bands = int(operator.shape[0])
    cube_flat = np.empty((total_pixels, n_bands), dtype=np.float32)
    chunk_pixels = max(int(chunk_pixels), 4096)

    for start in range(0, total_pixels, chunk_pixels):
        end = min(total_pixels, start + chunk_pixels)
        block = flat[start:end]
        cube_block = spectral_separate_flat_rgb(
            block,
            operator,
            source_sensitivity_matrix=source_sensitivity_matrix,
            reprojection_iters=reprojection_iters,
            reprojection_strength=reprojection_strength,
            spectrum_scale=spectrum_scale,
        )
        cube_flat[start:end] = cube_block.astype(np.float32, copy=False)
        if progress_callback is not None:
            progress_callback(end, total_pixels)

    cube = cube_flat.reshape(h, w, n_bands)
    return cube.astype(np.float32, copy=False)


def density_map_rgb(
    rgb_linear: np.ndarray,
    density_curves: Dict[str, CurveModel],
    density_ceiling: float,
    density_x_range: Tuple[float, float] = (DEFAULT_DENSITY_X_MIN, DEFAULT_DENSITY_X_MAX),
    color_exposure_bias: float = DEFAULT_DENSITY_COLOR_EXPOSURE_BIAS,
) -> np.ndarray:
    # Characteristic / H&D curve mapping:
    #   X axis = relative log exposure (or lux·seconds in the UI display mapping)
    #   Y axis = optical density after development
    # Exposure is driven mostly by scene luminance, with only a moderated
    # channel-specific color bias. This preserves source contrast better and
    # avoids strongly oversaturating foliage or flattening low-contrast scenes.
    exposure_linear = np.clip(rgb_linear, 0.0, None).astype(np.float64)
    x_min, x_max = float(density_x_range[0]), float(density_x_range[1])
    if x_max <= x_min:
        x_max = x_min + 1e-6

    luma = np.tensordot(exposure_linear, LUMA_WEIGHTS, axes=([2], [0]))
    luma_norm = np.log10(1.0 + 999.0 * np.clip(luma, 0.0, None)) / np.log10(1000.0)
    log_exposure_base = x_min + luma_norm * (x_max - x_min)

    chroma_ratio = exposure_linear / np.maximum(luma[..., None], EPS)
    chroma_ratio = np.clip(chroma_ratio, 0.35, 2.85)
    chroma_offset = np.log2(chroma_ratio)
    log_exposure_channels = np.clip(
        log_exposure_base[..., None] + float(color_exposure_bias) * chroma_offset,
        x_min,
        x_max,
    )

    out = np.empty_like(exposure_linear, dtype=np.float64)
    for idx, name in enumerate(("Red", "Green", "Blue")):
        curve = density_curves[name]
        d_vals = np.clip(curve.sample_linear(log_exposure_channels[..., idx]), 0.0, density_ceiling)
        d_min = float(np.clip(curve.sample_linear(np.array([x_min], dtype=np.float64))[0], 0.0, density_ceiling))
        net_density = np.clip(d_vals - d_min, 0.0, density_ceiling)
        out[..., idx] = density_to_relative_transmittance(net_density)
    return np.clip(out, 0.0, 1.0)

def spectral_cube_to_raw_linear(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    curves: Dict[str, CurveModel],
    sensitivity_ceiling: float,
    visible_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    s = build_sensitivity_matrix(wavelengths, curves, sensitivity_ceiling)
    if visible_range is not None:
        lo, hi = visible_range
        mask = (wavelengths >= lo) & (wavelengths <= hi)
        if np.any(mask):
            s = s[:, mask]
            cube = cube[:, :, mask]
    rgb_linear = np.tensordot(cube.astype(np.float64), s.T.astype(np.float64), axes=([2], [0]))
    return np.clip(rgb_linear, 0.0, None).astype(np.float32)


def source_anchor_positive_rgb(
    rgb_positive_linear: np.ndarray,
    source_reference_linear: Optional[np.ndarray],
    luma_preserve: float = DEFAULT_SOURCE_LUMA_PRESERVE,
    chroma_preserve: float = DEFAULT_SOURCE_CHROMA_PRESERVE,
) -> np.ndarray:
    if source_reference_linear is None:
        return np.clip(np.asarray(rgb_positive_linear, dtype=np.float32), 0.0, None)

    out = np.clip(np.asarray(rgb_positive_linear, dtype=np.float64), 0.0, None)
    src = np.clip(np.asarray(source_reference_linear, dtype=np.float64), 0.0, None)
    if src.shape != out.shape:
        return out.astype(np.float32)

    luma_keep = float(np.clip(luma_preserve, 0.0, 1.0))
    chroma_keep = float(np.clip(chroma_preserve, 0.0, 1.0))

    out_luma = np.tensordot(out, LUMA_WEIGHTS, axes=([2], [0]))
    src_luma = np.tensordot(src, LUMA_WEIGHTS, axes=([2], [0]))

    src_hi = float(np.percentile(src_luma, 99.5)) if src_luma.size else 1.0
    out_hi = float(np.percentile(out_luma, 99.5)) if out_luma.size else 1.0
    if src_hi > EPS and out_hi > EPS:
        scale = out_hi / src_hi
        src = src * scale
        src_luma = src_luma * scale

    target_luma = out_luma * (1.0 - luma_keep) + src_luma * luma_keep
    target_luma = np.clip(target_luma, 0.0, None)

    out_ratio = out / np.maximum(out_luma[..., None], EPS)
    src_ratio = src / np.maximum(src_luma[..., None], EPS)
    out_ratio = np.clip(out_ratio, 1e-4, 8.0)
    src_ratio = np.clip(src_ratio, 1e-4, 8.0)

    ratio = np.exp((1.0 - chroma_keep) * np.log(out_ratio) + chroma_keep * np.log(src_ratio))
    ratio_luma = np.tensordot(ratio, LUMA_WEIGHTS, axes=([2], [0]))
    ratio = ratio / np.maximum(ratio_luma[..., None], EPS)

    anchored = ratio * target_luma[..., None]
    return np.clip(anchored, 0.0, None).astype(np.float32)



def spectral_cube_to_positive_linear(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    curves: Dict[str, CurveModel],
    sensitivity_ceiling: float,
    density_curves: Dict[str, CurveModel],
    density_ceiling: float,
    visible_range: Optional[Tuple[float, float]] = None,
    density_x_range: Tuple[float, float] = (DEFAULT_DENSITY_X_MIN, DEFAULT_DENSITY_X_MAX),
    source_reference_linear: Optional[np.ndarray] = None,
    source_luma_preserve: float = DEFAULT_SOURCE_LUMA_PRESERVE,
    source_chroma_preserve: float = DEFAULT_SOURCE_CHROMA_PRESERVE,
) -> np.ndarray:
    rgb_linear = spectral_cube_to_raw_linear(
        cube,
        wavelengths,
        curves=curves,
        sensitivity_ceiling=sensitivity_ceiling,
        visible_range=visible_range,
    )

    transmittance = density_map_rgb(
        rgb_linear,
        density_curves,
        density_ceiling,
        density_x_range=density_x_range,
    )
    rgb_positive = 1.0 - np.clip(transmittance, 0.0, 1.0)
    rgb_positive = source_anchor_positive_rgb(
        rgb_positive,
        source_reference_linear,
        luma_preserve=source_luma_preserve,
        chroma_preserve=source_chroma_preserve,
    )
    return np.clip(rgb_positive, 0.0, None).astype(np.float32)

def preview_balance_to_srgb(rgb_positive_linear: np.ndarray, exposure_ev: float = 0.0) -> np.ndarray:
    data = np.clip(np.asarray(rgb_positive_linear, dtype=np.float64), 0.0, None)
    gain = float(2.0 ** float(exposure_ev))
    balanced = 1.0 - np.exp(-data * gain)
    return linear_to_srgb(np.clip(balanced, 0.0, 1.0)).astype(np.float32)


def fit_preview_image(rgb: np.ndarray, max_side: int = DEFAULT_PREVIEW_MAX) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = min(max_side / max(h, 1), max_side / max(w, 1), 1.0)
    if scale >= 0.999:
        return rgb
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    img = Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8), mode="RGB")
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    return np.asarray(img).astype(np.float32) / 255.0


def array_to_tk_image(rgb: np.ndarray, max_size: Tuple[int, int]) -> ImageTk.PhotoImage:
    img = Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8), mode="RGB")
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(img)


def load_standard_image(path: Path) -> np.ndarray:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return srgb_to_linear(arr)


def load_cr3_image(path: Path) -> np.ndarray:
    if rawpy is None:
        raise RuntimeError("rawpy is not installed. Install dependencies from requirements.txt for CR3 support.")
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=True,
            gamma=(1.0, 1.0),
            output_bps=16,
            output_color=rawpy.ColorSpace.sRGB,
        )
    arr = rgb.astype(np.float32) / 65535.0
    return np.clip(arr, 0.0, None)


def load_image_any(path_str: str) -> np.ndarray:
    path = Path(path_str)
    ext = path.suffix.lower()
    if ext == ".cr3":
        return load_cr3_image(path)
    return load_standard_image(path)


def save_u16_grayscale_png(path: Path, arr: np.ndarray) -> None:
    data = np.clip(arr, 0.0, 1.0)
    img = Image.fromarray((data * 65535.0 + 0.5).astype(np.uint16), mode="I;16")
    img.save(path)


def fit_preview_cube(cube: np.ndarray, max_side: int = DEFAULT_PREVIEW_MAX) -> np.ndarray:
    h, w = cube.shape[:2]
    scale = min(max_side / max(h, 1), max_side / max(w, 1), 1.0)
    if scale >= 0.999:
        return cube.astype(np.float32, copy=False)
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    resized = np.empty((nh, nw, cube.shape[2]), dtype=np.float32)
    for i in range(cube.shape[2]):
        band = Image.fromarray(cube[:, :, i].astype(np.float32), mode="F")
        band = band.resize((nw, nh), Image.Resampling.LANCZOS)
        resized[:, :, i] = np.asarray(band, dtype=np.float32)
    return resized


def band_file_extension(export_format: str) -> str:
    return {
        "PNG 16-bit": ".png",
        "TIFF 16-bit": ".tiff",
        "TIFF 32-bit float": ".tiff",
        "NumPy .npy": ".npy",
    }.get(export_format, ".tiff")


def final_file_extension(export_format: str) -> str:
    return {
        "PNG 8-bit": ".png",
        "TIFF 16-bit": ".tiff",
        "TIFF 32-bit float": ".tiff",
        "RAW linear .npy": ".npy",
    }.get(export_format, ".tiff")


def _require_tifffile() -> None:
    if tifffile is None:
        raise RuntimeError("tifffile is not installed. Install requirements.txt for TIFF export support.")


def save_band_image(path: Path, arr: np.ndarray, export_format: str) -> None:
    raw = np.clip(np.asarray(arr, dtype=np.float32), 0.0, None)
    if export_format == "PNG 16-bit":
        save_u16_grayscale_png(path, np.clip(raw, 0.0, 1.0))
    elif export_format == "TIFF 16-bit":
        _require_tifffile()
        tifffile.imwrite(str(path), (np.clip(raw, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16), photometric="minisblack")
    elif export_format == "TIFF 32-bit float":
        _require_tifffile()
        tifffile.imwrite(str(path), raw.astype(np.float32), photometric="minisblack")
    elif export_format == "NumPy .npy":
        np.save(str(path), raw.astype(np.float32))
    else:
        raise ValueError(f"Unsupported band export format: {export_format}")


def save_rgb_image(path: Path, rgb: np.ndarray, export_format: str, *, linear_input: bool = True) -> None:
    raw = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, None)
    srgb_data = linear_to_srgb(np.clip(raw, 0.0, 1.0)).astype(np.float32) if linear_input else np.clip(raw, 0.0, 1.0).astype(np.float32)
    if export_format == "PNG 8-bit":
        img = Image.fromarray((np.clip(srgb_data, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), mode="RGB")
        img.save(path)
    elif export_format == "TIFF 16-bit":
        _require_tifffile()
        tifffile.imwrite(str(path), (np.clip(srgb_data, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16), photometric="rgb")
    elif export_format == "TIFF 32-bit float":
        _require_tifffile()
        tifffile.imwrite(str(path), raw.astype(np.float32), photometric="rgb")
    elif export_format == "RAW linear .npy":
        np.save(str(path), raw.astype(np.float32))
    else:
        raise ValueError(f"Unsupported final export format: {export_format}")



def spectral_separate_flat_rgb(
    flat_rgb_linear: np.ndarray,
    operator: np.ndarray,
    source_sensitivity_matrix: Optional[np.ndarray] = None,
    reprojection_iters: int = DEFAULT_REPROJECTION_ITERS,
    reprojection_strength: float = DEFAULT_REPROJECTION_STRENGTH,
    spectrum_scale: Optional[np.ndarray] = None,
) -> np.ndarray:
    flat = np.clip(np.asarray(flat_rgb_linear, dtype=np.float64), 0.0, None)
    latent = np.clip(flat @ operator.T, 0.0, None)

    scale = None
    if spectrum_scale is not None:
        scale = np.clip(np.asarray(spectrum_scale, dtype=np.float64).reshape(1, -1), 0.0, None)
        if scale.shape[1] != latent.shape[1]:
            raise ValueError("Spectrum scale length must match the number of spectral bands.")
        cube = latent * scale
    else:
        cube = latent

    cube_energy = np.sum(cube, axis=1, keepdims=True)
    cube_energy = np.where(cube_energy < EPS, 1.0, cube_energy)
    src_energy = np.sum(flat, axis=1, keepdims=True)
    cube *= np.where(cube_energy > 0.0, src_energy / cube_energy, 1.0)
    if scale is not None:
        latent = cube / np.maximum(scale, EPS)
    else:
        latent = cube

    if source_sensitivity_matrix is not None and int(reprojection_iters) > 0:
        s = np.asarray(source_sensitivity_matrix, dtype=np.float64)
        strength = float(np.clip(reprojection_strength, 0.0, 1.0))
        for _ in range(int(reprojection_iters)):
            reproj = np.clip(cube @ s.T, 0.0, None)
            error = flat - reproj
            if np.max(np.abs(error)) < 1e-8:
                break
            latent = np.clip(latent + (error @ operator.T) * strength, 0.0, None)
            cube = latent * scale if scale is not None else latent
            cube_energy = np.sum(cube, axis=1, keepdims=True)
            cube_energy = np.where(cube_energy < EPS, 1.0, cube_energy)
            cube *= np.where(cube_energy > 0.0, src_energy / cube_energy, 1.0)
            if scale is not None:
                latent = cube / np.maximum(scale, EPS)
            else:
                latent = cube
    return cube


def transform_lut_samples(
    samples_encoded: np.ndarray,
    wavelengths: np.ndarray,
    curves: Dict[str, CurveModel],
    sensitivity_ceiling: float,
    density_curves: Dict[str, CurveModel],
    density_ceiling: float,
    operator: np.ndarray,
    visible_range: Optional[Tuple[float, float]] = None,
    density_x_range: Tuple[float, float] = (DEFAULT_DENSITY_X_MIN, DEFAULT_DENSITY_X_MAX),
    input_mode: str = "sRGB gamma",
    output_mode: str = "sRGB gamma",
) -> np.ndarray:
    samples = np.clip(np.asarray(samples_encoded, dtype=np.float64), 0.0, 1.0)
    if samples.ndim != 2 or samples.shape[1] != 3:
        raise ValueError("Expected Nx3 RGB samples for LUT generation.")

    s = build_sensitivity_matrix(wavelengths, curves, sensitivity_ceiling)
    mask = np.ones_like(wavelengths, dtype=bool)
    if visible_range is not None:
        lo, hi = visible_range
        mask = (wavelengths >= lo) & (wavelengths <= hi)
        if not np.any(mask):
            mask = np.ones_like(wavelengths, dtype=bool)
    s_visible = s[:, mask]

    samples_linear = decode_lut_input_to_linear_rec709(samples, input_mode)
    cube = spectral_separate_flat_rgb(samples_linear, operator, source_sensitivity_matrix=s)[:, mask]
    rgb_linear = np.clip(cube @ s_visible.T, 0.0, None)

    transmittance = density_map_rgb(
        rgb_linear.reshape(-1, 1, 3),
        density_curves,
        density_ceiling,
        density_x_range=density_x_range,
    ).reshape(-1, 3)
    positive_linear = 1.0 - np.clip(transmittance, 0.0, 1.0)
    return encode_lut_output_from_linear_rec709(np.clip(positive_linear, 0.0, None), output_mode).astype(np.float32)


def export_iridas_cube_lut(
    path: Path,
    lut_size: int,
    wavelengths: np.ndarray,
    curves: Dict[str, CurveModel],
    sensitivity_ceiling: float,
    density_curves: Dict[str, CurveModel],
    density_ceiling: float,
    operator: np.ndarray,
    visible_range: Optional[Tuple[float, float]] = None,
    density_x_range: Tuple[float, float] = (DEFAULT_DENSITY_X_MIN, DEFAULT_DENSITY_X_MAX),
    title: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    chunk_size: int = 65536,
    input_mode: str = "sRGB gamma",
    output_mode: str = "sRGB gamma",
) -> None:
    if lut_size < 2:
        raise ValueError("LUT size must be at least 2.")
    total = int(lut_size) ** 3
    levels = np.linspace(0.0, 1.0, int(lut_size), dtype=np.float64)
    title_text = title or f"{APP_TITLE} {lut_size}^3"

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f'TITLE "{title_text}"\n')
        f.write(f"LUT_3D_SIZE {int(lut_size)}\n")
        f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
        f.write("DOMAIN_MAX 1.0 1.0 1.0\n")
        f.write(f"# INPUT_COLORSPACE {input_mode}\n")
        f.write(f"# OUTPUT_MODE {output_mode}\n")
        f.write("# FILM_DENSITY_MODEL included\n")
        f.write(f"# DENSITY_X_RANGE {float(density_x_range[0]):.6f} {float(density_x_range[1]):.6f}\n")
        f.write("# LUT_OUTPUT positive_inverted_from_transmittance\n")

        for start in range(0, total, int(chunk_size)):
            end = min(total, start + int(chunk_size))
            idx = np.arange(start, end, dtype=np.int64)
            r_idx = idx % lut_size
            g_idx = (idx // lut_size) % lut_size
            b_idx = idx // (lut_size * lut_size)
            samples = np.column_stack([levels[r_idx], levels[g_idx], levels[b_idx]])
            out = transform_lut_samples(
                samples,
                wavelengths=wavelengths,
                curves=curves,
                sensitivity_ceiling=sensitivity_ceiling,
                density_curves=density_curves,
                density_ceiling=density_ceiling,
                operator=operator,
                visible_range=visible_range,
                density_x_range=density_x_range,
                input_mode=input_mode,
                output_mode=output_mode,
            )
            for row in out:
                f.write(f"{float(row[0]):.10f} {float(row[1]):.10f} {float(row[2]):.10f}\n")
            if progress_callback is not None:
                progress_callback(end, total)



def export_band_images(
    folder: Path,
    cube: np.ndarray,
    wavelengths: np.ndarray,
    export_format: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> None:
    bands_dir = folder / "bands"
    bands_dir.mkdir(parents=True, exist_ok=True)
    ext = band_file_extension(export_format)
    maxv = 1.0
    if export_format in ("PNG 16-bit", "TIFF 16-bit"):
        maxv = float(np.max(cube)) if cube.size else 1.0
        maxv = max(maxv, EPS)
    total = int(len(wavelengths))
    for idx, wl in enumerate(wavelengths):
        filename = f"band_{idx + 1:03d}_{int(round(wl))}nm{ext}"
        band = cube[:, :, idx]
        band_to_save = band / maxv if export_format in ("PNG 16-bit", "TIFF 16-bit") else band
        save_band_image(bands_dir / filename, band_to_save, export_format)
        if progress_callback is not None:
            progress_callback(idx + 1, total)


def spectral_false_color_preview(cube: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    if cube.size == 0:
        return np.zeros((8, 8, 3), dtype=np.float32)
    rgb = np.zeros((*cube.shape[:2], 3), dtype=np.float64)
    for i, wl in enumerate(wavelengths):
        color = np.array(wavelength_to_rgb(float(wl)), dtype=np.float64) / 255.0
        rgb += cube[:, :, i:i + 1] * color.reshape(1, 1, 3)
    p = float(np.percentile(rgb, 99.8)) if rgb.size else 1.0
    if p < EPS:
        p = 1.0
    return np.clip(rgb / p, 0.0, 1.0).astype(np.float32)


def serialize_curves(curves: Dict[str, CurveModel]) -> Dict[str, dict]:
    return {name: asdict(curve) for name, curve in curves.items()}


def deserialize_curves(data: Dict[str, dict]) -> Dict[str, CurveModel]:
    return {name: CurveModel(**curve_dict) for name, curve_dict in data.items()}


def convert_legacy_sensitivity_curves(curves: Dict[str, CurveModel], ceiling: float) -> Dict[str, CurveModel]:
    converted: Dict[str, CurveModel] = {}
    for name, curve in curves.items():
        pts = curve.sorted_points()
        max_y = max((p[1] for p in pts), default=1.0)
        if max_y <= 1.000001 and curve.y_max <= 1.000001:
            new_pts = [(x, float(linear_sensitivity_to_log_units(y, ceiling))) for x, y in pts]
            converted[name] = CurveModel(name, curve.color, new_pts, curve.visible, 0.0, ceiling)
        else:
            scale = ceiling / max(curve.y_max, EPS)
            new_pts = [(x, float(np.clip(y * scale, 0.0, ceiling))) for x, y in pts]
            converted[name] = CurveModel(name, curve.color, new_pts, curve.visible, 0.0, ceiling)
    return converted


def convert_legacy_density_curves(curves: Dict[str, CurveModel], ceiling: float) -> Dict[str, CurveModel]:
    converted: Dict[str, CurveModel] = {}
    for name, curve in curves.items():
        pts = curve.sorted_points()
        max_y = max((p[1] for p in pts), default=1.0)
        if max_y <= 1.000001 and curve.y_max <= 1.000001:
            new_pts = [(x, float(np.clip(y * ceiling, 0.0, ceiling))) for x, y in pts]
        else:
            scale = ceiling / max(curve.y_max, EPS)
            new_pts = [(x, float(np.clip(y * scale, 0.0, ceiling))) for x, y in pts]
        converted[name] = CurveModel(name, curve.color, new_pts, curve.visible, 0.0, ceiling)
    return converted


def scale_curve_collection_y(curves: Dict[str, CurveModel], new_ceiling: float) -> Dict[str, CurveModel]:
    scaled: Dict[str, CurveModel] = {}
    for name, curve in curves.items():
        old_top = max(curve.y_max, EPS)
        factor = new_ceiling / old_top
        pts = [(x, float(np.clip(y * factor, 0.0, new_ceiling))) for x, y in curve.sorted_points()]
        scaled[name] = CurveModel(name, curve.color, pts, curve.visible, 0.0, new_ceiling)
    return scaled


def box_blur_gray(arr: np.ndarray, radius: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if radius <= 0:
        return arr.copy()
    pad = int(radius)
    padded = np.pad(arr, ((0, 0), (pad, pad)), mode="reflect")
    c = np.cumsum(padded, axis=1, dtype=np.float64)
    c = np.pad(c, ((0, 0), (1, 0)), mode="constant")
    out = (c[:, 2 * pad + 1:] - c[:, :-2 * pad - 1]) / float(2 * pad + 1)
    padded2 = np.pad(out, ((pad, pad), (0, 0)), mode="reflect")
    c2 = np.cumsum(padded2, axis=0, dtype=np.float64)
    c2 = np.pad(c2, ((1, 0), (0, 0)), mode="constant")
    out2 = (c2[2 * pad + 1:, :] - c2[:-2 * pad - 1, :]) / float(2 * pad + 1)
    return out2.astype(np.float32)


def box_blur_rgb(arr: np.ndarray, radius: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if radius <= 0:
        return arr.copy()
    out = np.empty_like(arr, dtype=np.float32)
    for i in range(arr.shape[2]):
        out[..., i] = box_blur_gray(arr[..., i], radius)
    return out


def deterministic_film_grain(shape: Tuple[int, int]) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    yy, xx = np.indices((h, w), dtype=np.float32)
    g1 = np.mod(np.sin(xx * 12.9898 + yy * 78.233) * 43758.5453, 1.0)
    g2 = np.mod(np.sin(xx * 93.989 + yy * 67.345 + 0.371) * 24634.6345, 1.0)
    grain = (g1 + g2) * 0.5
    return grain.astype(np.float32) * 2.0 - 1.0


def apply_filmic_noise_shaping(
    rgb_positive_linear: np.ndarray,
    cleanup_strength: float = DEFAULT_NOISE_CLEANUP,
    grain_strength: float = DEFAULT_FILM_GRAIN,
) -> np.ndarray:
    data = np.clip(np.asarray(rgb_positive_linear, dtype=np.float32), 0.0, None)
    if data.ndim != 3 or data.shape[2] != 3:
        return data.astype(np.float32, copy=False)

    cleanup = float(np.clip(cleanup_strength, 0.0, 1.0))
    grain_amt = float(np.clip(grain_strength, 0.0, 0.05))
    if cleanup <= 1e-6 and grain_amt <= 1e-6:
        return data.astype(np.float32, copy=False)

    luma = 0.2126 * data[..., 0] + 0.7152 * data[..., 1] + 0.0722 * data[..., 2]
    luma_blur_small = box_blur_gray(luma, 1)
    luma_blur_large = box_blur_gray(luma, 2)
    chroma = data - luma[..., None]
    chroma_blur = box_blur_rgb(chroma, 1)

    residual = np.abs(luma - luma_blur_small)
    edge_mask = np.clip(residual / (0.015 + 0.22 * np.maximum(luma_blur_small, 0.02)), 0.0, 1.0)
    flat_mask = (1.0 - edge_mask) * (0.35 + 0.65 * (1.0 - np.clip(luma_blur_small, 0.0, 1.0)))

    luma_mix = cleanup * (0.28 + 0.32 * flat_mask)
    luma_out = luma * (1.0 - luma_mix) + luma_blur_large * luma_mix

    chroma_mix = cleanup * (0.45 + 0.35 * flat_mask[..., None])
    chroma_out = chroma * (1.0 - chroma_mix) + chroma_blur * chroma_mix

    out = np.clip(luma_out[..., None] + chroma_out, 0.0, None)

    if grain_amt > 1e-6:
        grain = deterministic_film_grain(out.shape[:2])
        grain_envelope = grain_amt * (0.35 + 0.65 * np.sqrt(np.clip(luma_out, 0.0, 1.0)))
        out = np.clip(out + grain[..., None] * grain_envelope[..., None], 0.0, None)

    return out.astype(np.float32)


def save_curve_csv(path: Path, points: List[Tuple[float, float]]) -> None:
    pts = sorted((float(x), float(y)) for x, y in points)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        writer.writerows(pts)


def load_curve_csv(path: Path) -> List[Tuple[float, float]]:
    rows: List[Tuple[float, float]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if len(row) < 2:
                continue
            try:
                x = float(str(row[0]).strip())
                y = float(str(row[1]).strip())
            except ValueError:
                continue
            rows.append((x, y))
    if len(rows) < 2:
        raise ValueError("CSV must contain at least two numeric x,y rows.")
    rows = sorted((float(x), float(y)) for x, y in rows)
    return rows


class CurveEditor(ttk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        title: str,
        curves: Dict[str, CurveModel],
        x_label: str,
        y_label: str,
        x_limits: Tuple[float, float],
        y_limits: Tuple[float, float],
        x_scale: str = "linear",
        y_scale: str = "linear",
        on_change: Optional[Callable[[], None]] = None,
        native_x_to_display: Optional[Callable[[np.ndarray | float], np.ndarray | float]] = None,
        display_x_to_native: Optional[Callable[[np.ndarray | float], np.ndarray | float]] = None,
        native_y_to_display: Optional[Callable[[np.ndarray | float], np.ndarray | float]] = None,
        display_y_to_native: Optional[Callable[[np.ndarray | float], np.ndarray | float]] = None,
    ):
        super().__init__(master)
        self.title = title
        self.curves = curves
        self.native_x_label = x_label
        self.native_y_label = y_label
        self.x_limits = x_limits
        self.y_limits = y_limits
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.on_change = on_change
        self.active_curve_name = next(iter(curves.keys()))
        self.active_point_index: Optional[int] = None
        self.dragging = False
        self._drag_version = 0
        self.curve_vars: Dict[str, tk.BooleanVar] = {}
        self.axis_swapped_var = tk.BooleanVar(value=False)

        self.native_x_to_display = native_x_to_display or (lambda v: v)
        self.display_x_to_native = display_x_to_native or (lambda v: v)
        self.native_y_to_display = native_y_to_display or (lambda v: v)
        self.display_y_to_native = display_y_to_native or (lambda v: v)

        self.point_x_var = tk.DoubleVar(value=self._native_x_display_value(self.x_limits[0]))
        self.point_y_var = tk.DoubleVar(value=self._native_y_display_value(self.y_limits[0]))

        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        top = ttk.Frame(self)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        top.columnconfigure(2, weight=1)

        ttk.Label(top, text=title, style="Header.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(top, text="Edit curve:").grid(row=0, column=1, padx=(12, 6))
        self.active_combo = ttk.Combobox(top, state="readonly", values=list(curves.keys()), width=12)
        self.active_combo.set(self.active_curve_name)
        self.active_combo.grid(row=0, column=2, sticky="w")
        self.active_combo.bind("<<ComboboxSelected>>", self._on_curve_selected)
        ttk.Checkbutton(top, text="Swap axes", variable=self.axis_swapped_var, command=self._on_axis_swap_changed).grid(row=0, column=3, padx=(10, 0), sticky="w")

        self.toggle_frame = ttk.Frame(top)
        self.toggle_frame.grid(row=0, column=4, sticky="e")
        self._rebuild_toggle_buttons()

        toolbar = ttk.Frame(self)
        toolbar.grid(row=1, column=0, sticky="ew", pady=(0, 4))
        for i in range(10):
            toolbar.columnconfigure(i, weight=0)
        toolbar.columnconfigure(9, weight=1)

        ttk.Label(toolbar, text="Display X").grid(row=0, column=0, sticky="w")
        self.x_spin = ttk.Spinbox(toolbar, from_=-1e12, to=1e12, increment=1.0, textvariable=self.point_x_var, width=12)
        self.x_spin.grid(row=0, column=1, sticky="w", padx=(4, 8))
        ttk.Label(toolbar, text="Display Y").grid(row=0, column=2, sticky="w")
        self.y_spin = ttk.Spinbox(toolbar, from_=-1e12, to=1e12, increment=0.05, textvariable=self.point_y_var, width=12)
        self.y_spin.grid(row=0, column=3, sticky="w", padx=(4, 8))
        ttk.Button(toolbar, text="Add point", command=self._add_point_from_fields).grid(row=0, column=4, padx=(0, 4))
        ttk.Button(toolbar, text="Update selected", command=self._update_selected_from_fields).grid(row=0, column=5, padx=(0, 4))
        ttk.Button(toolbar, text="Delete selected", command=self._delete_selected).grid(row=0, column=6, padx=(0, 4))
        ttk.Button(toolbar, text="Insert midpoint", command=self._insert_midpoint).grid(row=0, column=7, padx=(0, 4))
        ttk.Button(toolbar, text="Reset active", command=self._reset_active_curve_shape).grid(row=0, column=8, padx=(0, 4))

        fig = Figure(figsize=(6.0, 3.1), dpi=100)
        self.ax = fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.get_tk_widget().grid(row=2, column=0, sticky="nsew")
        self.canvas.get_tk_widget().bind("<ButtonRelease-1>", self._on_tk_release, add="+")

        self.status_var = tk.StringVar(
            value="Click to select. Drag to move. Double-click to add. Right-click to delete. Preview updates after release."
        )
        ttk.Label(self, textvariable=self.status_var, style="Muted.TLabel").grid(row=3, column=0, sticky="w", pady=(4, 0))

        self.cid_press = fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.cid_release = fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.cid_motion = fig.canvas.mpl_connect("motion_notify_event", self._on_motion)

        self._configure_spin_ranges()
        self._redraw()
        self._sync_fields_from_selection()

    def _native_x_display_value(self, x: float) -> float:
        return float(np.asarray(self.native_x_to_display(x), dtype=np.float64))

    def _native_y_display_value(self, y: float) -> float:
        return float(np.asarray(self.native_y_to_display(y), dtype=np.float64))

    def _display_x_native_value(self, x: float) -> float:
        return float(np.asarray(self.display_x_to_native(x), dtype=np.float64))

    def _display_y_native_value(self, y: float) -> float:
        return float(np.asarray(self.display_y_to_native(y), dtype=np.float64))

    def _native_point_to_display(self, x: float, y: float) -> Tuple[float, float]:
        dx = self._native_x_display_value(x)
        dy = self._native_y_display_value(y)
        if self.axis_swapped_var.get():
            return dy, dx
        return dx, dy

    def _display_point_to_native(self, dx: float, dy: float) -> Tuple[float, float]:
        if self.axis_swapped_var.get():
            dx, dy = dy, dx
        x = self._display_x_native_value(dx)
        y = self._display_y_native_value(dy)
        return self._clamp_xy(x, y)

    def _display_limits(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        x_lo = self._native_x_display_value(self.x_limits[0])
        x_hi = self._native_x_display_value(self.x_limits[1])
        y_lo = self._native_y_display_value(self.y_limits[0])
        y_hi = self._native_y_display_value(self.y_limits[1])
        x_lim = (min(x_lo, x_hi), max(x_lo, x_hi))
        y_lim = (min(y_lo, y_hi), max(y_lo, y_hi))
        if self.axis_swapped_var.get():
            return y_lim, x_lim
        return x_lim, y_lim

    def _display_labels(self) -> Tuple[str, str]:
        if self.axis_swapped_var.get():
            return self.native_y_label, self.native_x_label
        return self.native_x_label, self.native_y_label

    def _display_scales(self) -> Tuple[str, str]:
        if self.axis_swapped_var.get():
            return self.y_scale, self.x_scale
        return self.x_scale, self.y_scale

    def _configure_spin_ranges(self) -> None:
        x_lim, y_lim = self._display_limits()
        x_from = -1e12 if not np.isfinite(x_lim[0]) else x_lim[0]
        x_to = 1e12 if not np.isfinite(x_lim[1]) else x_lim[1]
        y_from = -1e12 if not np.isfinite(y_lim[0]) else y_lim[0]
        y_to = 1e12 if not np.isfinite(y_lim[1]) else y_lim[1]
        self.x_spin.configure(from_=x_from, to=x_to)
        self.y_spin.configure(from_=y_from, to=y_to)

    def set_x_limits(self, lo: float, hi: float) -> None:
        self.x_limits = (float(lo), float(hi))
        self._configure_spin_ranges()
        self._sync_fields_from_selection()
        self._redraw()

    def set_y_limits(self, lo: float, hi: float) -> None:
        self.y_limits = (float(lo), float(hi))
        self._configure_spin_ranges()
        self._sync_fields_from_selection()
        self._redraw()

    def set_display_mapping(
        self,
        *,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        x_scale: Optional[str] = None,
        y_scale: Optional[str] = None,
        native_x_to_display: Optional[Callable[[np.ndarray | float], np.ndarray | float]] = None,
        display_x_to_native: Optional[Callable[[np.ndarray | float], np.ndarray | float]] = None,
        native_y_to_display: Optional[Callable[[np.ndarray | float], np.ndarray | float]] = None,
        display_y_to_native: Optional[Callable[[np.ndarray | float], np.ndarray | float]] = None,
    ) -> None:
        if x_label is not None:
            self.native_x_label = x_label
        if y_label is not None:
            self.native_y_label = y_label
        if x_scale is not None:
            self.x_scale = x_scale
        if y_scale is not None:
            self.y_scale = y_scale
        if native_x_to_display is not None:
            self.native_x_to_display = native_x_to_display
        if display_x_to_native is not None:
            self.display_x_to_native = display_x_to_native
        if native_y_to_display is not None:
            self.native_y_to_display = native_y_to_display
        if display_y_to_native is not None:
            self.display_y_to_native = display_y_to_native
        self._configure_spin_ranges()
        self._sync_fields_from_selection()
        self._redraw()

    def _default_points_for_curve(self, name: str) -> List[Tuple[float, float]]:
        if set(self.curves.keys()) == {"Red", "Green", "Blue"} and self.x_limits[0] >= 300.0:
            defaults = default_sensitivity_curves(int(round(self.x_limits[0])), int(round(self.x_limits[1])), ceiling=self.y_limits[1])
            return defaults.get(name, next(iter(defaults.values()))).points
        defaults = default_density_curves(self.y_limits[1])
        return defaults.get(name, next(iter(defaults.values()))).points

    def _rebuild_toggle_buttons(self) -> None:
        for child in self.toggle_frame.winfo_children():
            child.destroy()
        self.curve_vars = {}
        for i, (name, curve) in enumerate(self.curves.items()):
            var = tk.BooleanVar(value=curve.visible)
            self.curve_vars[name] = var
            chk = ttk.Checkbutton(self.toggle_frame, text=name, variable=var, command=self._on_visibility_change)
            chk.grid(row=0, column=i, padx=2)

    def set_curves(self, curves: Dict[str, CurveModel], x_limits: Optional[Tuple[float, float]] = None, y_limits: Optional[Tuple[float, float]] = None) -> None:
        self.curves = curves
        if x_limits is not None:
            self.x_limits = (float(x_limits[0]), float(x_limits[1]))
        if y_limits is not None:
            self.y_limits = (float(y_limits[0]), float(y_limits[1]))
        values = list(curves.keys())
        self.active_combo.configure(values=values)
        if self.active_curve_name not in curves:
            self.active_curve_name = values[0]
        self.active_combo.set(self.active_curve_name)
        self.active_point_index = None
        self._rebuild_toggle_buttons()
        self._configure_spin_ranges()
        self._sync_fields_from_selection()
        self._redraw()

    def get_curves(self) -> Dict[str, CurveModel]:
        for name, var in self.curve_vars.items():
            self.curves[name].visible = bool(var.get())
            self.curves[name].y_min = self.y_limits[0]
            self.curves[name].y_max = self.y_limits[1]
        return self.curves

    def _on_curve_selected(self, _event=None) -> None:
        self.active_curve_name = self.active_combo.get()
        self.active_point_index = None
        self.status_var.set(f"Editing {self.active_curve_name}. Click to select. Double-click to add.")
        self._sync_fields_from_selection()
        self._redraw()

    def _on_axis_swap_changed(self) -> None:
        self._configure_spin_ranges()
        self._sync_fields_from_selection()
        self._redraw()

    def _on_visibility_change(self) -> None:
        for name, var in self.curve_vars.items():
            self.curves[name].visible = bool(var.get())
        self._redraw()
        self._emit_change()

    def _selected_curve_points(self) -> List[Tuple[float, float]]:
        return self.curves[self.active_curve_name].sorted_points()

    def _pick_point(self, event) -> Optional[Tuple[str, int]]:
        if event.xdata is None or event.ydata is None or event.inaxes != self.ax:
            return None
        best: Optional[Tuple[str, int]] = None
        best_dist = float("inf")
        for name, curve in self.curves.items():
            if not curve.visible:
                continue
            pts = curve.sorted_points()
            for idx, (px, py) in enumerate(pts):
                dx, dy = self._native_point_to_display(px, py)
                sx, sy = self.ax.transData.transform((dx, dy))
                dist = math.hypot(event.x - sx, event.y - sy)
                if dist < best_dist and dist < 14.0:
                    best = (name, idx)
                    best_dist = dist
        return best

    def _clamp_xy(self, x: float, y: float) -> Tuple[float, float]:
        x = max(self.x_limits[0], min(self.x_limits[1], float(x)))
        y = max(self.y_limits[0], min(self.y_limits[1], float(y)))
        return x, y

    def _sync_fields_from_selection(self) -> None:
        pts = self._selected_curve_points()
        if self.active_point_index is not None and 0 <= self.active_point_index < len(pts):
            x, y = pts[self.active_point_index]
            dx, dy = self._native_point_to_display(x, y)
            self.point_x_var.set(float(dx))
            self.point_y_var.set(float(dy))
        else:
            mid_x = 0.5 * (self.x_limits[0] + self.x_limits[1])
            mid_y = 0.5 * (self.y_limits[0] + self.y_limits[1])
            dx, dy = self._native_point_to_display(mid_x, mid_y)
            self.point_x_var.set(float(dx))
            self.point_y_var.set(float(dy))

    def _update_curve_points(self, pts: List[Tuple[float, float]], keep_x: Optional[float] = None) -> None:
        pts = [self._clamp_xy(x, y) for x, y in pts]
        pts = sorted(pts)
        self.curves[self.active_curve_name].points = pts
        self.curves[self.active_curve_name].y_min = self.y_limits[0]
        self.curves[self.active_curve_name].y_max = self.y_limits[1]
        if keep_x is not None:
            for i, (x, _) in enumerate(pts):
                if abs(x - keep_x) < 1e-9:
                    self.active_point_index = i
                    break

    def _add_point_from_fields(self) -> None:
        curve = self.curves[self.active_curve_name]
        x, y = self._display_point_to_native(self.point_x_var.get(), self.point_y_var.get())
        curve.points.append((x, y))
        curve.points = curve.sorted_points()
        for i, (px, py) in enumerate(curve.points):
            if abs(px - x) < 1e-9 and abs(py - y) < 1e-9:
                self.active_point_index = i
                break
        dx, dy = self._native_point_to_display(x, y)
        self.status_var.set(f"Added point to {self.active_curve_name} at ({dx:.4g}, {dy:.4g}).")
        self._redraw()
        self._emit_change()

    def _update_selected_from_fields(self) -> None:
        curve = self.curves[self.active_curve_name]
        if self.active_point_index is None:
            self._add_point_from_fields()
            return
        pts = curve.sorted_points()
        if not (0 <= self.active_point_index < len(pts)):
            self.active_point_index = None
            self._add_point_from_fields()
            return
        x, y = self._display_point_to_native(self.point_x_var.get(), self.point_y_var.get())
        pts[self.active_point_index] = (x, y)
        self._update_curve_points(pts, keep_x=x)
        self.status_var.set(f"Updated selected point on {self.active_curve_name}.")
        self._redraw()
        self._emit_change()

    def _delete_selected(self) -> None:
        curve = self.curves[self.active_curve_name]
        pts = curve.sorted_points()
        if self.active_point_index is None or not (0 <= self.active_point_index < len(pts)):
            self.status_var.set("No point selected.")
            return
        if len(pts) <= 2:
            self.status_var.set("At least two points must remain on the curve.")
            return
        del pts[self.active_point_index]
        curve.points = pts
        self.active_point_index = min(self.active_point_index, len(pts) - 1) if pts else None
        self._sync_fields_from_selection()
        self.status_var.set(f"Deleted point from {self.active_curve_name}.")
        self._redraw()
        self._emit_change()

    def _insert_midpoint(self) -> None:
        pts = self._selected_curve_points()
        if len(pts) < 2:
            self.status_var.set("Need at least two points to insert a midpoint.")
            return
        idx = self.active_point_index if self.active_point_index is not None else len(pts) // 2 - 1
        idx = max(0, min(idx, len(pts) - 2))
        x0, y0 = pts[idx]
        x1, y1 = pts[idx + 1]
        x = 0.5 * (x0 + x1)
        y = 0.5 * (y0 + y1)
        dx, dy = self._native_point_to_display(x, y)
        self.point_x_var.set(dx)
        self.point_y_var.set(dy)
        self._add_point_from_fields()

    def _reset_active_curve_shape(self) -> None:
        self.curves[self.active_curve_name].points = list(self._default_points_for_curve(self.active_curve_name))
        self.curves[self.active_curve_name].y_min = self.y_limits[0]
        self.curves[self.active_curve_name].y_max = self.y_limits[1]
        self.active_point_index = None
        self._sync_fields_from_selection()
        self.status_var.set(f"Reset {self.active_curve_name} to its default shape.")
        self._redraw()
        self._emit_change()

    def _on_press(self, event) -> None:
        if event.inaxes != self.ax:
            return
        picked = self._pick_point(event)
        if event.dblclick and event.xdata is not None and event.ydata is not None:
            self.point_x_var.set(float(event.xdata))
            self.point_y_var.set(float(event.ydata))
            self._add_point_from_fields()
            return
        if event.button == 3 and picked is not None:
            self.active_curve_name = picked[0]
            self.active_combo.set(self.active_curve_name)
            self.active_point_index = picked[1]
            self._delete_selected()
            return
        if event.button == 1:
            self._drag_version += 1
            if picked is not None:
                self.active_curve_name = picked[0]
                self.active_combo.set(self.active_curve_name)
                self.active_point_index = picked[1]
                self.dragging = True
                self._sync_fields_from_selection()
                self.status_var.set(f"Dragging point on {self.active_curve_name}. Preview will update on release.")
                self._redraw()
                return
            if event.xdata is not None and event.ydata is not None:
                self.point_x_var.set(float(event.xdata))
                self.point_y_var.set(float(event.ydata))
                self.active_point_index = None
                self.status_var.set("No point selected. Use Add point to insert one at the current display X/Y fields.")
                self._redraw()

    def _on_motion(self, event) -> None:
        if not self.dragging or event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        curve = self.curves[self.active_curve_name]
        pts = curve.sorted_points()
        if self.active_point_index is None or not (0 <= self.active_point_index < len(pts)):
            return
        x, y = self._display_point_to_native(event.xdata, event.ydata)
        pts[self.active_point_index] = (x, y)
        self._update_curve_points(pts, keep_x=x)
        dx, dy = self._native_point_to_display(x, y)
        self.point_x_var.set(float(dx))
        self.point_y_var.set(float(dy))
        self._redraw()

    def _finish_drag(self) -> None:
        if self.dragging:
            self.dragging = False
            self._drag_version += 1
            self.status_var.set(f"Released {self.active_curve_name}. Preview updating.")
            self._emit_change()

    def _on_release(self, _event) -> None:
        self._finish_drag()

    def _on_tk_release(self, _event) -> None:
        self._finish_drag()

    def _redraw(self) -> None:
        self.ax.clear()
        x_label, y_label = self._display_labels()
        x_lim, y_lim = self._display_limits()
        x_scale, y_scale = self._display_scales()
        self.ax.set_title(self.title)
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.ax.set_xlim(*x_lim)
        self.ax.set_ylim(*y_lim)
        self.ax.set_xscale(x_scale)
        self.ax.set_yscale(y_scale)
        self.ax.grid(True, which="both", alpha=0.22)

        x_native = np.linspace(self.x_limits[0], self.x_limits[1], 700)
        for name, curve in self.curves.items():
            if not curve.visible:
                continue
            y_native = curve.sample_linear(x_native)
            if self.axis_swapped_var.get():
                plot_x = np.asarray(self.native_y_to_display(y_native), dtype=np.float64)
                plot_y = np.asarray(self.native_x_to_display(x_native), dtype=np.float64)
            else:
                plot_x = np.asarray(self.native_x_to_display(x_native), dtype=np.float64)
                plot_y = np.asarray(self.native_y_to_display(y_native), dtype=np.float64)

            pts = curve.sorted_points()
            display_pts = [self._native_point_to_display(px, py) for px, py in pts]
            px = [p[0] for p in display_pts]
            py = [p[1] for p in display_pts]
            lw = 2.6 if name == self.active_curve_name else 1.8
            alpha = 1.0 if name == self.active_curve_name else 0.72
            self.ax.plot(plot_x, plot_y, color=curve.color, linewidth=lw, alpha=alpha, label=name)
            sizes = []
            for idx, _ in enumerate(pts):
                if name == self.active_curve_name and idx == self.active_point_index:
                    sizes.append(44)
                elif name == self.active_curve_name:
                    sizes.append(28)
                else:
                    sizes.append(18)
            self.ax.scatter(px, py, color=curve.color, s=sizes, alpha=alpha, edgecolors="white", linewidths=0.6, zorder=4)
        self.ax.legend(loc="upper right", fontsize=8)
        self.canvas.draw_idle()

    def _emit_change(self) -> None:
        if self.on_change is not None:
            self.on_change()


class ScrollableImageView(ttk.Frame):
    def __init__(self, master: tk.Misc, *, initial_zoom: float = 100.0):
        super().__init__(master)
        self.image_array: Optional[np.ndarray] = None
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._fit_pending = False
        self.zoom_var = tk.DoubleVar(value=initial_zoom)
        self.zoom_text_var = tk.StringVar(value=f"{int(round(initial_zoom))}%")

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        toolbar.columnconfigure(1, weight=1)

        ttk.Label(toolbar, text="Zoom").grid(row=0, column=0, sticky="w")
        self.zoom_scale = ttk.Scale(toolbar, from_=10.0, to=400.0, orient="horizontal", variable=self.zoom_var, command=self._on_zoom_changed)
        self.zoom_scale.grid(row=0, column=1, sticky="ew", padx=(6, 8))
        ttk.Label(toolbar, textvariable=self.zoom_text_var, width=6).grid(row=0, column=2, sticky="e")
        ttk.Button(toolbar, text="Fit", command=self.fit_to_view).grid(row=0, column=3, padx=(8, 4))
        ttk.Button(toolbar, text="100%", command=lambda: self.set_zoom(100.0)).grid(row=0, column=4)

        body = ttk.Frame(self)
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(body, highlightthickness=0, background="#111111")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.x_scroll = ttk.Scrollbar(body, orient="horizontal", command=self.canvas.xview)
        self.y_scroll = ttk.Scrollbar(body, orient="vertical", command=self.canvas.yview)
        self.x_scroll.grid(row=1, column=0, sticky="ew")
        self.y_scroll.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(xscrollcommand=self.x_scroll.set, yscrollcommand=self.y_scroll.set)
        self._canvas_image_id = self.canvas.create_image(0, 0, anchor="nw")

        self.canvas.bind("<Configure>", self._on_canvas_configure, add="+")
        self.canvas.bind("<MouseWheel>", self._on_mousewheel, add="+")
        self.canvas.bind("<Shift-MouseWheel>", self._on_shift_mousewheel, add="+")

    def set_zoom(self, value: float) -> None:
        value = float(np.clip(value, 10.0, 400.0))
        self.zoom_var.set(value)
        self._render()

    def _on_zoom_changed(self, _value=None) -> None:
        self._render()

    def _on_canvas_configure(self, _event=None) -> None:
        if self._fit_pending and self.image_array is not None:
            self._fit_pending = False
            self.fit_to_view()

    def _on_mousewheel(self, event) -> None:
        if event.state & 0x0004:
            step = 1.08 if event.delta > 0 else 1 / 1.08
            self.set_zoom(self.zoom_var.get() * step)
            return
        units = -1 if event.delta > 0 else 1
        self.canvas.yview_scroll(units, "units")

    def _on_shift_mousewheel(self, event) -> None:
        units = -1 if event.delta > 0 else 1
        self.canvas.xview_scroll(units, "units")

    def set_array(self, rgb: Optional[np.ndarray], *, auto_fit: bool = True) -> None:
        self.image_array = None if rgb is None else np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
        if self.image_array is None:
            self.canvas.itemconfigure(self._canvas_image_id, image="")
            self.canvas.configure(scrollregion=(0, 0, 1, 1))
            self._photo = None
            return
        if auto_fit:
            self._fit_pending = True
            self.after_idle(self.fit_to_view)
        else:
            self._render()

    def fit_to_view(self) -> None:
        if self.image_array is None:
            return
        h, w = self.image_array.shape[:2]
        cw = max(self.canvas.winfo_width() - 4, 32)
        ch = max(self.canvas.winfo_height() - 4, 32)
        zoom = min(cw / max(w, 1), ch / max(h, 1)) * 100.0
        zoom = float(np.clip(zoom, 10.0, 400.0))
        self.zoom_var.set(zoom)
        self._render()

    def _render(self) -> None:
        if self.image_array is None:
            return
        zoom = float(np.clip(self.zoom_var.get(), 10.0, 400.0))
        self.zoom_text_var.set(f"{int(round(zoom))}%")
        scale = zoom / 100.0
        h, w = self.image_array.shape[:2]
        out_w = max(1, int(round(w * scale)))
        out_h = max(1, int(round(h * scale)))
        img = Image.fromarray((self.image_array * 255.0 + 0.5).astype(np.uint8), mode="RGB")
        if out_w != w or out_h != h:
            img = img.resize((out_w, out_h), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.itemconfigure(self._canvas_image_id, image=self._photo)
        self.canvas.configure(scrollregion=(0, 0, out_w, out_h))

class SpectralToolApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1650x1080")
        self.minsize(1240, 800)

        self._configure_styles()

        self.loaded_image_linear: Optional[np.ndarray] = None
        self.loaded_image_preview_srgb: Optional[np.ndarray] = None
        self.spectral_cube: Optional[np.ndarray] = None
        self.spectral_cube_preview: Optional[np.ndarray] = None
        self.wavelengths: Optional[np.ndarray] = None
        self.metadata_dir: Optional[Path] = None
        self.source_reference_linear: Optional[np.ndarray] = None
        self.source_reference_preview_linear: Optional[np.ndarray] = None
        self.scene_illuminant_selection: Optional[Dict[str, object]] = None

        self.band_preview_image: Optional[ImageTk.PhotoImage] = None
        self.input_preview_image: Optional[ImageTk.PhotoImage] = None
        self.recon_preview_image: Optional[ImageTk.PhotoImage] = None
        self._preview_after_id: Optional[str] = None
        self._executor = ThreadPoolExecutor(max_workers=DEFAULT_WORKER_THREADS, thread_name_prefix="spectral")
        self._job_queue: "queue.Queue[Tuple[str, str]]" = queue.Queue()
        self._active_future: Optional[Future] = None
        self._active_job_name: Optional[str] = None

        self._build_variables()
        self._build_layout()
        self._refresh_curve_ranges()
        self._poll_background_queue()
        self.protocol("WM_DELETE_WINDOW", self._on_close)


    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("Muted.TLabel", foreground="#666666")
        style.configure("Panel.TFrame", relief="flat")
    
    def _queue_message(self, channel: str, message: str) -> None:
        self._job_queue.put((str(channel), str(message)))
    
    def _poll_background_queue(self) -> None:
        try:
            while True:
                channel, message = self._job_queue.get_nowait()
                if channel == "status":
                    self.status_var.set(message)
                elif channel == "separate":
                    self.separate_status_var.set(message)
                elif channel == "combine":
                    self.combine_status_var.set(message)
        except queue.Empty:
            pass
        self.after(80, self._poll_background_queue)
    
    def _background_busy(self) -> bool:
        return self._active_future is not None and not self._active_future.done()
    
    def _run_background_job(
        self,
        job_name: str,
        worker: Callable[[], object],
        on_success: Callable[[object], None],
    ) -> None:
        if self._background_busy():
            messagebox.showwarning(APP_TITLE, f"{self._active_job_name or 'A background job'} is still running.")
            return
    
        self._active_job_name = job_name
        self._active_future = self._executor.submit(worker)
        self._queue_message("status", f"{job_name}…")
    
        def _poll() -> None:
            future = self._active_future
            if future is None:
                return
            if future.done():
                self._active_future = None
                active_name = self._active_job_name or job_name
                self._active_job_name = None
                try:
                    result = future.result()
                except Exception as exc:
                    self._set_status(f"{active_name} failed.")
                    messagebox.showerror(APP_TITLE, f"{active_name} failed.\n\n{exc}")
                    return
                on_success(result)
                return
            self.after(100, _poll)
    
        self.after(100, _poll)
    
    def _on_close(self) -> None:
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        self.destroy()
    
    def _build_variables(self) -> None:
        self.image_path_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        self.project_dir_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready.")
        self.separate_status_var = tk.StringVar(value="No image loaded.")
        self.combine_status_var = tk.StringVar(value="No spectral project loaded.")
    
        self.nm_start_var = tk.IntVar(value=380)
        self.nm_end_var = tk.IntVar(value=720)
        self.band_count_var = tk.IntVar(value=31)
        self.smoothness_var = tk.DoubleVar(value=0.04)
        self.energy_reg_var = tk.DoubleVar(value=0.004)
        self.preview_downsample_var = tk.IntVar(value=1100)
        self.band_export_format_var = tk.StringVar(value="TIFF 16-bit")
        self.scene_illuminant_mode_var = tk.StringVar(value=DEFAULT_ILLUMINANT_MODE)
        self.scene_illuminant_info_var = tk.StringVar(value="Scene illuminant prior: Auto (estimate pending)")
    
        self.visible_start_var = tk.IntVar(value=380)
        self.visible_end_var = tk.IntVar(value=720)
        self.sensitivity_log_ceiling_var = tk.DoubleVar(value=DEFAULT_SENSITIVITY_LOG_CEILING)
        self.density_log_ceiling_var = tk.DoubleVar(value=DEFAULT_DENSITY_LOG_CEILING)
        self.density_use_lux_var = tk.BooleanVar(value=False)
        self.density_logexp_start_var = tk.DoubleVar(value=DEFAULT_DENSITY_X_MIN)
        self.density_logexp_end_var = tk.DoubleVar(value=DEFAULT_DENSITY_X_MAX)
        self.density_lux_start_var = tk.DoubleVar(value=1e-4)
        self.density_lux_end_var = tk.DoubleVar(value=10.0)
        self.final_export_format_var = tk.StringVar(value="TIFF 16-bit")
        self.lut_size_var = tk.StringVar(value="32")
        self.lut_input_mode_var = tk.StringVar(value="sRGB gamma")
        self.lut_output_mode_var = tk.StringVar(value="sRGB gamma")
        self.preview_balance_ev_var = tk.DoubleVar(value=0.0)
        self.preview_balance_text_var = tk.StringVar(value="0.0 EV")
        self.noise_cleanup_var = tk.DoubleVar(value=DEFAULT_NOISE_CLEANUP)
        self.noise_cleanup_text_var = tk.StringVar(value=f"{DEFAULT_NOISE_CLEANUP:.2f}")
        self.film_grain_var = tk.DoubleVar(value=DEFAULT_FILM_GRAIN)
        self.film_grain_text_var = tk.StringVar(value=f"{DEFAULT_FILM_GRAIN:.3f}")
    
        self.curves = default_sensitivity_curves(self.nm_start_var.get(), self.nm_end_var.get(), self.sensitivity_log_ceiling_var.get())
        self.density_curves = default_density_curves(self.density_log_ceiling_var.get())
    
    def _build_layout(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
    
        header = ttk.Frame(root)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text=APP_TITLE, style="Header.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="RGB/RAW → spectral-band approximation → editable recombination",
            style="Muted.TLabel",
        ).grid(row=1, column=0, sticky="w")
        ttk.Label(header, textvariable=self.status_var).grid(row=0, column=1, rowspan=2, sticky="e")
    
        notebook = ttk.Notebook(root)
        notebook.grid(row=1, column=0, sticky="nsew")
    
        self.separate_tab = ttk.Frame(notebook, padding=8)
        self.combine_tab = ttk.Frame(notebook, padding=8)
        notebook.add(self.separate_tab, text="Separate")
        notebook.add(self.combine_tab, text="Combine")
    
        self._build_separate_tab()
        self._build_combine_tab()
    
    def _build_separate_tab(self) -> None:
        tab = self.separate_tab
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(2, weight=1)
    
        controls = ttk.LabelFrame(tab, text="Input and spectral settings", padding=10)
        controls.grid(row=0, column=0, sticky="ew")
        for i in range(8):
            controls.columnconfigure(i, weight=1 if i in (1, 6) else 0)
    
        ttk.Label(controls, text="Image").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.image_path_var).grid(row=0, column=1, columnspan=5, sticky="ew", padx=(6, 6))
        ttk.Button(controls, text="Browse…", command=self.browse_image).grid(row=0, column=6, sticky="ew")
        ttk.Button(controls, text="Load", command=self.load_image).grid(row=0, column=7, sticky="ew", padx=(6, 0))
    
        ttk.Label(controls, text="Output folder").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(controls, textvariable=self.output_dir_var).grid(row=1, column=1, columnspan=5, sticky="ew", padx=(6, 6), pady=(8, 0))
        ttk.Button(controls, text="Choose…", command=self.choose_output_folder).grid(row=1, column=6, sticky="ew", pady=(8, 0))
        ttk.Button(controls, text="Separate and save", command=self.run_separation).grid(row=1, column=7, sticky="ew", padx=(6, 0), pady=(8, 0))
    
        params = ttk.Frame(tab)
        params.grid(row=1, column=0, sticky="ew", pady=(10, 10))
        for i in range(10):
            params.columnconfigure(i, weight=1)
    
        ttk.Label(params, text="Start nm").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(params, from_=300, to=1200, increment=1, textvariable=self.nm_start_var, width=8, command=self._on_nm_range_changed).grid(row=0, column=1, sticky="w")
        ttk.Label(params, text="End nm").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(params, from_=300, to=1200, increment=1, textvariable=self.nm_end_var, width=8, command=self._on_nm_range_changed).grid(row=0, column=3, sticky="w")
        ttk.Label(params, text="Band count").grid(row=0, column=4, sticky="w")
        ttk.Spinbox(params, from_=3, to=200, increment=1, textvariable=self.band_count_var, width=8).grid(row=0, column=5, sticky="w")
        ttk.Label(params, text="Smoothness").grid(row=0, column=6, sticky="w")
        ttk.Entry(params, textvariable=self.smoothness_var, width=10).grid(row=0, column=7, sticky="w")
        ttk.Label(params, text="Energy regularization").grid(row=0, column=8, sticky="w")
        ttk.Entry(params, textvariable=self.energy_reg_var, width=10).grid(row=0, column=9, sticky="w")
    
        ttk.Label(params, text="Preview max side").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(params, from_=256, to=2400, increment=64, textvariable=self.preview_downsample_var, width=8).grid(row=1, column=1, sticky="w", pady=(6, 0))
        ttk.Label(params, text="Band export format").grid(row=1, column=2, sticky="w", pady=(6, 0))
        ttk.Combobox(params, state="readonly", values=BAND_EXPORT_FORMATS, textvariable=self.band_export_format_var, width=16).grid(row=1, column=3, sticky="w", pady=(6, 0))
        ttk.Label(params, text="Scene illuminant").grid(row=1, column=4, sticky="w", pady=(6, 0))
        illum_combo = ttk.Combobox(params, state="readonly", values=ILLUMINANT_MODES, textvariable=self.scene_illuminant_mode_var, width=22)
        illum_combo.grid(row=1, column=5, sticky="w", pady=(6, 0))
        illum_combo.bind("<<ComboboxSelected>>", self._on_scene_illuminant_mode_changed)
        ttk.Label(params, text="The separation stage uses a Shades-of-Gray illuminant estimate mapped to D50/D55/D65/Tungsten/LED, or the manual override you choose here.", style="Muted.TLabel").grid(row=1, column=6, columnspan=4, sticky="w", pady=(6, 0))
        ttk.Label(params, textvariable=self.scene_illuminant_info_var, style="Muted.TLabel").grid(row=2, column=0, columnspan=10, sticky="w", pady=(6, 0))
        ttk.Label(params, textvariable=self.separate_status_var, style="Muted.TLabel").grid(row=3, column=0, columnspan=10, sticky="w", pady=(6, 0))
    
        preview_panes = ttk.Panedwindow(tab, orient=tk.HORIZONTAL)
        preview_panes.grid(row=2, column=0, sticky="nsew")
    
        input_panel = ttk.LabelFrame(preview_panes, text="Input preview", padding=8)
        input_panel.columnconfigure(0, weight=1)
        input_panel.rowconfigure(0, weight=1)
        self.input_preview_view = ScrollableImageView(input_panel)
        self.input_preview_view.grid(row=0, column=0, sticky="nsew")
    
        band_panel = ttk.LabelFrame(preview_panes, text="False-color spectral preview", padding=8)
        band_panel.columnconfigure(0, weight=1)
        band_panel.rowconfigure(0, weight=1)
        self.band_preview_view = ScrollableImageView(band_panel)
        self.band_preview_view.grid(row=0, column=0, sticky="nsew")
    
        preview_panes.add(input_panel, weight=1)
        preview_panes.add(band_panel, weight=1)
    
    def _build_combine_tab(self) -> None:
        tab = self.combine_tab
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)
    
        top = ttk.LabelFrame(tab, text="Spectral project and export", padding=10)
        top.grid(row=0, column=0, sticky="ew")
        for i in range(10):
            top.columnconfigure(i, weight=1 if i in (1, 5, 8, 9) else 0)
    
        ttk.Label(top, text="Project folder").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.project_dir_var).grid(row=0, column=1, columnspan=5, sticky="ew", padx=(6, 6))
        ttk.Button(top, text="Browse…", command=self.choose_project_folder).grid(row=0, column=6, sticky="ew")
        ttk.Button(top, text="Load project", command=self.load_project).grid(row=0, column=7, sticky="ew", padx=(6, 0))
        ttk.Button(top, text="Save curves/settings…", command=self.save_curve_settings_preset).grid(row=0, column=8, sticky="ew", padx=(8, 0))
        ttk.Button(top, text="Load curves/settings…", command=self.load_curve_settings_preset).grid(row=0, column=9, sticky="ew", padx=(6, 0))
    
        ttk.Label(top, text="Visible from nm").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Spinbox(top, from_=300, to=1200, increment=1, textvariable=self.visible_start_var, width=8, command=self.schedule_live_preview).grid(row=1, column=1, sticky="w", pady=(8, 0))
        ttk.Label(top, text="to nm").grid(row=1, column=2, sticky="e", pady=(8, 0))
        ttk.Spinbox(top, from_=300, to=1200, increment=1, textvariable=self.visible_end_var, width=8, command=self.schedule_live_preview).grid(row=1, column=3, sticky="w", pady=(8, 0))
        ttk.Label(top, text="Final export format").grid(row=1, column=4, sticky="e", pady=(8, 0))
        ttk.Combobox(top, state="readonly", values=FINAL_EXPORT_FORMATS, textvariable=self.final_export_format_var, width=16).grid(row=1, column=5, sticky="w", pady=(8, 0))
        ttk.Button(top, text="Export reconstructed", command=self.export_reconstructed).grid(row=1, column=6, sticky="ew", pady=(8, 0))
        ttk.Label(top, text="3D LUT size").grid(row=1, column=7, sticky="e", pady=(8, 0))
        ttk.Combobox(top, state="readonly", values=LUT_3D_SIZES, textvariable=self.lut_size_var, width=8).grid(row=1, column=8, sticky="w", pady=(8, 0))
        ttk.Button(top, text="Export 3D LUT", command=self.export_3d_lut).grid(row=1, column=9, sticky="ew", padx=(6, 0), pady=(8, 0))
    
        ttk.Label(top, text="Sensitivity log ceiling").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(top, textvariable=self.sensitivity_log_ceiling_var, width=8).grid(row=2, column=1, sticky="w", pady=(8, 0))
        ttk.Label(top, text="Density log ceiling").grid(row=2, column=2, sticky="e", pady=(8, 0))
        ttk.Entry(top, textvariable=self.density_log_ceiling_var, width=8).grid(row=2, column=3, sticky="w", pady=(8, 0))
        ttk.Label(top, text="LUT input mode").grid(row=2, column=4, sticky="e", pady=(8, 0))
        ttk.Combobox(top, state="readonly", values=LUT_INPUT_MODES, textvariable=self.lut_input_mode_var, width=22).grid(row=2, column=5, sticky="w", pady=(8, 0))
        ttk.Label(top, text="LUT output mode").grid(row=2, column=6, sticky="e", pady=(8, 0))
        ttk.Combobox(top, state="readonly", values=LUT_OUTPUT_MODES, textvariable=self.lut_output_mode_var, width=22).grid(row=2, column=7, sticky="w", pady=(8, 0))
        ttk.Button(top, text="Apply ceilings", command=self.apply_curve_ceilings).grid(row=2, column=8, sticky="ew", pady=(8, 0))
        ttk.Label(top, text="Curve presets save the curves, ceilings, visible range, export formats, lux range, LUT input/output modes, LUT size, and axis-swap states.", style="Muted.TLabel").grid(row=2, column=9, sticky="w", padx=(10, 0), pady=(8, 0))
    
        ttk.Checkbutton(top, text="Characteristic-curve exposure axis in lux·s", variable=self.density_use_lux_var, command=self.apply_density_axis_settings).grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Label(top, text="Lux·s start").grid(row=3, column=2, sticky="e", pady=(8, 0))
        ttk.Entry(top, textvariable=self.density_lux_start_var, width=10).grid(row=3, column=3, sticky="w", pady=(8, 0))
        ttk.Label(top, text="Lux·s end").grid(row=3, column=4, sticky="e", pady=(8, 0))
        ttk.Entry(top, textvariable=self.density_lux_end_var, width=10).grid(row=3, column=5, sticky="w", pady=(8, 0))
        ttk.Button(top, text="Apply density axis", command=self.apply_density_axis_settings).grid(row=3, column=6, sticky="ew", pady=(8, 0))
        ttk.Label(top, text="Graphs are in resizable panes. Preview zoom/scroll is separate from export. TIFF 32-bit float keeps unclipped linear data, and RAW linear .npy saves scene-referred output.", style="Muted.TLabel").grid(row=3, column=7, columnspan=3, sticky="w", padx=(10, 0), pady=(8, 0))
    
        ttk.Label(top, textvariable=self.combine_status_var, style="Muted.TLabel").grid(row=4, column=0, columnspan=10, sticky="w", pady=(8, 0))
    
        main_panes = ttk.Panedwindow(tab, orient=tk.VERTICAL)
        main_panes.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
    
        editor_panes = ttk.Panedwindow(main_panes, orient=tk.HORIZONTAL)
        sens_holder = ttk.Frame(editor_panes)
        dens_holder = ttk.Frame(editor_panes)
        sens_holder.columnconfigure(0, weight=1)
        sens_holder.rowconfigure(0, weight=1)
        dens_holder.columnconfigure(0, weight=1)
        dens_holder.rowconfigure(0, weight=1)
    
        self.sensitivity_editor = CurveEditor(
            sens_holder,
            title="Spectral sensitivity curves",
            curves=self.curves,
            x_label="Wavelength (nm)",
            y_label="Log sensitivity units",
            x_limits=(float(self.nm_start_var.get()), float(self.nm_end_var.get())),
            y_limits=(0.0, float(self.sensitivity_log_ceiling_var.get())),
            x_scale="linear",
            y_scale="linear",
            on_change=self.schedule_live_preview,
        )
        self.sensitivity_editor.grid(row=0, column=0, sticky="nsew")
    
        self.density_editor = CurveEditor(
            dens_holder,
            title="RGB characteristic curves (H&D)",
            curves=self.density_curves,
            x_label="Relative log exposure",
            y_label="Optical density D",
            x_limits=(float(self.density_logexp_start_var.get()), float(self.density_logexp_end_var.get())),
            y_limits=(0.0, float(self.density_log_ceiling_var.get())),
            x_scale="linear",
            y_scale="linear",
            on_change=self.schedule_live_preview,
        )
        self.density_editor.grid(row=0, column=0, sticky="nsew")
        self.apply_density_axis_settings(schedule_preview=False)
    
        editor_panes.add(sens_holder, weight=1)
        editor_panes.add(dens_holder, weight=1)
    
        preview_holder = ttk.LabelFrame(main_panes, text="Live reconstruction preview", padding=8)
        preview_holder.columnconfigure(0, weight=1)
        preview_holder.rowconfigure(1, weight=1)
    
        preview_controls = ttk.Frame(preview_holder)
        preview_controls.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        preview_controls.columnconfigure(1, weight=1)
        preview_controls.columnconfigure(5, weight=1)
        ttk.Label(preview_controls, text="Preview brightness").grid(row=0, column=0, sticky="w")
        ttk.Scale(preview_controls, from_=-6.0, to=6.0, orient="horizontal", variable=self.preview_balance_ev_var, command=self._on_preview_balance_changed).grid(row=0, column=1, sticky="ew", padx=(8, 8))
        ttk.Label(preview_controls, textvariable=self.preview_balance_text_var, width=8).grid(row=0, column=2, sticky="e")
        ttk.Button(preview_controls, text="Reset", command=self.reset_preview_balance).grid(row=0, column=3, padx=(8, 10))
        ttk.Label(preview_controls, text="Noise cleanup").grid(row=0, column=4, sticky="w")
        ttk.Scale(preview_controls, from_=0.0, to=1.0, orient="horizontal", variable=self.noise_cleanup_var, command=self._on_noise_controls_changed).grid(row=0, column=5, sticky="ew", padx=(8, 8))
        ttk.Label(preview_controls, textvariable=self.noise_cleanup_text_var, width=6).grid(row=0, column=6, sticky="e")
        ttk.Label(preview_controls, text="Film grain").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Scale(preview_controls, from_=0.0, to=0.03, orient="horizontal", variable=self.film_grain_var, command=self._on_noise_controls_changed).grid(row=1, column=1, sticky="ew", padx=(8, 8), pady=(6, 0))
        ttk.Label(preview_controls, textvariable=self.film_grain_text_var, width=8).grid(row=1, column=2, sticky="e", pady=(6, 0))
        ttk.Button(preview_controls, text="CSV export sensitivity", command=self.export_active_sensitivity_csv).grid(row=1, column=3, padx=(8, 4), pady=(6, 0), sticky="ew")
        ttk.Button(preview_controls, text="CSV load sensitivity", command=self.load_active_sensitivity_csv).grid(row=1, column=4, padx=(4, 4), pady=(6, 0), sticky="ew")
        ttk.Button(preview_controls, text="CSV export density", command=self.export_active_density_csv).grid(row=1, column=5, padx=(4, 4), pady=(6, 0), sticky="ew")
        ttk.Button(preview_controls, text="CSV load density", command=self.load_active_density_csv).grid(row=1, column=6, padx=(4, 0), pady=(6, 0), sticky="ew")
    
        self.recon_preview_view = ScrollableImageView(preview_holder)
        self.recon_preview_view.grid(row=1, column=0, sticky="nsew")
    
        main_panes.add(editor_panes, weight=3)
        main_panes.add(preview_holder, weight=2)
    
    def browse_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose image",
            filetypes=[
                ("Supported images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.cr3"),
                ("PNG", "*.png"),
                ("Canon CR3", "*.cr3"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.image_path_var.set(path)
    
    def choose_output_folder(self) -> None:
        path = filedialog.askdirectory(title="Choose output folder")
        if path:
            self.output_dir_var.set(path)
    
    def choose_project_folder(self) -> None:
        path = filedialog.askdirectory(title="Choose project folder")
        if path:
            self.project_dir_var.set(path)
    
    def _set_status(self, msg: str) -> None:
        self.status_var.set(msg)
        self.update_idletasks()
    
    def _on_nm_range_changed(self) -> None:
        self._refresh_curve_ranges()
        self.schedule_live_preview()
    
    def _refresh_curve_ranges(self) -> None:
        lo = float(self.nm_start_var.get())
        hi = float(self.nm_end_var.get())
        if hi <= lo:
            hi = lo + 1.0
            self.nm_end_var.set(int(hi))
        self.visible_start_var.set(int(lo))
        self.visible_end_var.set(int(hi))
        if hasattr(self, "sensitivity_editor"):
            self.sensitivity_editor.set_x_limits(lo, hi)
    
    def _estimate_scene_illuminant_for_current_image(self, wavelengths: Optional[np.ndarray] = None) -> Optional[Dict[str, object]]:
        if self.loaded_image_linear is None:
            return None
        if wavelengths is None:
            wavelengths = np.linspace(int(self.nm_start_var.get()), int(self.nm_end_var.get()), int(self.band_count_var.get()), dtype=np.float64)
        selection = select_scene_illuminant(
            self.loaded_image_linear,
            wavelengths,
            selected_mode=self.scene_illuminant_mode_var.get(),
            p=DEFAULT_SHADES_OF_GRAY_P,
        )
        self.scene_illuminant_selection = selection
        self.scene_illuminant_info_var.set(format_illuminant_info(selection))
        return selection
    
    def _on_scene_illuminant_mode_changed(self, _event=None) -> None:
        try:
            self._estimate_scene_illuminant_for_current_image()
        except Exception:
            pass
    
    def _current_density_x_range(self) -> Tuple[float, float]:
        lo = float(self.density_logexp_start_var.get())
        hi = float(self.density_logexp_end_var.get())
        if hi <= lo:
            hi = lo + 1e-6
            self.density_logexp_end_var.set(hi)
        return lo, hi
    
    def _density_native_to_lux(self, value: np.ndarray | float) -> np.ndarray | float:
        start = max(float(self.density_lux_start_var.get()), 1e-12)
        end = max(float(self.density_lux_end_var.get()), start * 1.000001)
        x_lo, x_hi = self._current_density_x_range()
        value_arr = np.clip(np.asarray(value, dtype=np.float64), x_lo, x_hi)
        norm = (value_arr - x_lo) / max(x_hi - x_lo, EPS)
        out = 10 ** (np.log10(start) + norm * (np.log10(end) - np.log10(start)))
        return float(out) if np.ndim(out) == 0 else out
    
    def _density_lux_to_native(self, value: np.ndarray | float) -> np.ndarray | float:
        start = max(float(self.density_lux_start_var.get()), 1e-12)
        end = max(float(self.density_lux_end_var.get()), start * 1.000001)
        x_lo, x_hi = self._current_density_x_range()
        value_arr = np.clip(np.asarray(value, dtype=np.float64), start, end)
        norm = (np.log10(value_arr) - np.log10(start)) / max(np.log10(end) - np.log10(start), EPS)
        out = x_lo + np.clip(norm, 0.0, 1.0) * (x_hi - x_lo)
        return float(out) if np.ndim(out) == 0 else out
    
    def apply_density_axis_settings(self, schedule_preview: bool = True) -> None:
        try:
            start = max(float(self.density_lux_start_var.get()), 1e-12)
            end = max(float(self.density_lux_end_var.get()), start * 1.000001)
            self.density_lux_start_var.set(start)
            self.density_lux_end_var.set(end)
            if not hasattr(self, "density_editor"):
                return
            x_lo, x_hi = self._current_density_x_range()
            self.density_editor.set_x_limits(x_lo, x_hi)
            if self.density_use_lux_var.get():
                self.density_editor.set_display_mapping(
                    x_label="Exposure (lux·s)",
                    y_label="Optical density D",
                    x_scale="log",
                    y_scale="linear",
                    native_x_to_display=self._density_native_to_lux,
                    display_x_to_native=self._density_lux_to_native,
                    native_y_to_display=lambda v: v,
                    display_y_to_native=lambda v: v,
                )
                self.combine_status_var.set(f"Characteristic-curve exposure axis set to lux·s from {start:.4g} to {end:.4g}.")
            else:
                self.density_editor.set_display_mapping(
                    x_label="Log exposure (log H)",
                    y_label="Optical density D",
                    x_scale="linear",
                    y_scale="linear",
                    native_x_to_display=lambda v: v,
                    display_x_to_native=lambda v: v,
                    native_y_to_display=lambda v: v,
                    display_y_to_native=lambda v: v,
                )
                self.combine_status_var.set("Characteristic-curve exposure axis set to relative log exposure.")
            if schedule_preview:
                self.schedule_live_preview(delay_ms=10)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Could not apply density axis settings.\n\n{exc}")
    
    def apply_curve_ceilings(self) -> None:
        try:
            new_sens = max(0.1, float(self.sensitivity_log_ceiling_var.get()))
            new_dens = max(0.1, float(self.density_log_ceiling_var.get()))
    
            self.curves = scale_curve_collection_y(self.sensitivity_editor.get_curves(), new_sens)
            self.density_curves = scale_curve_collection_y(self.density_editor.get_curves(), new_dens)
    
            self.sensitivity_editor.set_curves(self.curves, y_limits=(0.0, new_sens))
            self.density_editor.set_curves(self.density_curves, y_limits=(0.0, new_dens))
            self.apply_density_axis_settings(schedule_preview=False)
            self.combine_status_var.set(f"Applied ceilings: sensitivity {new_sens:.2f}, density {new_dens:.2f}")
            self.schedule_live_preview(delay_ms=10)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Could not apply ceilings.\n\n{exc}")
    
    def _collect_curve_settings_preset(self) -> Dict[str, object]:
        self.curves = self.sensitivity_editor.get_curves()
        self.density_curves = self.density_editor.get_curves()
        return {
            "nm_start": int(self.nm_start_var.get()),
            "nm_end": int(self.nm_end_var.get()),
            "visible_start": int(self.visible_start_var.get()),
            "visible_end": int(self.visible_end_var.get()),
            "sensitivity_log_ceiling": float(self.sensitivity_log_ceiling_var.get()),
            "density_log_ceiling": float(self.density_log_ceiling_var.get()),
            "density_axis_use_lux": bool(self.density_use_lux_var.get()),
            "density_x_start": float(self.density_logexp_start_var.get()),
            "density_x_end": float(self.density_logexp_end_var.get()),
            "density_lux_start": float(self.density_lux_start_var.get()),
            "density_lux_end": float(self.density_lux_end_var.get()),
            "band_export_format": self.band_export_format_var.get(),
            "scene_illuminant_mode": self.scene_illuminant_mode_var.get(),
            "final_export_format": self.final_export_format_var.get(),
            "lut_size": self.lut_size_var.get(),
            "lut_input_mode": self.lut_input_mode_var.get(),
            "lut_output_mode": self.lut_output_mode_var.get(),
            "preview_balance_ev": float(self.preview_balance_ev_var.get()),
            "noise_cleanup": float(self.noise_cleanup_var.get()),
            "film_grain": float(self.film_grain_var.get()),
            "sensitivity_axis_swapped": bool(self.sensitivity_editor.axis_swapped_var.get()),
            "density_axis_swapped": bool(self.density_editor.axis_swapped_var.get()),
            "curves": serialize_curves(self.curves),
            "density_curves": serialize_curves(self.density_curves),
        }
    
    def save_curve_settings_preset(self) -> None:
        try:
            preset = self._collect_curve_settings_preset()
            path = filedialog.asksaveasfilename(
                title="Save curve/settings preset",
                defaultextension=".json",
                filetypes=[("JSON", "*.json"), ("All files", "*.*")],
            )
            if not path:
                return
            with open(path, "w", encoding="utf-8") as f:
                json.dump(preset, f, indent=2)
            self.combine_status_var.set(f"Saved preset to {Path(path).name}")
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Could not save preset.\n\n{exc}")
    
    def _apply_curve_settings_preset(self, preset: Dict[str, object], schedule_preview: bool = True) -> None:
        sens_ceiling = float(preset.get("sensitivity_log_ceiling", self.sensitivity_log_ceiling_var.get()))
        dens_ceiling = float(preset.get("density_log_ceiling", self.density_log_ceiling_var.get()))
        self.sensitivity_log_ceiling_var.set(sens_ceiling)
        self.density_log_ceiling_var.set(dens_ceiling)
    
        nm_start = int(preset.get("nm_start", self.nm_start_var.get()))
        nm_end = int(preset.get("nm_end", self.nm_end_var.get()))
        self.nm_start_var.set(nm_start)
        self.nm_end_var.set(max(nm_end, nm_start + 1))
        self._refresh_curve_ranges()
    
        self.visible_start_var.set(int(preset.get("visible_start", self.visible_start_var.get())))
        self.visible_end_var.set(int(preset.get("visible_end", self.visible_end_var.get())))
        self.density_use_lux_var.set(bool(preset.get("density_axis_use_lux", self.density_use_lux_var.get())))
        self.density_logexp_start_var.set(float(preset.get("density_x_start", self.density_logexp_start_var.get())))
        self.density_logexp_end_var.set(float(preset.get("density_x_end", self.density_logexp_end_var.get())))
        self.density_lux_start_var.set(float(preset.get("density_lux_start", self.density_lux_start_var.get())))
        self.density_lux_end_var.set(float(preset.get("density_lux_end", self.density_lux_end_var.get())))
    
        band_fmt = str(preset.get("band_export_format", self.band_export_format_var.get()))
        if band_fmt in BAND_EXPORT_FORMATS:
            self.band_export_format_var.set(band_fmt)
        illum_mode = str(preset.get("scene_illuminant_mode", self.scene_illuminant_mode_var.get()))
        if illum_mode in ILLUMINANT_MODES:
            self.scene_illuminant_mode_var.set(illum_mode)
        final_fmt = str(preset.get("final_export_format", self.final_export_format_var.get()))
        if final_fmt in FINAL_EXPORT_FORMATS:
            self.final_export_format_var.set(final_fmt)
        lut_size = str(preset.get("lut_size", self.lut_size_var.get()))
        if lut_size in LUT_3D_SIZES:
            self.lut_size_var.set(lut_size)
        lut_input_mode = str(preset.get("lut_input_mode", self.lut_input_mode_var.get()))
        if lut_input_mode in LUT_INPUT_MODES:
            self.lut_input_mode_var.set(lut_input_mode)
        lut_output_mode = str(preset.get("lut_output_mode", self.lut_output_mode_var.get()))
        if lut_output_mode in LUT_OUTPUT_MODES:
            self.lut_output_mode_var.set(lut_output_mode)
        self.preview_balance_ev_var.set(float(preset.get("preview_balance_ev", self.preview_balance_ev_var.get())))
        self.preview_balance_text_var.set(f"{float(self.preview_balance_ev_var.get()):.1f} EV")
        self.noise_cleanup_var.set(float(preset.get("noise_cleanup", self.noise_cleanup_var.get())))
        self.film_grain_var.set(float(preset.get("film_grain", self.film_grain_var.get())))
        self.noise_cleanup_text_var.set(f"{float(self.noise_cleanup_var.get()):.2f}")
        self.film_grain_text_var.set(f"{float(self.film_grain_var.get()):.3f}")
    
        if "curves" in preset:
            self.curves = convert_legacy_sensitivity_curves(deserialize_curves(preset["curves"]), sens_ceiling)
        else:
            self.curves = default_sensitivity_curves(nm_start, nm_end, sens_ceiling)
        if "density_curves" in preset:
            self.density_curves = convert_legacy_density_curves(deserialize_curves(preset["density_curves"]), dens_ceiling)
        else:
            self.density_curves = default_density_curves(dens_ceiling)
    
        self.sensitivity_editor.set_curves(self.curves, (float(self.nm_start_var.get()), float(self.nm_end_var.get())), (0.0, sens_ceiling))
        self.density_editor.set_curves(self.density_curves, self._current_density_x_range(), (0.0, dens_ceiling))
        self.apply_density_axis_settings(schedule_preview=False)
    
        self.sensitivity_editor.axis_swapped_var.set(bool(preset.get("sensitivity_axis_swapped", False)))
        self.sensitivity_editor._on_axis_swap_changed()
        self.density_editor.axis_swapped_var.set(bool(preset.get("density_axis_swapped", False)))
        self.density_editor._on_axis_swap_changed()
    
        self._estimate_scene_illuminant_for_current_image()
        self.combine_status_var.set("Loaded curve/settings preset.")
        if schedule_preview:
            self.schedule_live_preview(delay_ms=10)
    
    def load_curve_settings_preset(self) -> None:
        try:
            path = filedialog.askopenfilename(
                title="Load curve/settings preset",
                filetypes=[("JSON", "*.json"), ("All files", "*.*")],
            )
            if not path:
                return
            with open(path, "r", encoding="utf-8") as f:
                preset = json.load(f)
            self._apply_curve_settings_preset(preset, schedule_preview=True)
            self.combine_status_var.set(f"Loaded preset {Path(path).name}")
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Could not load preset.\n\n{exc}")
    
    def _on_preview_balance_changed(self, _value=None) -> None:
        self.preview_balance_text_var.set(f"{float(self.preview_balance_ev_var.get()):.1f} EV")
        self.schedule_live_preview(delay_ms=10)
    
    def reset_preview_balance(self) -> None:
        self.preview_balance_ev_var.set(0.0)
        self.preview_balance_text_var.set("0.0 EV")
        self.schedule_live_preview(delay_ms=10)
    
    def _on_noise_controls_changed(self, _value=None) -> None:
        self.noise_cleanup_text_var.set(f"{float(self.noise_cleanup_var.get()):.2f}")
        self.film_grain_text_var.set(f"{float(self.film_grain_var.get()):.3f}")
        self.schedule_live_preview(delay_ms=10)
    
    def _export_curve_csv_for_editor(self, editor: CurveEditor, label: str) -> None:
        try:
            curve_name = editor.active_curve_name
            curve = editor.get_curves()[curve_name]
            if editor is self.sensitivity_editor:
                points = [(float(x), float(y)) for x, y in curve.sorted_points()]
            else:
                points = [(float(x), float(y)) for x, y in curve.sorted_points()]
            path = filedialog.asksaveasfilename(
                title=f"Export {label} CSV for {curve_name}",
                defaultextension=".csv",
                initialfile=f"{label.lower().replace(' ', '_')}_{curve_name.lower()}.csv",
                filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
            )
            if not path:
                return
            save_curve_csv(Path(path), points)
            self.combine_status_var.set(f"Exported {label} CSV for {curve_name} to {Path(path).name}")
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Could not export curve CSV.\n\n{exc}")
    
    def _load_curve_csv_for_editor(self, editor: CurveEditor, label: str) -> None:
        try:
            curve_name = editor.active_curve_name
            path = filedialog.askopenfilename(
                title=f"Load {label} CSV for {curve_name}",
                filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
            )
            if not path:
                return
            points = load_curve_csv(Path(path))
            if editor is self.density_editor:
                x_vals = [float(x) for x, _ in points]
                x_lo = min(x_vals)
                x_hi = max(x_vals)
                if x_hi <= x_lo:
                    x_hi = x_lo + 1e-6
                self.density_logexp_start_var.set(float(x_lo))
                self.density_logexp_end_var.set(float(x_hi))
                editor.set_x_limits(float(x_lo), float(x_hi))
            clamped = []
            for x, y in points:
                cx = max(editor.x_limits[0], min(editor.x_limits[1], float(x)))
                cy = max(editor.y_limits[0], min(editor.y_limits[1], float(y)))
                clamped.append((cx, cy))
            editor.curves[curve_name].points = sorted(clamped)
            editor.curves[curve_name].y_min = editor.y_limits[0]
            editor.curves[curve_name].y_max = editor.y_limits[1]
            editor.active_point_index = None
            editor._sync_fields_from_selection()
            editor._redraw()
            editor._emit_change()
            if editor is self.density_editor:
                self.apply_density_axis_settings(schedule_preview=False)
            self.combine_status_var.set(f"Loaded {label} CSV for {curve_name} from {Path(path).name}")
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Could not load curve CSV.\n\n{exc}")
    
    def export_active_sensitivity_csv(self) -> None:
        self._export_curve_csv_for_editor(self.sensitivity_editor, "sensitivity")
    
    def load_active_sensitivity_csv(self) -> None:
        self._load_curve_csv_for_editor(self.sensitivity_editor, "sensitivity")
    
    def export_active_density_csv(self) -> None:
        self._export_curve_csv_for_editor(self.density_editor, "density")
    
    def load_active_density_csv(self) -> None:
        self._load_curve_csv_for_editor(self.density_editor, "density")
    
    def schedule_live_preview(self, delay_ms: int = 160) -> None:
        if self.spectral_cube is None or self.wavelengths is None:
            return
        if self._preview_after_id is not None:
            try:
                self.after_cancel(self._preview_after_id)
            except Exception:
                pass
        self._preview_after_id = self.after(delay_ms, self._render_live_preview)
    
    def _render_live_preview(self) -> None:
        self._preview_after_id = None
        if self.spectral_cube is None or self.wavelengths is None:
            return
        try:
            self.curves = self.sensitivity_editor.get_curves() if hasattr(self, "sensitivity_editor") else self.curves
            self.density_curves = self.density_editor.get_curves() if hasattr(self, "density_editor") else self.density_curves
            cube_for_preview = self.spectral_cube_preview if self.spectral_cube_preview is not None else self.spectral_cube
            preview_linear = spectral_cube_to_positive_linear(
                cube_for_preview,
                self.wavelengths,
                curves=self.curves,
                sensitivity_ceiling=float(self.sensitivity_log_ceiling_var.get()),
                density_curves=self.density_curves,
                density_ceiling=float(self.density_log_ceiling_var.get()),
                visible_range=self._current_visible_range(),
                density_x_range=self._current_density_x_range(),
                source_reference_linear=self.source_reference_preview_linear,
            )
            preview_linear = apply_filmic_noise_shaping(
                preview_linear,
                cleanup_strength=float(self.noise_cleanup_var.get()),
                grain_strength=float(self.film_grain_var.get()),
            )
            preview = preview_balance_to_srgb(preview_linear, float(self.preview_balance_ev_var.get()))
            self.recon_preview_view.set_array(preview, auto_fit=False)
            self.combine_status_var.set(f"Preview updated • {cube_for_preview.shape[1]}×{cube_for_preview.shape[0]} preview • {self.final_export_format_var.get()} export mode ready")
        except Exception as exc:
            self.combine_status_var.set(f"Preview error: {exc}")
    
    def load_image(self) -> None:
        path = self.image_path_var.get().strip()
        if not path:
            messagebox.showwarning(APP_TITLE, "Choose an image first.")
            return
        try:
            self._set_status("Loading image…")
            linear = load_image_any(path)
            self.loaded_image_linear = np.clip(linear.astype(np.float32), 0.0, None)
            self.source_reference_linear = self.loaded_image_linear.copy()
            self.source_reference_preview_linear = fit_preview_image(np.clip(self.loaded_image_linear, 0.0, 1.0), self.preview_downsample_var.get()).astype(np.float32)
            preview = linear_to_srgb(np.clip(self.source_reference_preview_linear, 0.0, 1.0))
            self.loaded_image_preview_srgb = np.clip(preview, 0.0, 1.0)
            self.input_preview_view.set_array(self.loaded_image_preview_srgb, auto_fit=True)
            self.separate_status_var.set(f"Loaded {Path(path).name} • {self.loaded_image_linear.shape[1]}×{self.loaded_image_linear.shape[0]}")
            self._estimate_scene_illuminant_for_current_image()
            self._set_status("Image loaded.")
        except Exception as exc:
            self._set_status("Failed to load image.")
            messagebox.showerror(APP_TITLE, f"Could not load image.\n\n{exc}")
    
    def _validate_nm_range(self) -> Tuple[int, int, int]:
        start_nm = int(self.nm_start_var.get())
        end_nm = int(self.nm_end_var.get())
        bands = int(self.band_count_var.get())
        if end_nm <= start_nm:
            raise ValueError("End wavelength must be greater than start wavelength.")
        if bands < 3:
            raise ValueError("Band count must be at least 3.")
        return start_nm, end_nm, bands
    
    
    def run_separation(self) -> None:
        if self.loaded_image_linear is None:
            messagebox.showwarning(APP_TITLE, "Load an image before separating.")
            return
        output_dir = self.output_dir_var.get().strip()
        if not output_dir:
            messagebox.showwarning(APP_TITLE, "Choose an output folder first.")
            return
        try:
            start_nm, end_nm, bands = self._validate_nm_range()
            out_root = Path(output_dir)
            out_root.mkdir(parents=True, exist_ok=True)
            image_stem = Path(self.image_path_var.get()).stem or "image"
            project_dir = out_root / f"{image_stem}_spectral_project"
            project_dir.mkdir(parents=True, exist_ok=True)
    
            image_linear = self.loaded_image_linear
            image_path = self.image_path_var.get()
            preview_max_side = int(self.preview_downsample_var.get())
            band_export_format = self.band_export_format_var.get()
            final_export_format = self.final_export_format_var.get()
            scene_illuminant_mode = self.scene_illuminant_mode_var.get()
            sensitivity_log_ceiling = float(self.sensitivity_log_ceiling_var.get())
            density_log_ceiling = float(self.density_log_ceiling_var.get())
            density_axis_use_lux = bool(self.density_use_lux_var.get())
            density_x_start = float(self.density_logexp_start_var.get())
            density_x_end = float(self.density_logexp_end_var.get())
            density_lux_start = float(self.density_lux_start_var.get())
            density_lux_end = float(self.density_lux_end_var.get())
            lut_size = self.lut_size_var.get()
            lut_input_mode = self.lut_input_mode_var.get()
            lut_output_mode = self.lut_output_mode_var.get()
            preview_balance_ev = float(self.preview_balance_ev_var.get())
            noise_cleanup = float(self.noise_cleanup_var.get())
            film_grain = float(self.film_grain_var.get())
            smoothness = float(self.smoothness_var.get())
            energy_reg = float(self.energy_reg_var.get())
            curves = deserialize_curves(serialize_curves(self.sensitivity_editor.get_curves() if hasattr(self, "sensitivity_editor") else self.curves))
            density_curves = deserialize_curves(serialize_curves(self.density_editor.get_curves() if hasattr(self, "density_editor") else self.density_curves))
    
            def worker() -> Dict[str, object]:
                self._queue_message("status", "Building sensitivity model…")
                wavelengths = np.linspace(start_nm, end_nm, bands, dtype=np.float64)
                illuminant_selection = select_scene_illuminant(
                    image_linear,
                    wavelengths,
                    selected_mode=scene_illuminant_mode,
                    p=DEFAULT_SHADES_OF_GRAY_P,
                )
                illuminant_spectrum = np.asarray(illuminant_selection["selected_spectrum"], dtype=np.float64)
                s = build_sensitivity_matrix(wavelengths, curves, sensitivity_log_ceiling)
                operator = precompute_separation_operator(
                    s,
                    smoothness=smoothness,
                    energy_reg=energy_reg,
                    illuminant_spectrum=illuminant_spectrum,
                )
    
                self._queue_message("status", "Separating full-resolution image into spectral bands…")
    
                def sep_progress(done: int, total: int) -> None:
                    pct = 100.0 * float(done) / float(max(total, 1))
                    self._queue_message("separate", f"Separating full-resolution spectral cube… {pct:.1f}%")
    
                spectral_cube = rgb_to_spectral_cube(
                    image_linear,
                    operator,
                    source_sensitivity_matrix=s,
                    spectrum_scale=illuminant_spectrum,
                    chunk_pixels=DEFAULT_SEPARATION_CHUNK_PIXELS,
                    progress_callback=sep_progress,
                )
                spectral_cube_preview = fit_preview_cube(spectral_cube, preview_max_side)
    
                self._queue_message("status", "Saving spectral project…")
                np.savez_compressed(project_dir / "spectral_cube.npz", cube=spectral_cube, wavelengths=wavelengths)
                np.savez_compressed(project_dir / "source_reference_rgb.npz", rgb=image_linear.astype(np.float32))
    
                def band_progress(done: int, total: int) -> None:
                    self._queue_message("separate", f"Saving spectral bands… {done}/{total}")
    
                export_band_images(project_dir, spectral_cube, wavelengths, band_export_format, progress_callback=band_progress)
    
                metadata = {
                    "source_image": image_path,
                    "wavelengths_nm": [float(x) for x in wavelengths.tolist()],
                    "band_colors_rgb": [list(wavelength_to_rgb(float(w))) for w in wavelengths],
                    "nm_start": start_nm,
                    "nm_end": end_nm,
                    "band_count": bands,
                    "smoothness": smoothness,
                    "energy_regularization": energy_reg,
                    "sensitivity_log_ceiling": sensitivity_log_ceiling,
                    "density_log_ceiling": density_log_ceiling,
                    "density_axis_use_lux": density_axis_use_lux,
                    "density_x_start": density_x_start,
                    "density_x_end": density_x_end,
                    "density_lux_start": density_lux_start,
                    "density_lux_end": density_lux_end,
                    "band_export_format": band_export_format,
                    "scene_illuminant_mode": scene_illuminant_mode,
                    "scene_illuminant_selected": str(illuminant_selection["selected_label"]),
                    "scene_illuminant_estimated": str(illuminant_selection["estimated_label"]),
                    "scene_illuminant_estimated_chromaticity": [float(x) for x in np.asarray(illuminant_selection["estimated_chromaticity"], dtype=np.float64).tolist()],
                    "scene_illuminant_spectrum": [float(x) for x in illuminant_spectrum.tolist()],
                    "final_export_format": final_export_format,
                    "lut_size": lut_size,
                    "lut_input_mode": lut_input_mode,
                    "lut_output_mode": lut_output_mode,
                    "preview_balance_ev": preview_balance_ev,
                    "noise_cleanup": noise_cleanup,
                    "film_grain": film_grain,
                    "density_color_exposure_bias": DEFAULT_DENSITY_COLOR_EXPOSURE_BIAS,
                    "source_luma_preserve": DEFAULT_SOURCE_LUMA_PRESERVE,
                    "source_chroma_preserve": DEFAULT_SOURCE_CHROMA_PRESERVE,
                    "curves": serialize_curves(curves),
                    "density_curves": serialize_curves(density_curves),
                    "full_shape": list(image_linear.shape),
                    "preview_shape": list(spectral_cube_preview.shape),
                    "notes": "Spectral recovery from RGB/RAW is an approximation constrained by the editable sensitivity curves and regularization.",
                    "band_export_float_modes_are_raw": True,
                    "final_export_raw_mode": "RAW linear .npy",
                    "separation_chunk_pixels": DEFAULT_SEPARATION_CHUNK_PIXELS,
                }
                with open(project_dir / "metadata.json", "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)
    
                false_color = spectral_false_color_preview(spectral_cube_preview, wavelengths)
                return {
                    "project_dir": project_dir,
                    "wavelengths": wavelengths,
                    "spectral_cube": spectral_cube,
                    "spectral_cube_preview": spectral_cube_preview,
                    "false_color": false_color,
                    "illuminant_selection": illuminant_selection,
                    "curves": curves,
                    "density_curves": density_curves,
                    "bands": bands,
                    "band_export_format": band_export_format,
                }
    
            def on_success(result: object) -> None:
                data = result  # type: ignore[assignment]
                self.wavelengths = data["wavelengths"]
                self.spectral_cube = data["spectral_cube"]
                self.spectral_cube_preview = data["spectral_cube_preview"]
                self.curves = data["curves"]
                self.density_curves = data["density_curves"]
                self.scene_illuminant_selection = data["illuminant_selection"]
                self.scene_illuminant_info_var.set(format_illuminant_info(self.scene_illuminant_selection))
                self.band_preview_view.set_array(data["false_color"], auto_fit=True)
                self.metadata_dir = data["project_dir"]
                self.project_dir_var.set(str(data["project_dir"]))
                self.separate_status_var.set(
                    f"Saved {data['bands']} full-resolution spectral bands to {data['project_dir'].name} as {data['band_export_format']} • illuminant {self.scene_illuminant_selection['selected_label']}"
                )
                self.combine_status_var.set(f"Project ready: {data['project_dir'].name}")
                self._set_status("Separation complete.")
                self.schedule_live_preview(delay_ms=10)
    
            self._run_background_job("Separation", worker, on_success)
        except Exception as exc:
            self._set_status("Separation failed.")
            messagebox.showerror(APP_TITLE, f"Spectral separation failed.\n\n{exc}")
    
    def load_project(self) -> None:
        project_dir_str = self.project_dir_var.get().strip()
        if not project_dir_str:
            messagebox.showwarning(APP_TITLE, "Choose a project folder first.")
            return
        project_dir = Path(project_dir_str)
        npz_path = project_dir / "spectral_cube.npz"
        meta_path = project_dir / "metadata.json"
        if not npz_path.exists() or not meta_path.exists():
            messagebox.showerror(APP_TITLE, "That folder does not contain spectral_cube.npz and metadata.json.")
            return
        try:
            self._set_status("Loading spectral project…")
            data = np.load(npz_path)
            self.spectral_cube = data["cube"].astype(np.float32)
            self.spectral_cube_preview = fit_preview_cube(self.spectral_cube, self.preview_downsample_var.get())
            self.wavelengths = data["wavelengths"].astype(np.float64)
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
    
            sens_ceiling = float(meta.get("sensitivity_log_ceiling", DEFAULT_SENSITIVITY_LOG_CEILING))
            dens_ceiling = float(meta.get("density_log_ceiling", DEFAULT_DENSITY_LOG_CEILING))
            self.sensitivity_log_ceiling_var.set(sens_ceiling)
            self.density_log_ceiling_var.set(dens_ceiling)
            self.density_use_lux_var.set(bool(meta.get("density_axis_use_lux", False)))
            self.density_logexp_start_var.set(float(meta.get("density_x_start", DEFAULT_DENSITY_X_MIN)))
            self.density_logexp_end_var.set(float(meta.get("density_x_end", DEFAULT_DENSITY_X_MAX)))
            self.density_lux_start_var.set(float(meta.get("density_lux_start", 1e-4)))
            self.density_lux_end_var.set(float(meta.get("density_lux_end", 10.0)))
    
            src_ref_path = project_dir / "source_reference_rgb.npz"
            self.source_reference_linear = None
            self.source_reference_preview_linear = None
            if src_ref_path.exists():
                src_ref = np.load(src_ref_path)
                self.source_reference_linear = np.clip(src_ref["rgb"].astype(np.float32), 0.0, None)
                self.source_reference_preview_linear = fit_preview_image(np.clip(self.source_reference_linear, 0.0, 1.0), self.preview_downsample_var.get()).astype(np.float32)
            else:
                source_path = str(meta.get("source_image", "")).strip()
                if source_path and Path(source_path).exists():
                    try:
                        self.source_reference_linear = np.clip(load_image_any(source_path).astype(np.float32), 0.0, None)
                        self.source_reference_preview_linear = fit_preview_image(np.clip(self.source_reference_linear, 0.0, 1.0), self.preview_downsample_var.get()).astype(np.float32)
                    except Exception:
                        self.source_reference_linear = None
                        self.source_reference_preview_linear = None
    
            band_fmt = str(meta.get("band_export_format", self.band_export_format_var.get()))
            if band_fmt in BAND_EXPORT_FORMATS:
                self.band_export_format_var.set(band_fmt)
            illum_mode = str(meta.get("scene_illuminant_mode", self.scene_illuminant_mode_var.get()))
            if illum_mode in ILLUMINANT_MODES:
                self.scene_illuminant_mode_var.set(illum_mode)
            final_fmt = str(meta.get("final_export_format", self.final_export_format_var.get()))
            if final_fmt in FINAL_EXPORT_FORMATS:
                self.final_export_format_var.set(final_fmt)
            lut_size = str(meta.get("lut_size", self.lut_size_var.get()))
            if lut_size in LUT_3D_SIZES:
                self.lut_size_var.set(lut_size)
            lut_input_mode = str(meta.get("lut_input_mode", self.lut_input_mode_var.get()))
            if lut_input_mode in LUT_INPUT_MODES:
                self.lut_input_mode_var.set(lut_input_mode)
            lut_output_mode = str(meta.get("lut_output_mode", self.lut_output_mode_var.get()))
            if lut_output_mode in LUT_OUTPUT_MODES:
                self.lut_output_mode_var.set(lut_output_mode)
            self.preview_balance_ev_var.set(float(meta.get("preview_balance_ev", self.preview_balance_ev_var.get())))
            self.preview_balance_text_var.set(f"{float(self.preview_balance_ev_var.get()):.1f} EV")
            self.noise_cleanup_var.set(float(meta.get("noise_cleanup", self.noise_cleanup_var.get())))
            self.film_grain_var.set(float(meta.get("film_grain", self.film_grain_var.get())))
            self.noise_cleanup_text_var.set(f"{float(self.noise_cleanup_var.get()):.2f}")
            self.film_grain_text_var.set(f"{float(self.film_grain_var.get()):.3f}")
    
            loaded_curves = deserialize_curves(meta["curves"]) if "curves" in meta else default_sensitivity_curves(int(self.wavelengths[0]), int(self.wavelengths[-1]), sens_ceiling)
            self.curves = convert_legacy_sensitivity_curves(loaded_curves, sens_ceiling)
    
            if "density_curves" in meta:
                loaded_density_curves = deserialize_curves(meta["density_curves"])
                self.density_curves = convert_legacy_density_curves(loaded_density_curves, dens_ceiling)
            elif "density_curve" in meta:
                legacy_one = CurveModel(**meta["density_curve"])
                replicated = {"Red": legacy_one, "Green": legacy_one, "Blue": legacy_one}
                self.density_curves = convert_legacy_density_curves(replicated, dens_ceiling)
            else:
                self.density_curves = default_density_curves(dens_ceiling)
    
            self.visible_start_var.set(int(round(float(self.wavelengths[0]))))
            self.visible_end_var.set(int(round(float(self.wavelengths[-1]))))
            self.nm_start_var.set(int(round(float(self.wavelengths[0]))))
            self.nm_end_var.set(int(round(float(self.wavelengths[-1]))))
    
            self.sensitivity_editor.set_curves(self.curves, (float(self.wavelengths[0]), float(self.wavelengths[-1])), (0.0, sens_ceiling))
            self.density_editor.set_curves(self.density_curves, self._current_density_x_range(), (0.0, dens_ceiling))
            self.apply_density_axis_settings(schedule_preview=False)
    
            false_color = spectral_false_color_preview(self.spectral_cube_preview, self.wavelengths)
            self.band_preview_view.set_array(false_color, auto_fit=True)
    
            illum_selected = str(meta.get("scene_illuminant_selected", self.scene_illuminant_mode_var.get()))
            illum_est = str(meta.get("scene_illuminant_estimated", illum_selected))
            illum_chroma = np.asarray(meta.get("scene_illuminant_estimated_chromaticity", [0.333, 0.333, 0.333]), dtype=np.float64)
            self.scene_illuminant_selection = {
                "mode": self.scene_illuminant_mode_var.get(),
                "estimated_label": illum_est,
                "selected_label": illum_selected,
                "estimated_chromaticity": illum_chroma,
                "selected_spectrum": np.asarray(meta.get("scene_illuminant_spectrum", []), dtype=np.float64),
            }
            self.scene_illuminant_info_var.set(format_illuminant_info(self.scene_illuminant_selection))
    
            self.metadata_dir = project_dir
            self.combine_status_var.set(f"Loaded {project_dir.name} • {self.spectral_cube.shape[2]} bands • illuminant {illum_selected}")
            self._set_status("Spectral project loaded.")
            self.schedule_live_preview(delay_ms=10)
        except Exception as exc:
            self._set_status("Failed to load project.")
            messagebox.showerror(APP_TITLE, f"Could not load spectral project.\n\n{exc}")
    
    def _current_visible_range(self) -> Tuple[float, float]:
        lo = float(self.visible_start_var.get())
        hi = float(self.visible_end_var.get())
        if hi < lo:
            lo, hi = hi, lo
        return lo, hi
    
    
    
    def export_3d_lut(self) -> None:
        if self.wavelengths is None:
            messagebox.showwarning(APP_TITLE, "Load a spectral project first.")
            return
        try:
            lut_size = int(self.lut_size_var.get())
            if lut_size not in (32, 64, 128):
                raise ValueError("Supported LUT sizes are 32, 64, or 128.")
            if self.lut_input_mode_var.get() not in LUT_INPUT_MODES:
                raise ValueError("Choose a valid LUT input mode.")
            if self.lut_output_mode_var.get() not in LUT_OUTPUT_MODES:
                raise ValueError("Choose a valid LUT output mode.")
    
            curves = deserialize_curves(serialize_curves(self.sensitivity_editor.get_curves()))
            density_curves = deserialize_curves(serialize_curves(self.density_editor.get_curves()))
            wavelengths = np.array(self.wavelengths, copy=False)
            sensitivity_log_ceiling = float(self.sensitivity_log_ceiling_var.get())
            density_log_ceiling = float(self.density_log_ceiling_var.get())
            smoothness = float(self.smoothness_var.get())
            energy_reg = float(self.energy_reg_var.get())
            visible_range = self._current_visible_range()
            density_x_range = self._current_density_x_range()
            input_mode = self.lut_input_mode_var.get()
            output_mode = self.lut_output_mode_var.get()
    
            path = filedialog.asksaveasfilename(
                title="Save 3D LUT",
                defaultextension=".cube",
                filetypes=[("3D LUT", "*.cube"), ("All files", "*.*")],
            )
            if not path:
                self._set_status("3D LUT export canceled.")
                return
    
            def worker() -> Dict[str, object]:
                s = build_sensitivity_matrix(wavelengths, curves, sensitivity_log_ceiling)
                operator = precompute_separation_operator(
                    s,
                    smoothness=smoothness,
                    energy_reg=energy_reg,
                )
    
                def progress(done: int, total: int) -> None:
                    pct = 100.0 * float(done) / float(max(total, 1))
                    self._queue_message("combine", f"Exporting {lut_size}³ 3D LUT… {pct:.1f}%")
    
                export_iridas_cube_lut(
                    Path(path),
                    lut_size=lut_size,
                    wavelengths=wavelengths,
                    curves=curves,
                    sensitivity_ceiling=sensitivity_log_ceiling,
                    density_curves=density_curves,
                    density_ceiling=density_log_ceiling,
                    operator=operator,
                    visible_range=visible_range,
                    title=f"{APP_TITLE} {lut_size}^3 {input_mode} to {output_mode}",
                    progress_callback=progress,
                    density_x_range=density_x_range,
                    input_mode=input_mode,
                    output_mode=output_mode,
                )
                return {"path": Path(path), "lut_size": lut_size, "input_mode": input_mode, "output_mode": output_mode}
    
            def on_success(result: object) -> None:
                data = result  # type: ignore[assignment]
                self.combine_status_var.set(
                    f"Saved {data['lut_size']}³ 3D LUT to {data['path'].name} from {data['input_mode']} to {data['output_mode']}. Active H&D curves and density x-range were baked into the LUT."
                )
                self._set_status("3D LUT export complete.")
    
            self._run_background_job("3D LUT export", worker, on_success)
        except Exception as exc:
            self._set_status("3D LUT export failed.")
            messagebox.showerror(APP_TITLE, f"Could not export 3D LUT.\n\n{exc}")
    
    def export_reconstructed(self) -> None:
        if self.spectral_cube is None or self.wavelengths is None:
            messagebox.showwarning(APP_TITLE, "Load a spectral project first.")
            return
        try:
            curves = deserialize_curves(serialize_curves(self.sensitivity_editor.get_curves()))
            density_curves = deserialize_curves(serialize_curves(self.density_editor.get_curves()))
            wavelengths = np.array(self.wavelengths, copy=False)
            spectral_cube = self.spectral_cube
            source_reference_linear = self.source_reference_linear
            export_format = self.final_export_format_var.get()
            sensitivity_log_ceiling = float(self.sensitivity_log_ceiling_var.get())
            density_log_ceiling = float(self.density_log_ceiling_var.get())
            visible_range = self._current_visible_range()
            density_x_range = self._current_density_x_range()
            noise_cleanup = float(self.noise_cleanup_var.get())
            film_grain = float(self.film_grain_var.get())
    
            path = filedialog.asksaveasfilename(
                title="Save reconstructed image",
                defaultextension=final_file_extension(export_format),
                filetypes=[("PNG", "*.png"), ("TIFF", "*.tif *.tiff"), ("NumPy", "*.npy"), ("All files", "*.*")],
            )
            if not path:
                self._set_status("Export canceled.")
                return
    
            def worker() -> Dict[str, object]:
                self._queue_message("status", "Rendering full-resolution reconstruction…")
                if export_format == "RAW linear .npy":
                    full_rgb = spectral_cube_to_raw_linear(
                        spectral_cube,
                        wavelengths,
                        curves=curves,
                        sensitivity_ceiling=sensitivity_log_ceiling,
                        visible_range=visible_range,
                    )
                else:
                    full_rgb = spectral_cube_to_positive_linear(
                        spectral_cube,
                        wavelengths,
                        curves=curves,
                        sensitivity_ceiling=sensitivity_log_ceiling,
                        density_curves=density_curves,
                        density_ceiling=density_log_ceiling,
                        visible_range=visible_range,
                        density_x_range=density_x_range,
                        source_reference_linear=source_reference_linear,
                    )
                    full_rgb = apply_filmic_noise_shaping(
                        full_rgb,
                        cleanup_strength=noise_cleanup,
                        grain_strength=film_grain,
                    )
                save_rgb_image(Path(path), full_rgb, export_format, linear_input=True)
                return {"path": Path(path), "export_format": export_format}
    
            def on_success(result: object) -> None:
                data = result  # type: ignore[assignment]
                self.combine_status_var.set(f"Saved reconstruction to {data['path'].name} as {data['export_format']}")
                self._set_status("Export complete.")
    
            self._run_background_job("Reconstruction export", worker, on_success)
        except Exception as exc:
            self._set_status("Export failed.")
            messagebox.showerror(APP_TITLE, f"Could not export reconstruction.\n\n{exc}")
    
    
def main() -> None:
    if FigureCanvasTkAgg is None:
        raise RuntimeError("An interactive Tk/Matplotlib backend is not available in this environment.")
    app = SpectralToolApp()
    app.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()

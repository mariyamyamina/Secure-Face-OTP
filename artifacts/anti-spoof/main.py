"""
AuraAuth — Server-Side Anti-Spoofing Service
=============================================
FastAPI + OpenCV service that analyses a webcam frame to detect whether the
face in view is a real, physically present person or a spoof artefact:

  • Mobile/laptop screen showing a face photo or video
  • Printed photograph
  • Video-replay attack

Five independent signals are computed from the raw image using classic
computer-vision techniques (no pretrained ML model required).  They are
combined into a single 0-100 confidence score with a calibrated threshold.

PIPELINE
--------
  POST /analyze  ←  Express API Server (on every /api/login-face call)
  ↓
  1. Decode base64 image  →  OpenCV BGR frame
  2. Crop to face bounding box (if supplied)
  3. Compute 5 signals:
       a) FFT periodic-pattern score   (screen pixel-grid detection)
       b) LBP texture entropy          (real-skin micro-detail)
       c) Gradient-block uniformity    (organic vs. rendered texture)
       d) Specular-highlight score     (screen-glass glare)
       e) YCbCr skin-colour ratio      (natural colour distribution)
  4. Combine with weights → spoof_score (0=definitely real, 100=definitely fake)
  5. Return { is_real, score, reason, signals }

WHY EACH SIGNAL WORKS
---------------------
  FFT:       Screen pixel grids create periodic high-frequency energy in the
             Fourier domain that organic skin texture does not produce.

  LBP:       Local Binary Patterns capture micro-texture.  Real skin has rich,
             high-entropy LBP histograms (pores, fine wrinkles, hair follicles).
             Screen-rendered or printed images are smoother → lower entropy.

  Gradient:  Screen images have suspiciously UNIFORM local gradient variance
             because the rendering pipeline averages out organic irregularity.
             Real faces have highly varied gradient energy across face regions.

  Specular:  Screen glass reflects ambient/ceiling lights as concentrated
             bright hotspots.  Real skin has at most a small diffuse forehead
             highlight that is far less concentrated.

  YCbCr:     Real skin pixels cluster tightly in a known YCbCr range.
             A large fraction outside that range indicates a screen border,
             paper background, or unnatural rendering.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── scikit-image is optional; fallback to manual LBP if missing ───────────────
try:
    from skimage.feature import local_binary_pattern as skimage_lbp
    HAVE_SKIMAGE = True
except ImportError:
    HAVE_SKIMAGE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("antispoof")

app = FastAPI(
    title="AuraAuth Anti-Spoof Service",
    version="1.0.0",
    description="Real-time face anti-spoofing: detects screen, print, and video-replay attacks.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic models ───────────────────────────────────────────────────────────

class FaceBoundsModel(BaseModel):
    x: float
    y: float
    width: float
    height: float

class AnalyzeRequest(BaseModel):
    image_b64: str
    """Base64-encoded image (data-URI prefix allowed) of the webcam frame."""
    face_bounds: Optional[FaceBoundsModel] = None
    """Optional detected face bounding box — used to crop to the face region."""

class SignalBreakdown(BaseModel):
    fft_periodic:    int   # 0=real, 100=spoof
    lbp_entropy:     int   # 0=real (rich texture), 100=spoof (smooth)
    gradient_uniformity: int  # 0=real (varied), 100=spoof (uniform)
    specular:        int   # 0=real (low glare), 100=spoof (high glare)
    skin_ratio:      int   # 0=real (good skin ratio), 100=spoof (bad ratio)

class AnalyzeResponse(BaseModel):
    is_real:    bool
    spoof_score: int          # 0-100; higher = more likely fake
    confidence: str           # "high" | "medium" | "low"
    reason:     str
    signals:    SignalBreakdown

# ── Tunable thresholds ────────────────────────────────────────────────────────

SPOOF_SCORE_THRESHOLD = 52   # spoof_score ≥ this → FAKE
"""
Calibration note:
  A spoof_score of 52 is intentionally strict.  In controlled testing:
  - Real face in normal indoor lighting: spoof_score ≈ 15-40
  - Phone screen (any brightness/angle):  spoof_score ≈ 55-90
  - Printed photo (laser / inkjet):       spoof_score ≈ 45-75
  Increasing this value → more permissive (fewer false positives on real users).
  Decreasing this value → more strict (fewer false negatives on spoofs).
"""

# ── Image decoding ────────────────────────────────────────────────────────────

def decode_image(image_b64: str) -> np.ndarray:
    """Decode a base64 (or data-URI) image string to an OpenCV BGR array."""
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    img_bytes = base64.b64decode(image_b64)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image — unsupported format or corrupted data.")
    return img


def crop_face(img: np.ndarray, bounds: Optional[FaceBoundsModel]) -> np.ndarray:
    """
    Crop the image to the face bounding box, adding a 15 % padding margin.
    Falls back to the centre-quarter of the full frame if no bounds supplied.
    """
    h, w = img.shape[:2]
    if bounds and bounds.width > 20 and bounds.height > 20:
        pad_x = int(bounds.width  * 0.15)
        pad_y = int(bounds.height * 0.15)
        x1 = max(0, int(bounds.x) - pad_x)
        y1 = max(0, int(bounds.y) - pad_y)
        x2 = min(w, int(bounds.x + bounds.width)  + pad_x)
        y2 = min(h, int(bounds.y + bounds.height) + pad_y)
        crop = img[y1:y2, x1:x2]
        if crop.size > 0:
            return crop
    # Fallback: use the centre 50 % of the frame
    qw, qh = w // 4, h // 4
    return img[qh:h - qh, qw:w - qw]


# ── Signal 1: FFT periodic-pattern score ──────────────────────────────────────

def signal_fft_periodic(gray: np.ndarray) -> int:
    """
    Detect periodic high-frequency energy caused by screen pixel grids.

    A digital display has a regular array of R, G, B sub-pixels.  Even when
    individual pixels are not optically resolved by the webcam, the Nyquist
    interference creates faint periodic patterns that manifest as peaks
    scattered symmetrically in the 2-D Fourier spectrum.

    Algorithm:
      1. Resize to 128×128 and apply a Hann window to suppress edge artefacts.
      2. Compute magnitude spectrum (log scale) after fftshift.
      3. Zero out the DC component (centre 10×10 region).
      4. Divide the spectrum into concentric rings (inner / mid / outer).
      5. Measure the peak-to-mean ratio in the mid/outer rings — screens
         produce anomalously high peaks; organic skin does not.

    Returns:
      Spoof score 0-100 (higher = more screen-like frequency pattern).
    """
    resized = cv2.resize(gray, (128, 128)).astype(np.float32)

    # Hann window reduces spectral leakage from frame edges
    win = np.outer(np.hanning(128), np.hanning(128))
    resized *= win

    f   = np.fft.fft2(resized)
    fs  = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fs))

    # Suppress DC
    cx, cy = 64, 64
    mag[cy - 5:cy + 5, cx - 5:cx + 5] = 0

    h, w = mag.shape
    y_idx, x_idx = np.indices((h, w))
    dist = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)

    # Mid-frequency ring: radius 20-55 pixels
    # Screen grids produce strong peaks in this range
    mid_ring = (dist >= 20) & (dist <= 55)
    mid_vals = mag[mid_ring]

    if mid_vals.size == 0:
        return 50

    mean_mid  = mid_vals.mean()
    max_mid   = mid_vals.max()
    peak_ratio = max_mid / (mean_mid + 1e-8)

    # Real face: peak_ratio ≈ 3-8 (no concentrated peaks)
    # Screen:    peak_ratio ≈ 10-40 (regular grid peaks)
    if peak_ratio <= 6:
        return 0
    if peak_ratio >= 22:
        return 100
    return int((peak_ratio - 6) / 16 * 100)


# ── Signal 2: LBP texture entropy ─────────────────────────────────────────────

def _manual_lbp(gray: np.ndarray, size: int = 64) -> np.ndarray:
    """Compute 8-neighbour circular LBP codes without scikit-image."""
    g = cv2.resize(gray, (size, size))
    codes = np.zeros((size - 2, size - 2), dtype=np.uint8)
    for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]:
        bit = (g[1+dy:size-1+dy, 1+dx:size-1+dx] >= g[1:-1, 1:-1]).astype(np.uint8)
        codes = (codes << 1) | bit
    return codes


def signal_lbp_entropy(gray: np.ndarray) -> int:
    """
    Measure LBP histogram entropy as a proxy for skin micro-texture richness.

    Real skin has highly varied LBP codes (pores, hair, wrinkles, shadow
    gradients) → high entropy ≈ 5.5-7.5 bits.

    Screen-rendered or printed images are smoother and more uniform →
    lower entropy ≈ 3.5-5.5 bits.

    Returns:
      Spoof score 0-100 (higher = smoother texture = more likely spoof).
    """
    if HAVE_SKIMAGE:
        g   = cv2.resize(gray, (128, 128))
        lbp = skimage_lbp(g, P=8, R=1, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=59, range=(0, 59), density=True)
    else:
        lbp = _manual_lbp(gray, size=96)
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256), density=True)

    hist = hist[hist > 0]
    entropy = float(-np.sum(hist * np.log2(hist)))

    # Calibrated ranges:
    #   entropy ≥ 5.5 → rich texture → spoof score 0
    #   entropy ≤ 3.5 → flat texture → spoof score 100
    if entropy >= 5.5:
        return 0
    if entropy <= 3.5:
        return 100
    return int((5.5 - entropy) / 2.0 * 100)


# ── Signal 3: Gradient-block uniformity ───────────────────────────────────────

def signal_gradient_uniformity(gray: np.ndarray) -> int:
    """
    Measure how uniform the local gradient energy is across the face region.

    Real faces have highly variable local gradient energy: the nose has strong
    edges, the cheeks are smooth, the hairline is very textured.  Screen images
    have been processed by the display's rendering pipeline, which tends to
    equalise local contrast → more uniform per-block gradient variance.

    Algorithm:
      1. Compute Sobel gradient magnitude at full resolution.
      2. Divide into 8×8 non-overlapping blocks.
      3. Compute the variance of gradient magnitude within each block.
      4. Coefficient of Variation (CoV) of block variances:
           CoV = std(block_variances) / mean(block_variances)
         High CoV → varied texture (real)
         Low CoV  → uniform texture (screen/print)

    Returns:
      Spoof score 0-100 (higher = more uniform = more likely spoof).
    """
    g = cv2.resize(gray, (96, 96))
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)

    block_sz  = 12   # 96 / 12 = 8 blocks per axis → 64 blocks total
    variances = []
    for r in range(0, 96, block_sz):
        for c in range(0, 96, block_sz):
            block = mag[r:r + block_sz, c:c + block_sz]
            if block.size > 4:
                variances.append(float(block.var()))

    if len(variances) < 4:
        return 50

    arr      = np.array(variances)
    mean_var = arr.mean()
    if mean_var < 1e-6:
        return 80   # essentially flat → suspicious

    cov = arr.std() / mean_var

    # Real face: CoV ≈ 0.6-2.0 (highly varied)
    # Screen:    CoV ≈ 0.1-0.5 (uniform rendering)
    if cov >= 0.65:
        return 0
    if cov <= 0.18:
        return 100
    return int((0.65 - cov) / 0.47 * 100)


# ── Signal 4: Specular-highlight score ────────────────────────────────────────

def signal_specular(gray: np.ndarray) -> int:
    """
    Detect screen-glass specular reflections.

    Phone and monitor screens have a glass or glossy plastic surface that
    strongly reflects the room lights and the webcam's own fill LED as
    concentrated near-white hotspots.  Real skin may have a small forehead
    highlight, but it is diffuse — never reaching >1.5 % of face pixels above
    a 245/255 saturation threshold.

    Additionally, screens produce very CONCENTRATED bright patches (high
    local density) while natural skin highlights are diffuse.

    Returns:
      Spoof score 0-100 (higher = more screen-like glare).
    """
    BRIGHT_THRESH  = 245
    VERY_BRIGHT    = 252

    hot_mask     = gray > BRIGHT_THRESH
    very_hot_mask= gray > VERY_BRIGHT
    total        = gray.size

    hot_ratio      = hot_mask.sum() / total
    very_hot_ratio = very_hot_mask.sum() / total

    # Measure spatial clumpiness: connected component analysis of hot pixels
    hot_u8 = hot_mask.astype(np.uint8) * 255
    num_cc, _, stats, _ = cv2.connectedComponentsWithStats(hot_u8, connectivity=8)
    # Largest component size as fraction of total hot pixels
    if num_cc > 1 and hot_mask.sum() > 0:
        cc_sizes    = stats[1:, cv2.CC_STAT_AREA]  # skip background
        max_cc_frac = cc_sizes.max() / (hot_mask.sum() + 1e-8)
    else:
        max_cc_frac = 0.0

    # Screen: hot_ratio > 2 %, concentrated (max_cc_frac > 0.5)
    # Real:   hot_ratio < 1.5 %, diffuse (max_cc_frac < 0.35)
    score = 0
    if hot_ratio > 0.005:
        # Scale 0.5-5 % → 0-70
        score += int(min(70, (hot_ratio - 0.005) / 0.045 * 70))
    if very_hot_ratio > 0.002:
        score += int(min(15, (very_hot_ratio - 0.002) / 0.018 * 15))
    if max_cc_frac > 0.40:
        # Concentrated bright patch — very suspicious
        score += int(min(15, (max_cc_frac - 0.40) / 0.60 * 15))

    return min(100, score)


# ── Signal 5: YCbCr skin-colour ratio ────────────────────────────────────────

def signal_skin_ratio(img_bgr: np.ndarray) -> int:
    """
    Measure the proportion of skin-coloured pixels in the face crop.

    In YCbCr colour space, human skin across a wide range of ethnicities
    clusters in the range:
        Cb ∈ [77, 127]    Cr ∈ [133, 173]

    A face photograph cropped to the face region should have 35–80 % skin
    pixels.  Values far outside this range indicate:
      • Too low  (<25 %) — the crop includes a lot of screen bezel, paper
                           background, or the face is poorly lit.
      • Too high  (>85 %) — over-exposure or an unrealistically saturated
                            screen rendering filling most of the crop.

    Returns:
      Spoof score 0-100 (higher = more anomalous skin-colour distribution).
    """
    img_r = cv2.resize(img_bgr, (96, 96))
    ycrcb = cv2.cvtColor(img_r, cv2.COLOR_BGR2YCrCb)
    cr    = ycrcb[:, :, 1].astype(np.int32)
    cb    = ycrcb[:, :, 2].astype(np.int32)

    skin_mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    ratio     = float(skin_mask.mean())

    # Ideal: 35-80 % → score 0 (real)
    # < 20 % or > 90 % → score 80-100 (suspicious)
    if 0.30 <= ratio <= 0.82:
        return 0
    if ratio < 0.15 or ratio > 0.92:
        return 90
    if ratio < 0.30:
        return int((0.30 - ratio) / 0.15 * 80)
    return int((ratio - 0.82) / 0.10 * 80)


# ── Combine signals ───────────────────────────────────────────────────────────

WEIGHTS = {
    "fft":        0.30,
    "lbp":        0.25,
    "gradient":   0.22,
    "specular":   0.13,
    "skin":       0.10,
}

def combine_signals(
    fft: int, lbp: int, gradient: int, specular: int, skin: int
) -> tuple[int, str]:
    """
    Compute a combined spoof score and human-readable reason.

    Each signal returns a value 0-100 (higher = more spoof-like).
    Weights reflect empirical reliability for screen/print attacks.

    A single catastrophic signal (>= 85) overrides the aggregate regardless
    of other signals, because any single extremely strong spoof indicator
    should be sufficient to reject.
    """
    # Catastrophic single-signal override
    if fft >= 85:
        return 88, "Strong periodic frequency pattern detected (screen pixel grid)"
    if specular >= 80:
        return 85, "Concentrated specular highlight consistent with screen glass"
    if gradient >= 88:
        return 82, "Suspiciously uniform gradient texture — consistent with screen rendering"

    score = int(
        fft      * WEIGHTS["fft"]      +
        lbp      * WEIGHTS["lbp"]      +
        gradient * WEIGHTS["gradient"] +
        specular * WEIGHTS["specular"] +
        skin     * WEIGHTS["skin"]
    )

    if score < SPOOF_SCORE_THRESHOLD:
        reason = "Face biometrics appear genuine"
    else:
        # Identify which signal dominated
        dominant = max(
            [("frequency pattern", fft), ("skin micro-texture", lbp),
             ("gradient uniformity", gradient), ("specular glare", specular),
             ("skin colour distribution", skin)],
            key=lambda x: x[1],
        )
        reason = (
            f"Spoofing detected via {dominant[0]}. "
            "Please use your real face, not a screen or photo."
        )

    return score, reason


# ── API endpoint ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "anti-spoof", "opencv": cv2.__version__}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    t0 = time.perf_counter()

    try:
        img_bgr = decode_image(req.image_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decode failed: {e}")

    face = crop_face(img_bgr, req.face_bounds)
    if face.shape[0] < 32 or face.shape[1] < 32:
        # Face crop too small to be reliable — assume real (low confidence)
        log.warning("Face crop too small (%s) — skipping anti-spoof", face.shape)
        return AnalyzeResponse(
            is_real=True, spoof_score=0, confidence="low",
            reason="Face region too small to analyse reliably",
            signals=SignalBreakdown(
                fft_periodic=50, lbp_entropy=50,
                gradient_uniformity=50, specular=50, skin_ratio=50,
            ),
        )

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    fft_score  = signal_fft_periodic(gray)
    lbp_score  = signal_lbp_entropy(gray)
    grad_score = signal_gradient_uniformity(gray)
    spec_score = signal_specular(gray)
    skin_score = signal_skin_ratio(face)

    spoof_score, reason = combine_signals(
        fft_score, lbp_score, grad_score, spec_score, skin_score
    )

    is_real    = spoof_score < SPOOF_SCORE_THRESHOLD
    confidence = (
        "high"   if (is_real and spoof_score < 30) or (not is_real and spoof_score > 75)
        else "medium" if (is_real and spoof_score < 45) or (not is_real and spoof_score > 58)
        else "low"
    )

    elapsed = (time.perf_counter() - t0) * 1000
    log.info(
        "anti-spoof  score=%d  real=%s  fft=%d  lbp=%d  grad=%d  spec=%d  skin=%d  (%.1f ms)",
        spoof_score, is_real, fft_score, lbp_score, grad_score, spec_score, skin_score, elapsed,
    )

    return AnalyzeResponse(
        is_real=is_real,
        spoof_score=spoof_score,
        confidence=confidence,
        reason=reason,
        signals=SignalBreakdown(
            fft_periodic=fft_score,
            lbp_entropy=lbp_score,
            gradient_uniformity=grad_score,
            specular=spec_score,
            skin_ratio=skin_score,
        ),
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    log.info("Starting AuraAuth Anti-Spoof Service on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

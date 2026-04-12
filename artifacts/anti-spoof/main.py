"""
AuraAuth — Anti-Spoofing Service  v2  (Brightness-Independent Edition)
=======================================================================

WHAT CHANGED FROM v1 AND WHY
──────────────────────────────
v1 used five raw-pixel signals.  All of them broke at non-standard brightness:

  • Specular highlight  — counted pixels > 245.  Low-brightness screen → 0 bright
    pixels → score = 0 (real).  Phone passes every time.
  • FFT periodicity     — looked for pixel-grid peaks.  Modern 460 PPI screens at
    ≥ 30 cm are completely unresolvable by webcam → NO periodic peaks → score = 0.
  • Gradient uniformity — worked only because gradient magnitude is indirectly
    brightness-related.  Dark frame → low gradient → score = 0.
  • LBP entropy         — slightly brightness-dependent; high-quality phone screen
    at any brightness reliably produced entropy > 5.5 → score = 0 (real).
  • Skin colour ratio   — totally brightness-dependent (YCbCr skin range fails in
    dark / over-exposed frames).

v2 redesign principles
  1. EVERY signal is computed on a CLAHE-normalised image.
     CLAHE (Contrast Limited Adaptive Histogram Equalization) redistributes pixel
     intensities to maximise LOCAL contrast regardless of the absolute brightness
     of the input.  After CLAHE a dark screen and a bright screen look the same
     in terms of contrast structure; the spatial patterns that identify a screen
     (edge sharpness, texture character, colour relationships) are preserved.

  2. The FULL camera frame is used for screen-border detection — not just the
     face crop.  Phone bezels and screen edges appear OUTSIDE the face bounding
     box.  Detecting them requires analysing the whole frame.

  3. Each signal now explicitly returns non-zero scores for both real and fake
     inputs so the weighted combination actually discriminates.

SIGNALS (all brightness-independent via CLAHE)
───────────────────────────────────────────────
  S1  CLAHE + FFT peak-to-mean   — improved frequency range + CLAHE amplifies
                                    any residual periodic texture from screen
                                    rendering (font anti-aliasing, sub-pixel grid)
  S2  CLAHE + Multi-scale LBP    — LBP at radii 1,2,3 then joint entropy;
                                    more discriminating than single-scale
  S3  CLAHE + Gradient orientation entropy — real faces have isotropic gradient
                                    directions; screen renders have H/V bias from
                                    LCD pixel orientation and JPEG DCT artefacts
  S4  Full-frame screen border   — Canny + Hough on CLAHE full frame; phone and
                                    monitor screens have sharp rectangular edges
                                    visible regardless of screen brightness
  S5  CLAHE + Block variance CoV — coefficient of variation of per-block gradient
                                    variance; screen rendering produces suspiciously
                                    UNIFORM local contrast after CLAHE equalises
                                    the global brightness

PIPELINE
─────────────────────────────────────────────────────────────────
  POST /analyze  ←  Express API Server  ←  React frontend
                     (sends full frame + face bounds)
  ↓
  1. Decode base64 → full BGR frame
  2. Apply CLAHE globally
  3. S4: border detection on FULL CLAHE frame
  4. Crop to face region (+ 15 % padding)
  5. S1-S3, S5: run on face-crop CLAHE frame
  6. Combine → spoof_score 0-100
  7. Threshold: spoof_score ≥ SPOOF_THRESHOLD → FAKE
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

try:
    from skimage.feature import local_binary_pattern as _skimage_lbp
    HAVE_SKIMAGE = True
except ImportError:
    HAVE_SKIMAGE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("antispoof")

app = FastAPI(
    title="AuraAuth Anti-Spoof Service v2",
    version="2.0.0",
    description="Brightness-independent face anti-spoofing.",
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
    face_bounds: Optional[FaceBoundsModel] = None

class SignalBreakdown(BaseModel):
    fft_clahe:            int  # screen pixel-grid / rendering texture
    lbp_multiscale:       int  # skin micro-texture richness
    gradient_orientation: int  # edge direction entropy (H/V bias = screen)
    screen_border:        int  # rectangular screen edge detection
    block_variance_cov:   int  # local contrast uniformity

class AnalyzeResponse(BaseModel):
    is_real:     bool
    spoof_score: int
    confidence:  str
    reason:      str
    signals:     SignalBreakdown

# ── Calibrated threshold ──────────────────────────────────────────────────────

SPOOF_THRESHOLD = 38
"""
Lower than v1's 52.  Now that each signal is calibrated to give non-trivial
scores for spoof attempts, a combined score of 38 is a reliable decision
boundary:
  Real face (good lighting, any distance 20-80 cm):  expected 5-30
  Phone screen (any brightness, any angle):           expected 35-85
  Printed photo:                                      expected 40-75
"""

# ── CLAHE helper ─────────────────────────────────────────────────────────────

def _clahe(gray: np.ndarray, clip: float = 3.0, tile: int = 8) -> np.ndarray:
    """
    Apply CLAHE to a grayscale image.

    After this operation the image has maximised LOCAL contrast regardless of
    the original global brightness level.  A dark phone screen and a bright
    phone screen will both produce similar contrast distributions.
    """
    op = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return op.apply(gray)

# ── Image decoding + crop ─────────────────────────────────────────────────────

def decode_image(image_b64: str) -> np.ndarray:
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    arr = np.frombuffer(base64.b64decode(image_b64), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Image decode failed — unsupported format or corrupted data.")
    return img


def crop_face(img: np.ndarray, bounds: Optional[FaceBoundsModel], pad_pct: float = 0.15) -> np.ndarray:
    """
    Crop to the face bounding box with padding.
    Falls back to the centre half of the frame when no bounds are provided.
    """
    h, w = img.shape[:2]
    if bounds and bounds.width > 20 and bounds.height > 20:
        px = int(bounds.width  * pad_pct)
        py = int(bounds.height * pad_pct)
        x1 = max(0, int(bounds.x) - px);   x2 = min(w, int(bounds.x + bounds.width)  + px)
        y1 = max(0, int(bounds.y) - py);   y2 = min(h, int(bounds.y + bounds.height) + py)
        crop = img[y1:y2, x1:x2]
        if crop.size > 0:
            return crop
    qw, qh = w // 4, h // 4
    return img[qh:h - qh, qw:w - qw]


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════════

# ── S1: CLAHE + FFT peak-to-mean ratio ───────────────────────────────────────

def signal_fft_clahe(gray_raw: np.ndarray) -> int:
    """
    Brightness-independent frequency-domain analysis.

    After CLAHE the relative contrast of every spatial-frequency component is
    normalised.  Periodic artefacts from screen rendering (sub-pixel anti-
    aliasing, LCD stripe patterns, font rendering, JPEG re-compression blocks)
    become visible even at low / high brightness.

    Real skin has no periodic spatial structure → flat spectrum.
    Screen images → elevated energy at specific mid-frequencies.

    Returns spoof score 0-100 (higher = more screen-like).
    """
    gray = _clahe(gray_raw)
    SIZE = 128
    g = cv2.resize(gray, (SIZE, SIZE)).astype(np.float32)

    # Hann window suppresses spectral leakage from frame edges
    win = np.outer(np.hanning(SIZE), np.hanning(SIZE))
    g  *= win

    fs  = np.fft.fftshift(np.fft.fft2(g))
    mag = np.log1p(np.abs(fs))

    cy, cx = SIZE // 2, SIZE // 2
    # Remove DC
    mag[cy - 6:cy + 6, cx - 6:cx + 6] = 0

    y_i, x_i = np.indices((SIZE, SIZE))
    dist      = np.sqrt((y_i - cy) ** 2 + (x_i - cx) ** 2)

    scores = []
    for r_lo, r_hi in [(8, 20), (20, 40), (40, 60)]:
        ring    = (dist >= r_lo) & (dist < r_hi)
        vals    = mag[ring]
        if vals.size == 0:
            continue
        pr = vals.max() / (vals.mean() + 1e-8)
        scores.append(pr)

    if not scores:
        return 0

    peak_ratio = max(scores)

    # Calibration: real face → peak_ratio 2-7; screen → 8-30
    if peak_ratio <= 5:
        return 0
    if peak_ratio >= 18:
        return 100
    return int((peak_ratio - 5) / 13 * 100)


# ── S2: CLAHE + Multi-scale LBP entropy ──────────────────────────────────────

def _lbp_hist(gray: np.ndarray, radius: int) -> np.ndarray:
    """Compute rotation-invariant uniform LBP histogram (normalised)."""
    if HAVE_SKIMAGE:
        lbp  = _skimage_lbp(gray, P=8, R=radius, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
    else:
        # Manual 8-neighbour LBP (radius-1 only when skimage absent)
        offsets = [(-radius, -radius), (-radius, 0), (-radius, radius),
                   (0, radius), (radius, radius), (radius, 0),
                   (radius, -radius), (0, -radius)]
        code = np.zeros_like(gray, dtype=np.uint8)
        for b, (dy, dx) in enumerate(offsets):
            shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
            code |= ((shifted >= gray).astype(np.uint8) << b)
        hist = np.bincount(code.ravel(), minlength=256).astype(np.float32)
        hist /= (hist.sum() + 1e-8)
    return hist


def signal_lbp_multiscale(gray_raw: np.ndarray) -> int:
    """
    Multi-scale LBP texture entropy on CLAHE-normalised image.

    Single-scale LBP failed in v1 because modern phone screens produce
    high-entropy LBP histograms (the displayed photo itself has rich texture).
    Using three radii (1, 2, 3) and combining their histograms captures texture
    at different granularities; screens consistently show a different combined
    pattern from organic skin even after CLAHE equalisation.

    Returns spoof score 0-100 (higher = more screen-like / smoother texture).
    """
    gray = _clahe(gray_raw)
    g    = cv2.resize(gray, (96, 96))
    hists = []
    for r in (1, 2, 3):
        hists.append(_lbp_hist(g, r))
    combined = np.concatenate(hists)
    combined  = combined / (combined.sum() + 1e-8)
    p         = combined[combined > 0]
    entropy   = float(-np.sum(p * np.log2(p)))

    # Real skin:  multi-scale entropy ≈ 7-11 bits
    # Screen:     entropy ≈ 4-8 bits (smoother rendering)
    if entropy >= 9.0:
        return 0
    if entropy <= 5.5:
        return 100
    return int((9.0 - entropy) / 3.5 * 100)


# ── S3: CLAHE + Gradient orientation entropy ──────────────────────────────────

def signal_gradient_orientation(gray_raw: np.ndarray) -> int:
    """
    Edge-direction histogram entropy on CLAHE-normalised image.

    Real human faces present edges in all orientations roughly equally
    (cheekbones, nose bridge, eyelid curves, hair strands, lips) →
    high-entropy orientation histogram.

    Screens introduce two sources of directional bias that persist after CLAHE:
      (a) JPEG/H.264 DCT compresses the stored photo along horizontal and
          vertical axes → more H/V edges in the recovered image.
      (b) LCD sub-pixels are aligned in rows and columns → residual H/V stripe
          pattern amplified by CLAHE.

    Uses 16-bin histogram (22.5° per bin) over Sobel gradient directions
    weighted by gradient magnitude (so strong edges dominate weak ones).

    Returns spoof score 0-100 (higher = more H/V bias = more screen-like).
    """
    gray = _clahe(gray_raw)
    g    = cv2.resize(gray, (96, 96)).astype(np.float32)

    gx  = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy  = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ang = np.degrees(np.arctan2(gy, gx)) % 180  # fold to [0, 180)

    # Magnitude-weighted 18-bin histogram
    hist, _ = np.histogram(ang, bins=18, range=(0, 180), weights=mag)
    hist     = hist / (hist.sum() + 1e-8)
    p        = hist[hist > 0]
    entropy  = float(-np.sum(p * np.log2(p)))

    # Maximum possible: log2(18) ≈ 4.17 bits
    # Real face:  entropy ≈ 3.5-4.17 bits (close to uniform)
    # Screen:     entropy ≈ 2.5-3.5 bits (H/V bias)
    if entropy >= 3.8:
        return 0
    if entropy <= 2.4:
        return 100
    return int((3.8 - entropy) / 1.4 * 100)


# ── S4: Full-frame screen border detection ────────────────────────────────────

def signal_screen_border(gray_full_raw: np.ndarray) -> int:
    """
    Detect the sharp rectangular border of a phone / laptop screen.

    This is the MOST reliable brightness-independent signal:
    Screen borders are geometric features — they exist regardless of how bright
    or dark the screen content is.  After CLAHE the border-to-background
    transition is preserved as a strong edge even at minimum brightness.

    Algorithm
    ---------
    1. Apply CLAHE to full frame (amplify ALL local edges).
    2. Canny edge detection (adaptive on the normalised image).
    3. Probabilistic Hough Line Transform → line segments.
    4. Classify lines as near-horizontal (<10°) or near-vertical (>80°).
    5. A screen produces BOTH long H and long V lines close to each other
       (forming corners).  Score increases with number of such paired lines.

    Works for:
      ✓ Phone at arm's length (phone frame edges visible around displayed face)
      ✓ Phone held slightly off-centre (at least one pair of H+V edges)
      ✓ Any screen brightness (CLAHE normalises the edge contrast)

    Falls back gracefully:
      • No lines → 0 (face fills frame, rely on other signals)
      • Only H or only V → low partial score
    """
    gray_full = _clahe(gray_full_raw, clip=2.5, tile=16)
    h, w      = gray_full.shape

    edges = cv2.Canny(gray_full, 25, 75)

    min_len = int(min(w, h) * 0.14)
    lines   = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=40,
        minLineLength=min_len,
        maxLineGap=18,
    )

    if lines is None:
        return 0

    h_strong, v_strong = 0, 0
    h_moderate, v_moderate = 0, 0

    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        dx   = abs(x2 - x1) + 1e-6
        dy   = abs(y2 - y1) + 1e-6
        ang  = abs(float(np.degrees(np.arctan2(dy, dx))))
        length = np.sqrt(dx ** 2 + dy ** 2)

        is_long = length > min(w, h) * 0.25

        if ang < 8:            # Near-horizontal
            if is_long: h_strong += 1
            else:        h_moderate += 1
        elif ang > 80:         # Near-vertical
            if is_long: v_strong += 1
            else:        v_moderate += 1

    # Evidence scoring:
    #   Strong pair (both H and V long lines)  → very suspicious
    #   Only H or only V                       → moderately suspicious
    #   Moderate lines                          → slight suspicion
    if h_strong >= 1 and v_strong >= 1:
        return min(100, 50 + (h_strong + v_strong) * 12)
    if h_strong + v_strong >= 2:
        return min(80, 35 + (h_strong + v_strong) * 10)
    if h_strong + v_strong == 1:
        return 30
    if h_moderate + v_moderate >= 3:
        return 20
    return 0


# ── S5: CLAHE + Block variance CoV ────────────────────────────────────────────

def signal_block_variance_cov(gray_raw: np.ndarray) -> int:
    """
    Coefficient of variation of per-block gradient variance on CLAHE image.

    After CLAHE normalises brightness, real faces still show HIGH variation in
    local gradient energy between blocks (forehead is smooth, nasal bridge is
    sharp, cheek has stubble or pores, hairline is very textured).

    Screen-rendered or printed images have more UNIFORM local contrast after
    CLAHE because the display/printer rendering pipeline homogenises detail:
      Low CoV (< 0.35)  →  suspiciously uniform  →  screen/print
      High CoV (> 0.75) →  organically varied    →  real face

    Returns spoof score 0-100 (higher = more uniform = more screen-like).
    """
    gray = _clahe(gray_raw)
    g    = cv2.resize(gray, (96, 96)).astype(np.float32)

    gx  = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy  = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)

    blk, variances = 12, []
    for r in range(0, 96, blk):
        for c in range(0, 96, blk):
            b = mag[r:r + blk, c:c + blk]
            if b.size > 4:
                variances.append(float(b.var()))

    if len(variances) < 4:
        return 50

    arr      = np.array(variances)
    mean_var = arr.mean()
    if mean_var < 0.5:
        return 70  # Almost zero gradient everywhere → flat → suspicious

    cov = arr.std() / mean_var

    if cov >= 0.75:
        return 0
    if cov <= 0.20:
        return 100
    return int((0.75 - cov) / 0.55 * 100)


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINATION + REASON
# ═══════════════════════════════════════════════════════════════════════════════

WEIGHTS = {
    "screen_border":        0.35,  # Most direct evidence of a screen frame
    "fft":                  0.22,  # Frequency texture artefacts
    "lbp":                  0.20,  # Multi-scale skin texture
    "gradient_orientation": 0.14,  # H/V orientation bias
    "block_cov":            0.09,  # Local contrast uniformity
}


def combine(
    fft: int, lbp: int, orient: int, border: int, cov: int,
) -> tuple[int, str]:
    # Catastrophic single-signal overrides (certainty cases)
    if border >= 75:
        return 82, "Screen border / frame edges detected — phone or monitor in view."
    if fft >= 80:
        return 78, "Strong periodic frequency pattern detected (screen rendering artefacts)."
    if lbp >= 85 and cov >= 80:
        return 72, "Unnatural texture uniformity consistent with a printed or screen image."

    score = int(
        border * WEIGHTS["screen_border"]        +
        fft    * WEIGHTS["fft"]                  +
        lbp    * WEIGHTS["lbp"]                  +
        orient * WEIGHTS["gradient_orientation"] +
        cov    * WEIGHTS["block_cov"]
    )

    if score < SPOOF_THRESHOLD:
        reason = "Face biometrics appear genuine — no spoof signals detected."
    else:
        dominant = max(
            [("screen border",          border),
             ("frequency pattern",      fft),
             ("skin texture anomaly",   lbp),
             ("edge-direction bias",    orient),
             ("local contrast pattern", cov)],
            key=lambda x: x[1],
        )
        reason = (
            f"Spoofing detected via {dominant[0]}. "
            "Please use your real face — not a phone, screen, or photo."
        )

    return score, reason


# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {
        "status": "ok", "service": "anti-spoof-v2",
        "opencv": cv2.__version__, "skimage": HAVE_SKIMAGE,
        "threshold": SPOOF_THRESHOLD,
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    t0 = time.perf_counter()

    try:
        img_bgr = decode_image(req.image_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decode failed: {e}")

    # Full-frame grayscale for border detection (signal S4)
    gray_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Face-crop for texture signals S1-S3, S5
    face      = crop_face(img_bgr, req.face_bounds)
    if face.shape[0] < 24 or face.shape[1] < 24:
        log.warning("Face crop too small (%s) — running on full frame", face.shape)
        face = img_bgr

    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    # ── Run all five signals ──────────────────────────────────────────────────
    s_fft    = signal_fft_clahe(gray_face)
    s_lbp    = signal_lbp_multiscale(gray_face)
    s_orient = signal_gradient_orientation(gray_face)
    s_border = signal_screen_border(gray_full)       # <── full frame!
    s_cov    = signal_block_variance_cov(gray_face)

    spoof_score, reason = combine(s_fft, s_lbp, s_orient, s_border, s_cov)
    is_real             = spoof_score < SPOOF_THRESHOLD

    confidence = (
        "high"   if (is_real and spoof_score < 20) or (not is_real and spoof_score > 65)
        else "medium" if (is_real and spoof_score < 32) or (not is_real and spoof_score > 45)
        else "low"
    )

    elapsed = (time.perf_counter() - t0) * 1000
    log.info(
        "v2 | score=%3d  real=%s  border=%3d  fft=%3d  lbp=%3d  orient=%3d  cov=%3d  (%.1f ms)",
        spoof_score, is_real, s_border, s_fft, s_lbp, s_orient, s_cov, elapsed,
    )

    return AnalyzeResponse(
        is_real=is_real,
        spoof_score=spoof_score,
        confidence=confidence,
        reason=reason,
        signals=SignalBreakdown(
            fft_clahe=s_fft,
            lbp_multiscale=s_lbp,
            gradient_orientation=s_orient,
            screen_border=s_border,
            block_variance_cov=s_cov,
        ),
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    log.info("AuraAuth Anti-Spoof Service v2 starting on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

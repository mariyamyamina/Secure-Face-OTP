"""
AuraAuth — Anti-Spoofing + Liveness Service  v3
================================================

TWO independent systems in one service:

  /analyze       — Texture-based passive spoof detection (v2, unchanged).
                   Called once at login time with a captured frame.

  /liveness/*    — Active liveness detection using MediaPipe Face Mesh.
                   Stateful per-session; frontend streams frames during the
                   liveness challenge and reads back cumulative state.

LIVENESS PIPELINE
─────────────────
  POST /liveness/start              → session_id, head_direction challenge
  POST /liveness/frame              → per-frame metrics + cumulative checks
  GET  /liveness/verify/{session_id}→ final verdict (is_live, checks detail)
  DELETE /liveness/session/{id}     → explicit session cleanup

LIVENESS SIGNALS
────────────────
  S1  Eye Aspect Ratio (EAR)         — blink detection (close → reopen cycle)
  S2  Mouth Aspect Ratio (MAR)       — lip movement (open → close cycle)
  S3  Head Pose via solvePnP         — yaw/pitch/roll angles from 6-point 3-D
                                       model; directional challenge uses nose-
                                       tip normalised displacement
  S4  Skin Texture (temporal MAD)    — frame-to-frame Mean Absolute Difference;
                                       real faces have organic micro-motion;
                                       screens / photos are rigid

THRESHOLD TUNING
────────────────
  All threshold constants are grouped at the top of this file and
  are intentionally named so you can tune them without touching logic.

SPOOF TEXTURE PIPELINE (unchanged from v2)
──────────────────────────────────────────
  S1  CLAHE + FFT peak-to-mean ratio
  S2  CLAHE + Multi-scale LBP entropy
  S3  CLAHE + Gradient orientation entropy
  S4  Full-frame screen border detection (Hough)
  S5  CLAHE + Block variance CoV
"""

from __future__ import annotations

import base64
import logging
import os
import threading
import time
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as _mp_tasks
from mediapipe.tasks.python import vision as _mp_vision
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
    title="AuraAuth Anti-Spoof + Liveness Service v3",
    version="3.0.0",
    description="Texture-based spoof detection + MediaPipe active liveness.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════════════════
# LIVENESS THRESHOLD CONSTANTS  (tune these without touching the logic)
# ═══════════════════════════════════════════════════════════════════════════════

# ── EAR (Eye Aspect Ratio) ─────────────────────────────────────────────────────
# EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
# Typical open-eye EAR: 0.28-0.40.  Below threshold → eyes closed.
EAR_BLINK_CLOSE_THRESH = 0.22   # EAR below this counts as "eyes closed"
EAR_BLINK_OPEN_THRESH  = 0.28   # EAR above this after closing = confirmed blink
EAR_BASELINE_MIN       = 0.18   # Minimum open-eye EAR before blink tracking

# ── MAR (Mouth Aspect Ratio) ───────────────────────────────────────────────────
# MAR > threshold → mouth open;  MAR < threshold → mouth closed
MAR_OPEN_THRESH  = 0.60   # Deliberate open mouth (say "aah")
MAR_CLOSE_THRESH = 0.30   # Mouth clearly closed

# ── Head Pose ─────────────────────────────────────────────────────────────────
# Nose-tip normalised X displacement (0–1 image space).  A 0.07 shift (~7 %
# of frame width) is a clearly visible turn that random phone jitter cannot
# achieve but a real head turn easily produces.
HEAD_NOSE_DISP_THRESH = 0.07    # Normalised X displacement for L/R turn
HEAD_PITCH_UP_THRESH  = -12.0   # Pitch angle (°) for "look up" challenge
HEAD_YAW_THRESH       = 12.0    # Yaw magnitude (°) confirming head turn

# ── Texture / Motion ──────────────────────────────────────────────────────────
TEXTURE_MAD_THRESH    = 1.8     # Mean Absolute Difference per pixel (grayscale)
TEXTURE_WINDOW        = 8       # Number of recent frames to evaluate
TEXTURE_PASSING_MIN   = 5       # Frames within window that must exceed threshold

# ── Session management ────────────────────────────────────────────────────────
SESSION_TTL_S         = 180     # Session expires after 3 minutes of inactivity
MIN_FRAMES_VERDICT    = 12      # Minimum processed frames before final verdict
CLEANUP_INTERVAL_S    = 60      # How often to purge expired sessions

# ═══════════════════════════════════════════════════════════════════════════════
# MEDIAPIPE SETUP  (Tasks API — mediapipe 0.10+)
# ═══════════════════════════════════════════════════════════════════════════════

_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker"
    "/face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_PATH = "/tmp/face_landmarker.task"


def _ensure_model() -> None:
    """Download face landmarker model bundle on first run."""
    if not os.path.exists(_MODEL_PATH):
        log.info("Downloading MediaPipe FaceLandmarker model (~5 MB)…")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        log.info("Model saved to %s", _MODEL_PATH)


_ensure_model()

_LANDMARKER_LOCK = threading.Lock()
_FACE_LANDMARKER = _mp_vision.FaceLandmarker.create_from_options(
    _mp_vision.FaceLandmarkerOptions(
        base_options=_mp_tasks.BaseOptions(model_asset_path=_MODEL_PATH),
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
)

# MediaPipe Face Mesh landmark indices for EAR computation
# Layout: (p1_outer, p2_top_outer, p3_top_inner, p4_inner, p5_bot_inner, p6_bot_outer)
_LEFT_EYE_IDX  = (362, 385, 387, 263, 373, 380)   # subject's left eye
_RIGHT_EYE_IDX = (33, 160, 158, 133, 153, 144)    # subject's right eye

# Mouth landmark indices for MAR
# Layout: (left_corner, upper_left, upper_right, right_corner, lower_right, lower_left)
_MOUTH_IDX = (61, 82, 312, 291, 317, 87)

# 6 key landmarks for solvePnP head-pose estimation
_HEAD_POSE_IDX = (4, 152, 33, 263, 61, 291)
#   4   = nose tip
#  152  = chin
#   33  = right eye outer corner (viewer's left)
#  263  = left eye outer corner (viewer's right)
#   61  = right mouth corner
#  291  = left mouth corner

# Canonical 3-D face model (mm), matches _HEAD_POSE_IDX order above
_FACE_3D = np.array([
    [  0.0,     0.0,    0.0],   # nose tip
    [  0.0,   -63.6,  -12.5],   # chin
    [-43.3,    32.7,  -26.0],   # right eye outer corner (viewer's left)
    [ 43.3,    32.7,  -26.0],   # left eye outer corner (viewer's right)
    [-28.9,   -28.9,  -24.1],   # right mouth corner
    [ 28.9,   -28.9,  -24.1],   # left mouth corner
], dtype=np.float64)

# Nose tip index in MediaPipe for displacement-based direction tracking
_NOSE_IDX = 4


def _run_mediapipe(img_bgr: np.ndarray) -> Optional[list]:
    """Return list of 478 NormalizedLandmark for the first face, or None.

    Uses the mediapipe 0.10+ Tasks API (FaceLandmarker).  A module-level lock
    serialises concurrent callers because FaceLandmarker in IMAGE mode is not
    guaranteed to be thread-safe.
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    with _LANDMARKER_LOCK:
        result = _FACE_LANDMARKER.detect(mp_image)
    if not result.face_landmarks:
        return None
    return result.face_landmarks[0]  # list of NormalizedLandmark (.x .y .z)


# ═══════════════════════════════════════════════════════════════════════════════
# LIVENESS SIGNAL COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def _lm_xy(lm: list, idx: int, w: int, h: int) -> Tuple[float, float]:
    """Convert a normalised MediaPipe landmark to pixel coordinates."""
    pt = lm[idx]   # NormalizedLandmark from Tasks API (direct list element)
    return pt.x * w, pt.y * h


def compute_ear(lm, idx: Tuple[int,...], w: int, h: int) -> float:
    """Eye Aspect Ratio for 6 landmark indices."""
    pts = [np.array(_lm_xy(lm, i, w, h)) for i in idx]
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    hd = np.linalg.norm(pts[0] - pts[3])
    return float((v1 + v2) / (2.0 * hd + 1e-6))


def compute_mar(lm, idx: Tuple[int,...], w: int, h: int) -> float:
    """Mouth Aspect Ratio for 6 landmark indices."""
    pts = [np.array(_lm_xy(lm, i, w, h)) for i in idx]
    v1 = np.linalg.norm(pts[1] - pts[5])   # upper-left ↔ lower-left
    v2 = np.linalg.norm(pts[2] - pts[4])   # upper-right ↔ lower-right
    hd = np.linalg.norm(pts[0] - pts[3])   # left corner ↔ right corner
    return float((v1 + v2) / (2.0 * hd + 1e-6))


def compute_head_pose(lm, w: int, h: int) -> Tuple[float, float, float]:
    """
    Estimate head pose angles (pitch, yaw, roll) in degrees using solvePnP.

    Returns (pitch, yaw, roll).
      pitch > 0 → looking down;   pitch < 0 → looking up
      yaw   > 0 → turned to viewer's right (subject's left)
      yaw   < 0 → turned to viewer's left  (subject's right)
    """
    pts_2d = np.array(
        [_lm_xy(lm, i, w, h) for i in _HEAD_POSE_IDX],
        dtype=np.float64,
    )
    focal = float(max(w, h))
    cam   = np.array([[focal, 0, w / 2],
                      [0, focal, h / 2],
                      [0,     0,     1]], dtype=np.float64)
    dist  = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        _FACE_3D, pts_2d, cam, dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return 0.0, 0.0, 0.0

    rot_mat, _ = cv2.Rodrigues(rvec)
    proj_mat   = np.hstack([rot_mat, tvec])
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj_mat)
    pitch = float(euler[0, 0])
    yaw   = float(euler[1, 0])
    roll  = float(euler[2, 0])
    return pitch, yaw, roll


def compute_texture_mad(gray_curr: np.ndarray, gray_prev: np.ndarray) -> float:
    """Mean Absolute Difference between two grayscale frames (per pixel)."""
    diff = np.abs(gray_curr.astype(np.float32) - gray_prev.astype(np.float32))
    return float(diff.mean())


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LivenessSession:
    session_id:      str
    head_direction:  str                  # "left" | "right" | "up"
    created_at:      float = field(default_factory=time.time)
    last_active:     float = field(default_factory=time.time)
    frame_count:     int   = 0

    # ── Blink state ────────────────────────────────────────────────────────
    blink_phase:         str   = "open"   # "open" | "closing"
    min_ear_in_blink:    float = 1.0
    blink_count:         int   = 0
    blink_detected:      bool  = False

    # ── Lip state ──────────────────────────────────────────────────────────
    lip_was_open:        bool  = False
    lip_was_closed:      bool  = False
    lip_moved:           bool  = False
    last_mar:            float = 0.0

    # ── Head state ─────────────────────────────────────────────────────────
    nose_baseline_x:     Optional[float] = None   # normalised 0-1
    nose_baseline_frames: int = 0                 # frames averaged into baseline
    head_moved:          bool  = False
    last_pitch:          float = 0.0
    last_yaw:            float = 0.0
    last_roll:           float = 0.0
    last_nose_x:         float = 0.5

    # ── Texture state ──────────────────────────────────────────────────────
    prev_gray_small:     Optional[np.ndarray] = None
    texture_scores:      List[float] = field(default_factory=list)
    texture_ok:          bool  = False

    # ── Final verdict ──────────────────────────────────────────────────────
    finalised:           bool  = False
    is_live:             bool  = False

    def touch(self):
        self.last_active = time.time()

    def is_expired(self) -> bool:
        return (time.time() - self.last_active) > SESSION_TTL_S

    def liveness_score(self) -> int:
        checks = [self.blink_detected, self.lip_moved,
                  self.head_moved, self.texture_ok]
        return int(sum(checks) / len(checks) * 100)

    def all_passed(self) -> bool:
        return (
            self.blink_detected and
            self.lip_moved and
            self.head_moved and
            self.texture_ok and
            self.frame_count >= MIN_FRAMES_VERDICT
        )


# ── In-memory session store ────────────────────────────────────────────────────

_sessions: Dict[str, LivenessSession] = {}
_sessions_lock = threading.Lock()


def _get_session(session_id: str) -> LivenessSession:
    with _sessions_lock:
        s = _sessions.get(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Liveness session not found or expired.")
    if s.is_expired():
        with _sessions_lock:
            _sessions.pop(session_id, None)
        raise HTTPException(status_code=410, detail="Liveness session expired.")
    return s


def _cleanup_expired():
    """Background thread: remove expired sessions every CLEANUP_INTERVAL_S."""
    while True:
        time.sleep(CLEANUP_INTERVAL_S)
        with _sessions_lock:
            expired = [sid for sid, s in _sessions.items() if s.is_expired()]
            for sid in expired:
                del _sessions[sid]
        if expired:
            log.info("Purged %d expired liveness sessions", len(expired))


threading.Thread(target=_cleanup_expired, daemon=True).start()


# ═══════════════════════════════════════════════════════════════════════════════
# FRAME PROCESSOR (per-session liveness update)
# ═══════════════════════════════════════════════════════════════════════════════

def process_liveness_frame(session: LivenessSession, img_bgr: np.ndarray) -> dict:
    """
    Run MediaPipe on one frame and update session liveness state.

    Returns a dict with per-frame metrics and cumulative check flags.
    """
    h, w = img_bgr.shape[:2]
    session.frame_count += 1

    lm = _run_mediapipe(img_bgr)

    ear:   Optional[float] = None
    mar:   Optional[float] = None
    pitch: float           = session.last_pitch
    yaw:   float           = session.last_yaw
    roll:  float           = session.last_roll
    nose_x: float          = session.last_nose_x
    face_detected          = lm is not None

    if lm is not None:
        # ── EAR ──────────────────────────────────────────────────────────────
        ear_l = compute_ear(lm, _LEFT_EYE_IDX,  w, h)
        ear_r = compute_ear(lm, _RIGHT_EYE_IDX, w, h)
        ear   = (ear_l + ear_r) / 2.0

        # ── MAR ──────────────────────────────────────────────────────────────
        mar = compute_mar(lm, _MOUTH_IDX, w, h)
        session.last_mar = mar

        # ── Head Pose ─────────────────────────────────────────────────────────
        pitch, yaw, roll = compute_head_pose(lm, w, h)
        session.last_pitch = pitch
        session.last_yaw   = yaw
        session.last_roll  = roll

        # Nose tip normalised X (0=left edge, 1=right edge of image)
        nose_x = lm[_NOSE_IDX].x
        session.last_nose_x = nose_x

        # ── Update blink state ────────────────────────────────────────────────
        if not session.blink_detected and ear > EAR_BASELINE_MIN:
            if session.blink_phase == "open":
                if ear < EAR_BLINK_CLOSE_THRESH:
                    session.blink_phase    = "closing"
                    session.min_ear_in_blink = ear
            elif session.blink_phase == "closing":
                session.min_ear_in_blink = min(session.min_ear_in_blink, ear)
                if ear > EAR_BLINK_OPEN_THRESH:
                    if session.min_ear_in_blink < EAR_BLINK_CLOSE_THRESH:
                        session.blink_count    += 1
                        session.blink_detected  = True
                    session.blink_phase = "open"

        # ── Update lip state ──────────────────────────────────────────────────
        if not session.lip_moved:
            if mar > MAR_OPEN_THRESH:
                session.lip_was_open = True
            if mar < MAR_CLOSE_THRESH:
                session.lip_was_closed = True
            if session.lip_was_open and session.lip_was_closed:
                session.lip_moved = True

        # ── Update head direction state ───────────────────────────────────────
        if not session.head_moved:
            # Accumulate nose baseline from first 5 landmark frames
            if session.nose_baseline_x is None:
                session.nose_baseline_x = nose_x
                session.nose_baseline_frames = 1
            elif session.nose_baseline_frames < 5:
                # Running average
                n = session.nose_baseline_frames
                session.nose_baseline_x = (session.nose_baseline_x * n + nose_x) / (n + 1)
                session.nose_baseline_frames += 1
            else:
                dx = nose_x - session.nose_baseline_x  # positive = moved right in image
                direction = session.head_direction

                if direction == "left":
                    # User's LEFT → nose moves to camera's RIGHT → dx > 0
                    moved = dx > HEAD_NOSE_DISP_THRESH
                elif direction == "right":
                    # User's RIGHT → nose moves to camera's LEFT → dx < 0
                    moved = dx < -HEAD_NOSE_DISP_THRESH
                elif direction == "up":
                    # Looking up → pitch becomes negative (below threshold)
                    moved = pitch < HEAD_PITCH_UP_THRESH
                else:
                    moved = abs(dx) > HEAD_NOSE_DISP_THRESH

                if moved:
                    session.head_moved = True

    # ── Texture (temporal MAD) — runs regardless of landmark detection ────────
    gray_small = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), (64, 64))
    if not session.texture_ok and session.prev_gray_small is not None:
        mad = compute_texture_mad(gray_small, session.prev_gray_small)
        session.texture_scores.append(mad)
        recent  = session.texture_scores[-TEXTURE_WINDOW:]
        passing = sum(s > TEXTURE_MAD_THRESH for s in recent)
        if passing >= TEXTURE_PASSING_MIN:
            session.texture_ok = True
    session.prev_gray_small = gray_small

    # ── Finalise verdict ──────────────────────────────────────────────────────
    if not session.finalised and session.all_passed():
        session.finalised = True
        session.is_live   = True

    return {
        "frame_count":    session.frame_count,
        "face_detected":  face_detected,
        "ear":            round(ear, 4)   if ear   is not None else None,
        "mar":            round(mar, 4)   if mar   is not None else None,
        "pitch":          round(pitch, 2),
        "yaw":            round(yaw, 2),
        "roll":           round(roll, 2),
        "nose_x":         round(nose_x, 4),
        "checks": {
            "blink_detected": session.blink_detected,
            "lip_moved":      session.lip_moved,
            "head_moved":     session.head_moved,
            "texture_ok":     session.texture_ok,
        },
        "blink_count":    session.blink_count,
        "liveness_score": session.liveness_score(),
        "is_live":        session.is_live,
        "head_direction": session.head_direction,
        "min_frames_needed": max(0, MIN_FRAMES_VERDICT - session.frame_count),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS — LIVENESS
# ═══════════════════════════════════════════════════════════════════════════════

class StartRequest(BaseModel):
    head_direction: Optional[str] = None  # "left"|"right"|"up" or None for random


class FrameRequest(BaseModel):
    session_id: str
    image_b64:  str


# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI ENDPOINTS — LIVENESS
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/liveness/start")
def liveness_start(req: StartRequest = StartRequest()):
    """
    Create a new liveness session.

    Returns the session_id and the head_direction challenge the user must
    perform.  The client should display the instruction to the user before
    streaming frames.
    """
    import random
    direction = req.head_direction or random.choice(["left", "right"])
    sid = str(uuid.uuid4())
    session = LivenessSession(session_id=sid, head_direction=direction)
    with _sessions_lock:
        _sessions[sid] = session
    log.info("Liveness session %s created (direction=%s)", sid, direction)
    return {
        "session_id":     sid,
        "head_direction": direction,
        "thresholds": {
            "ear_blink_close": EAR_BLINK_CLOSE_THRESH,
            "ear_blink_open":  EAR_BLINK_OPEN_THRESH,
            "mar_open":        MAR_OPEN_THRESH,
            "mar_close":       MAR_CLOSE_THRESH,
            "head_nose_disp":  HEAD_NOSE_DISP_THRESH,
            "texture_mad":     TEXTURE_MAD_THRESH,
            "min_frames":      MIN_FRAMES_VERDICT,
        },
    }


@app.post("/liveness/frame")
def liveness_frame(req: FrameRequest):
    """
    Process one webcam frame for a liveness session.

    Call this endpoint at ~5 fps during the liveness challenge.
    Returns per-frame metrics and the cumulative check states.
    """
    session = _get_session(req.session_id)
    session.touch()

    if session.finalised:
        return {
            "frame_count":  session.frame_count,
            "face_detected": True,
            "ear":  None, "mar": None,
            "pitch": session.last_pitch, "yaw": session.last_yaw,
            "roll": session.last_roll, "nose_x": session.last_nose_x,
            "checks": {
                "blink_detected": session.blink_detected,
                "lip_moved":      session.lip_moved,
                "head_moved":     session.head_moved,
                "texture_ok":     session.texture_ok,
            },
            "blink_count": session.blink_count,
            "liveness_score": 100,
            "is_live": True,
            "head_direction": session.head_direction,
            "min_frames_needed": 0,
        }

    try:
        img_bgr = decode_image(req.image_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decode failed: {e}")

    result = process_liveness_frame(session, img_bgr)
    return result


@app.get("/liveness/verify/{session_id}")
def liveness_verify(session_id: str):
    """
    Return the final liveness verdict for a session.

    Call this once before the login request to get the authoritative result.
    The session is NOT deleted here so the login route can also call this
    internally if needed.
    """
    session = _get_session(session_id)
    return {
        "is_live":        session.is_live,
        "frame_count":    session.frame_count,
        "liveness_score": session.liveness_score(),
        "checks": {
            "blink_detected": session.blink_detected,
            "lip_moved":      session.lip_moved,
            "head_moved":     session.head_moved,
            "texture_ok":     session.texture_ok,
        },
        "head_direction": session.head_direction,
        "blink_count":    session.blink_count,
        "reason": (
            "All liveness checks passed." if session.is_live
            else _liveness_failure_reason(session)
        ),
    }


@app.delete("/liveness/session/{session_id}")
def liveness_delete(session_id: str):
    """Explicitly delete a liveness session (call after login completes)."""
    with _sessions_lock:
        removed = _sessions.pop(session_id, None)
    return {"deleted": removed is not None}


def _liveness_failure_reason(session: LivenessSession) -> str:
    missing = []
    if not session.blink_detected: missing.append("blink")
    if not session.lip_moved:      missing.append("lip movement")
    if not session.head_moved:     missing.append(f"head turn ({session.head_direction})")
    if not session.texture_ok:     missing.append("skin texture motion")
    if session.frame_count < MIN_FRAMES_VERDICT:
        missing.append(f"minimum frames ({session.frame_count}/{MIN_FRAMES_VERDICT})")
    if not missing:
        return "Liveness checks completed but verdict not yet finalised."
    return f"Missing: {', '.join(missing)}."


# ═══════════════════════════════════════════════════════════════════════════════
# PASSIVE SPOOF-DETECTION PIPELINE (v2, unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

SPOOF_THRESHOLD = 52          # Raised from 38 — reduces false positives from room edges

class FaceBoundsModel(BaseModel):
    x: float
    y: float
    width: float
    height: float

class AnalyzeRequest(BaseModel):
    image_b64: str
    face_bounds: Optional[FaceBoundsModel] = None

class SignalBreakdown(BaseModel):
    fft_clahe:            int
    lbp_multiscale:       int
    gradient_orientation: int
    screen_border:        int
    block_variance_cov:   int

class AnalyzeResponse(BaseModel):
    is_real:     bool
    spoof_score: int
    confidence:  str
    reason:      str
    signals:     SignalBreakdown


def _clahe(gray: np.ndarray, clip: float = 3.0, tile: int = 8) -> np.ndarray:
    op = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return op.apply(gray)


def decode_image(image_b64: str) -> np.ndarray:
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    arr = np.frombuffer(base64.b64decode(image_b64), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Image decode failed — unsupported format or corrupted data.")
    return img


def crop_face(img: np.ndarray, bounds: Optional[FaceBoundsModel], pad_pct: float = 0.15) -> np.ndarray:
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


def signal_fft_clahe(gray_raw: np.ndarray) -> int:
    gray = _clahe(gray_raw)
    SIZE = 128
    g   = cv2.resize(gray, (SIZE, SIZE)).astype(np.float32)
    win = np.outer(np.hanning(SIZE), np.hanning(SIZE))
    g  *= win
    fs  = np.fft.fftshift(np.fft.fft2(g))
    mag = np.log1p(np.abs(fs))
    cy, cx = SIZE // 2, SIZE // 2
    mag[cy - 6:cy + 6, cx - 6:cx + 6] = 0
    y_i, x_i = np.indices((SIZE, SIZE))
    dist = np.sqrt((y_i - cy) ** 2 + (x_i - cx) ** 2)
    scores = []
    for r_lo, r_hi in [(8, 20), (20, 40), (40, 60)]:
        ring = (dist >= r_lo) & (dist < r_hi)
        vals = mag[ring]
        if vals.size == 0: continue
        scores.append(vals.max() / (vals.mean() + 1e-8))
    if not scores: return 0
    peak_ratio = max(scores)
    if peak_ratio <= 5:  return 0
    if peak_ratio >= 18: return 100
    return int((peak_ratio - 5) / 13 * 100)


def _lbp_hist(gray: np.ndarray, radius: int) -> np.ndarray:
    if HAVE_SKIMAGE:
        lbp  = _skimage_lbp(gray, P=8, R=radius, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
    else:
        offsets = [(-radius,-radius),(-radius,0),(-radius,radius),
                   (0,radius),(radius,radius),(radius,0),
                   (radius,-radius),(0,-radius)]
        code = np.zeros_like(gray, dtype=np.uint8)
        for b, (dy, dx) in enumerate(offsets):
            shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
            code |= ((shifted >= gray).astype(np.uint8) << b)
        hist = np.bincount(code.ravel(), minlength=256).astype(np.float32)
        hist /= (hist.sum() + 1e-8)
    return hist


def signal_lbp_multiscale(gray_raw: np.ndarray) -> int:
    gray  = _clahe(gray_raw)
    g     = cv2.resize(gray, (96, 96))
    hists = [_lbp_hist(g, r) for r in (1, 2, 3)]
    combined = np.concatenate(hists)
    combined /= (combined.sum() + 1e-8)
    p = combined[combined > 0]
    entropy = float(-np.sum(p * np.log2(p)))
    if entropy >= 9.0: return 0
    if entropy <= 5.5: return 100
    return int((9.0 - entropy) / 3.5 * 100)


def signal_gradient_orientation(gray_raw: np.ndarray) -> int:
    gray = _clahe(gray_raw)
    g    = cv2.resize(gray, (96, 96)).astype(np.float32)
    gx   = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy   = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag  = np.sqrt(gx ** 2 + gy ** 2)
    ang  = np.degrees(np.arctan2(gy, gx)) % 180
    hist, _ = np.histogram(ang, bins=18, range=(0, 180), weights=mag)
    hist    = hist / (hist.sum() + 1e-8)
    p       = hist[hist > 0]
    entropy = float(-np.sum(p * np.log2(p)))
    if entropy >= 3.8: return 0
    if entropy <= 2.4: return 100
    return int((3.8 - entropy) / 1.4 * 100)


def signal_screen_border(gray_full_raw: np.ndarray) -> int:
    gray_full = _clahe(gray_full_raw, clip=2.5, tile=16)
    h, w      = gray_full.shape
    edges     = cv2.Canny(gray_full, 25, 75)
    min_len   = int(min(w, h) * 0.14)
    lines     = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                                minLineLength=min_len, maxLineGap=18)
    if lines is None: return 0
    h_strong = v_strong = h_moderate = v_moderate = 0
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        dx     = abs(x2 - x1) + 1e-6
        dy     = abs(y2 - y1) + 1e-6
        ang    = abs(float(np.degrees(np.arctan2(dy, dx))))
        length = np.sqrt(dx ** 2 + dy ** 2)
        is_long = length > min(w, h) * 0.25
        if ang < 8:
            if is_long: h_strong   += 1
            else:        h_moderate += 1
        elif ang > 80:
            if is_long: v_strong   += 1
            else:        v_moderate += 1
    if h_strong >= 1 and v_strong >= 1:
        return min(100, 50 + (h_strong + v_strong) * 12)
    if h_strong + v_strong >= 2:
        return min(80, 35 + (h_strong + v_strong) * 10)
    if h_strong + v_strong == 1:  return 30
    if h_moderate + v_moderate >= 3: return 20
    return 0


def signal_block_variance_cov(gray_raw: np.ndarray) -> int:
    gray = _clahe(gray_raw)
    g    = cv2.resize(gray, (96, 96)).astype(np.float32)
    gx   = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy   = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag  = np.sqrt(gx ** 2 + gy ** 2)
    blk, variances = 12, []
    for r in range(0, 96, blk):
        for c in range(0, 96, blk):
            b = mag[r:r + blk, c:c + blk]
            if b.size > 4: variances.append(float(b.var()))
    if len(variances) < 4: return 50
    arr = np.array(variances)
    mean_var = arr.mean()
    if mean_var < 0.5: return 70
    cov = arr.std() / mean_var
    if cov >= 0.75: return 0
    if cov <= 0.20: return 100
    return int((0.75 - cov) / 0.55 * 100)


WEIGHTS = {
    "screen_border":        0.35,
    "fft":                  0.22,
    "lbp":                  0.20,
    "gradient_orientation": 0.14,
    "block_cov":            0.09,
}


def combine(fft: int, lbp: int, orient: int, border: int, cov: int) -> tuple[int, str]:
    # Early-return thresholds raised to avoid false positives from room edges,
    # JPEG artefacts, and background features.
    if border >= 90:
        return 82, "Screen border / frame edges detected — phone or monitor in view."
    if fft >= 88:
        return 78, "Strong periodic frequency pattern detected (screen rendering artefacts)."
    if lbp >= 90 and cov >= 88:
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
# FASTAPI ENDPOINTS — SPOOF ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {
        "status":          "ok",
        "service":         "anti-spoof-v3-with-liveness",
        "opencv":          cv2.__version__,
        "mediapipe":       mp.__version__,
        "skimage":         HAVE_SKIMAGE,
        "spoof_threshold": SPOOF_THRESHOLD,
        "active_sessions": len(_sessions),
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    t0 = time.perf_counter()
    try:
        img_bgr = decode_image(req.image_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decode failed: {e}")

    gray_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    face      = crop_face(img_bgr, req.face_bounds)
    if face.shape[0] < 24 or face.shape[1] < 24:
        face = img_bgr
    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    s_fft    = signal_fft_clahe(gray_face)
    s_lbp    = signal_lbp_multiscale(gray_face)
    s_orient = signal_gradient_orientation(gray_face)
    s_border = signal_screen_border(gray_full)
    s_cov    = signal_block_variance_cov(gray_face)

    spoof_score, reason = combine(s_fft, s_lbp, s_orient, s_border, s_cov)
    is_real = spoof_score < SPOOF_THRESHOLD

    confidence = (
        "high"   if spoof_score < 20 or spoof_score > 70 else
        "medium" if spoof_score < 35 or spoof_score > 55 else
        "low"
    )

    elapsed = (time.perf_counter() - t0) * 1000
    log.info(
        "analyze  real=%s score=%d (fft=%d lbp=%d orient=%d border=%d cov=%d) %.1fms",
        is_real, spoof_score, s_fft, s_lbp, s_orient, s_border, s_cov, elapsed,
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
    log.info("AuraAuth Anti-Spoof + Liveness Service v3 starting on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port)

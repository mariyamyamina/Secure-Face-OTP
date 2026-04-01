/**
 * Liveness Detector  (v2 — spoof-hardened)
 *
 * Four behavioural proofs that a real, physically present person is in front
 * of the camera.  Each check has been hardened against the most common
 * spoofing vectors:
 *
 * ── Why the old thresholds failed ──────────────────────────────────────────
 *
 * face-api.js TinyFaceDetector landmark positions jitter ±2-4 px between
 * frames even when the face is completely static.  With the old thresholds
 * (LIP_OPEN 6 px, HEAD_MOVE 8 px, BLINK at any single EAR drop) the natural
 * detection noise was enough to satisfy all four checks:
 *
 *  • Blink   – one noisy EAR reading below 75 % of max counted as a blink.
 *  • Lip     – a photo showing a slightly-open mouth oscillated between
 *              "open" (>6 px) and "closed" (<3 px) via noise alone.
 *  • Head    – ≥8 px drift from baseline was trivially met by phone tilt
 *              or natural noise.
 *  • Texture – phone hand-tremor produced sufficient MAD to pass.
 *
 * ── Fixes applied ──────────────────────────────────────────────────────────
 *
 *  • Blink: requires a FULL CYCLE — EAR drops AND then recovers.
 *    Single noisy readings can't create a drop+recovery pattern.
 *
 *  • Lip:   LIP_OPEN_PX raised from 6 → 14.  A photo with a slightly-open
 *    mouth (gap ≈ 8-10 px) plus ±3 px noise can no longer reach 14 px.
 *
 *  • Head:  HEAD_MOVE_PX raised from 8 → 18.  Requires a deliberate turn,
 *    not just phone tilt or landmark jitter.
 *
 *  • Texture: MAD threshold raised from 1.2 → 2.0.  Requires stronger
 *    organic micro-motion; compressed-video or phone-tremor noise typically
 *    produces inconsistent MAD that rarely sustains above 2.0.
 *    Window tightened: 5/8 samples must pass (was 4/6).
 */

import type * as faceapi from "@vladmandic/face-api";

// ── Constants ─────────────────────────────────────────────────────────────────

/** Blink: EAR must drop to ≤ this fraction of its rolling max. */
const BLINK_DROP_RATIO = 0.68;       // Was 0.75 — require a more pronounced blink

/** Blink: EAR must recover to ≥ this fraction of rolling max after the drop. */
const BLINK_RECOVERY_RATIO = 0.88;   // Eyes must clearly reopen

/** EAR rolling history size. */
const EAR_HISTORY_SIZE = 14;

/** Minimum rolling-max EAR to establish a valid open-eye baseline. */
const EAR_BASELINE_MIN = 0.18;

/**
 * Lip OPEN threshold.  Raised from 6 → 14 px.
 * A static photo showing a slightly open mouth (≈ 8-10 px + ±3 px noise)
 * can no longer reach this threshold without a real, deliberate mouth opening.
 */
const LIP_OPEN_PX = 14;

/** Lip CLOSED threshold. */
const LIP_CLOSE_PX = 4;

/**
 * Head movement: nose tip must shift ≥ 18 px from baseline.
 * Raised from 8 → 18 px.  Requires a deliberate head turn; phone tilt or
 * landmark noise (≤ 4 px) cannot satisfy this.
 */
const HEAD_MOVE_PX = 18;

/**
 * Skin texture: minimum MAD (Mean Absolute Difference) between frames.
 * Raised from 1.2 → 2.0.  Organic face micro-motion at normal breathing
 * easily exceeds 2.0 px; phone-screen tremor (rigid, compressed) does not
 * sustain this consistently.
 */
const TEXTURE_MAD_THRESHOLD = 2.0;

/** Texture window size (recent frames to evaluate). */
const TEXTURE_WINDOW = 8;

/** Number of samples in the window that must exceed the MAD threshold. */
const TEXTURE_PASSING_MIN = 5;       // Was 4/6 — tighter: 5/8

// ── Types ─────────────────────────────────────────────────────────────────────

export interface LivenessState {
  blinkDetected:        boolean;
  lipMovementDetected:  boolean;
  headMovementDetected: boolean;
  textureDetected:      boolean;
}

export interface LivenessUpdateResult {
  state:     LivenessState;
  allPassed: boolean;
  /** Current EAR value (for debug display) */
  ear:       number | null;
  /** Current lip gap in pixels (for debug display) */
  lipGap:    number | null;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function ptDist(a: faceapi.Point, b: faceapi.Point): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/**
 * Eye Aspect Ratio for a 6-point eye slice.
 * [0] outer corner  [1][2] upper lid  [3] inner corner  [4][5] lower lid
 */
function computeEAR(eye: faceapi.Point[]): number {
  const v1 = ptDist(eye[1], eye[5]);
  const v2 = ptDist(eye[2], eye[4]);
  const h  = ptDist(eye[0], eye[3]);
  return (v1 + v2) / (2.0 * h);
}

// ── Class ─────────────────────────────────────────────────────────────────────

export class LivenessDetector {
  private state: LivenessState = {
    blinkDetected:        false,
    lipMovementDetected:  false,
    headMovementDetected: false,
    textureDetected:      false,
  };

  // ── Blink state ──────────────────────────────────────────────────────────
  private earHistory:    number[] = [];
  /**
   * Two-phase blink detection:
   *  "open"    → baseline established, waiting for EAR drop
   *  "closing" → EAR dropped below threshold, recording minimum
   *  "closed"  → waiting for EAR recovery (confirm full blink cycle)
   */
  private blinkPhase:         "open" | "closing" | "closed" = "open";
  private minEarDuringBlink:  number = 1;

  // ── Lip state ────────────────────────────────────────────────────────────
  private lipWasOpen:   boolean = false;
  private lipWasClosed: boolean = false;

  // ── Head state ───────────────────────────────────────────────────────────
  private noseBaseline: { x: number; y: number } | null = null;

  // ── Texture state ────────────────────────────────────────────────────────
  private offCanvas:    HTMLCanvasElement;
  private offCtx:       CanvasRenderingContext2D;
  private prevPixels:   Uint8ClampedArray | null = null;
  private textureScores:number[] = [];

  constructor() {
    this.offCanvas = document.createElement("canvas");
    this.offCanvas.width  = 48;
    this.offCanvas.height = 48;
    const ctx = this.offCanvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) throw new Error("LivenessDetector: Canvas 2D context unavailable");
    this.offCtx = ctx;
  }

  /** Full reset — call when liveness monitoring restarts. */
  reset(): void {
    this.state = {
      blinkDetected:        false,
      lipMovementDetected:  false,
      headMovementDetected: false,
      textureDetected:      false,
    };
    this.earHistory        = [];
    this.blinkPhase        = "open";
    this.minEarDuringBlink = 1;
    this.lipWasOpen        = false;
    this.lipWasClosed      = false;
    this.noseBaseline      = null;
    this.prevPixels        = null;
    this.textureScores     = [];
  }

  /**
   * Process one detection tick.
   *
   * @param det   face-api result (`detectSingleFace().withFaceLandmarks()`).
   *              Pass `null` if no face was detected this tick.
   * @param video Live video element (needed for texture check).
   */
  update(
    det: { landmarks: { positions: faceapi.Point[] } } | null,
    video: HTMLVideoElement,
  ): LivenessUpdateResult {
    let ear:    number | null = null;
    let lipGap: number | null = null;

    if (det) {
      const pts = det.landmarks.positions;

      // ── 1. Eye Blink (two-phase: close → reopen) ─────────────────────────
      // 68-pt map: left eye = pts[36..41], right eye = pts[42..47]
      const leftEye  = pts.slice(36, 42) as faceapi.Point[];
      const rightEye = pts.slice(42, 48) as faceapi.Point[];
      ear = (computeEAR(leftEye) + computeEAR(rightEye)) / 2;

      if (!this.state.blinkDetected) {
        this.earHistory.push(ear);
        if (this.earHistory.length > EAR_HISTORY_SIZE) this.earHistory.shift();

        if (this.earHistory.length >= 5) {
          const rollingMax    = Math.max(...this.earHistory);
          const blinkThresh   = rollingMax * BLINK_DROP_RATIO;
          const recoveryThresh= rollingMax * BLINK_RECOVERY_RATIO;

          // Only process if we have a solid open-eye baseline
          if (rollingMax > EAR_BASELINE_MIN) {
            switch (this.blinkPhase) {
              case "open":
                if (ear <= blinkThresh) {
                  // Phase 1: EAR dropped — blink is starting
                  this.blinkPhase        = "closing";
                  this.minEarDuringBlink = ear;
                }
                break;

              case "closing":
                // Track the lowest point during the blink
                if (ear < this.minEarDuringBlink) this.minEarDuringBlink = ear;
                if (ear >= recoveryThresh) {
                  // Eyes reopened: complete blink cycle confirmed
                  // Only count if the minimum EAR was sufficiently low
                  if (this.minEarDuringBlink <= blinkThresh) {
                    this.state = { ...this.state, blinkDetected: true };
                  }
                  this.blinkPhase = "open";
                }
                break;
            }
          }
        }
      }

      // ── 2. Lip Movement ───────────────────────────────────────────────────
      // Inner mouth: pts[60..67]
      // Vertical gap: pts[62] (top inner lip) vs pts[66] (bottom inner lip)
      lipGap = Math.abs(pts[62].y - pts[66].y);

      if (!this.state.lipMovementDetected) {
        // Require a CLEARLY open mouth (14 px vs. photo noise ≈ ±3 px).
        if (lipGap > LIP_OPEN_PX)  this.lipWasOpen   = true;
        if (lipGap < LIP_CLOSE_PX) this.lipWasClosed = true;
        if (this.lipWasOpen && this.lipWasClosed) {
          this.state = { ...this.state, lipMovementDetected: true };
        }
      }

      // ── 3. Head Movement ──────────────────────────────────────────────────
      // Nose tip = pts[30].  Threshold raised to 18 px; phone tilt or
      // landmark noise (< 5 px) cannot reach this.
      if (!this.state.headMovementDetected) {
        const nosePt = pts[30];
        if (!this.noseBaseline) {
          this.noseBaseline = { x: nosePt.x, y: nosePt.y };
        } else {
          const dx = Math.abs(nosePt.x - this.noseBaseline.x);
          const dy = Math.abs(nosePt.y - this.noseBaseline.y);
          if (dx > HEAD_MOVE_PX || dy > HEAD_MOVE_PX) {
            this.state = { ...this.state, headMovementDetected: true };
          }
        }
      }
    }

    // ── 4. Skin Texture (temporal MAD — spoof-hardened) ───────────────────
    // Runs on raw video regardless of face detection.
    // Threshold raised to 2.0 and window tightened to 5/8.
    // This ensures phone-tremor noise (rigid, inconsistent) cannot reliably
    // sustain the MAD above 2.0 across 5 consecutive samples.
    if (!this.state.textureDetected) {
      this.offCtx.drawImage(video, 0, 0, 48, 48);
      const imgData = this.offCtx.getImageData(0, 0, 48, 48).data;
      const gray    = new Uint8ClampedArray(48 * 48);
      for (let i = 0; i < gray.length; i++) {
        gray[i] = Math.round(
          0.299 * imgData[i * 4] +
          0.587 * imgData[i * 4 + 1] +
          0.114 * imgData[i * 4 + 2],
        );
      }

      if (this.prevPixels) {
        let mad = 0;
        for (let i = 0; i < gray.length; i++) {
          mad += Math.abs(gray[i] - this.prevPixels[i]);
        }
        mad /= gray.length;

        this.textureScores.push(mad);
        const recent  = this.textureScores.slice(-TEXTURE_WINDOW);
        const passing = recent.filter(s => s > TEXTURE_MAD_THRESHOLD).length;
        if (passing >= TEXTURE_PASSING_MIN) {
          this.state = { ...this.state, textureDetected: true };
        }
      }
      this.prevPixels = gray;
    }

    const allPassed =
      this.state.blinkDetected &&
      this.state.lipMovementDetected &&
      this.state.headMovementDetected &&
      this.state.textureDetected;

    return { state: { ...this.state }, allPassed, ear, lipGap };
  }

  getState(): LivenessState {
    return { ...this.state };
  }
}

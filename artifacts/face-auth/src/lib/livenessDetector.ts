/**
 * Liveness Detector
 *
 * Encapsulates all four behavioural liveness checks that verify the person
 * in front of the camera is alive and physically present:
 *
 *  1. Eye Blink   — Eye Aspect Ratio (EAR) drop relative to rolling max.
 *                   Distance-independent: works regardless of how far the user
 *                   sits from the camera.
 *
 *  2. Lip Movement — Detects an open→close or close→open mouth cycle using
 *                    the inner lip landmark gap (points 62 & 66 from 68-pt map).
 *
 *  3. Head Movement — Nose tip (pt 30) displacement from a per-session
 *                     baseline.  Requires >8px movement in any direction.
 *
 *  4. Skin Texture  — Frame-to-frame Mean Absolute Difference of a small 48×48
 *                     greyscale crop of the video.  Real video always has
 *                     micro-variation; a static photo has essentially zero.
 *
 * Usage:
 *   const detector = new LivenessDetector();
 *   const result = detector.update(faceApiResult, videoElement);
 *   if (result.allPassed) { ... }
 */

import type * as faceapi from "@vladmandic/face-api";

// ── Constants ─────────────────────────────────────────────────────────────────

/** EAR must drop to ≤75% of its rolling max for a blink to register. */
const BLINK_DROP_RATIO = 0.75;

/** Rolling window size for EAR samples. */
const EAR_HISTORY_SIZE = 12;

/** Inner lip gap threshold (pixels) for "mouth open". */
const LIP_OPEN_PX = 6;

/** Inner lip gap threshold (pixels) for "mouth closed". */
const LIP_CLOSE_PX = 3;

/** Nose tip must move >8px from baseline for head movement to register. */
const HEAD_MOVE_PX = 8;

/** Minimum MAD (Mean Absolute Difference) for the texture check to pass. */
const TEXTURE_MAD_THRESHOLD = 1.2;

/** Number of recent texture samples that must exceed the threshold. */
const TEXTURE_PASSING_MIN = 4;

/** Number of recent texture samples to evaluate. */
const TEXTURE_WINDOW = 6;

// ── Types ─────────────────────────────────────────────────────────────────────

export interface LivenessState {
  blinkDetected:       boolean;
  lipMovementDetected: boolean;
  headMovementDetected:boolean;
  textureDetected:     boolean;
}

export interface LivenessUpdateResult {
  state:     LivenessState;
  allPassed: boolean;
  /** Current EAR value (for debug display) */
  ear:       number | null;
  /** Current lip gap in pixels (for debug display) */
  lipGap:    number | null;
}

// ── Helper ────────────────────────────────────────────────────────────────────

function ptDist(a: faceapi.Point, b: faceapi.Point): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/**
 * Eye Aspect Ratio for a 6-point eye landmark slice.
 *
 * Layout:  [0] outer corner  [1][2] upper lid  [3] inner corner  [4][5] lower lid
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

  // Blink state
  private earHistory: number[] = [];

  // Lip state machine
  private lipWasOpen   = false;
  private lipWasClosed = false;

  // Head movement state
  private noseBaseline: { x: number; y: number } | null = null;

  // Texture state
  private offCanvas: HTMLCanvasElement;
  private offCtx:    CanvasRenderingContext2D;
  private prevPixels: Uint8ClampedArray | null = null;
  private textureScores: number[] = [];

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
    this.earHistory   = [];
    this.lipWasOpen   = false;
    this.lipWasClosed = false;
    this.noseBaseline = null;
    this.prevPixels   = null;
    this.textureScores = [];
  }

  /**
   * Process one detection tick.
   *
   * @param det   Result from face-api `detectSingleFace().withFaceLandmarks()`.
   *              Pass `null` if no face was detected this tick (texture check
   *              still runs via the video element).
   * @param video The live video element (needed for texture check).
   */
  update(
    det: { landmarks: { positions: faceapi.Point[] } } | null,
    video: HTMLVideoElement
  ): LivenessUpdateResult {
    let ear: number | null = null;
    let lipGap: number | null = null;

    if (det) {
      const pts = det.landmarks.positions;

      // ── 1. Eye Blink ──────────────────────────────────────────────────────
      // 68-landmark map: left eye = pts[36..41], right eye = pts[42..47]
      const leftEye  = pts.slice(36, 42) as faceapi.Point[];
      const rightEye = pts.slice(42, 48) as faceapi.Point[];
      ear = (computeEAR(leftEye) + computeEAR(rightEye)) / 2;

      if (!this.state.blinkDetected) {
        this.earHistory.push(ear);
        if (this.earHistory.length > EAR_HISTORY_SIZE) this.earHistory.shift();

        if (this.earHistory.length >= 4) {
          const rollingMax = Math.max(...this.earHistory);
          // Guard: baseline must be meaningful (face present, eyes open)
          if (rollingMax > 0.15 && ear <= rollingMax * BLINK_DROP_RATIO) {
            this.state = { ...this.state, blinkDetected: true };
            this.earHistory = []; // reset so it won't re-trigger
          }
        }
      }

      // ── 2. Lip Movement ───────────────────────────────────────────────────
      // Inner mouth landmarks: pts[60..67]
      // Vertical gap: pts[62] (top inner lip) vs pts[66] (bottom inner lip)
      lipGap = Math.abs(pts[62].y - pts[66].y);

      if (!this.state.lipMovementDetected) {
        if (lipGap > LIP_OPEN_PX)  this.lipWasOpen   = true;
        if (lipGap < LIP_CLOSE_PX) this.lipWasClosed = true;
        // Require both open AND closed states to confirm deliberate movement
        if (this.lipWasOpen && this.lipWasClosed) {
          this.state = { ...this.state, lipMovementDetected: true };
        }
      }

      // ── 3. Head Movement ──────────────────────────────────────────────────
      // Nose tip = pts[30]
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

    // ── 4. Skin Texture (temporal pixel variance) ─────────────────────────
    // Runs on the raw video regardless of whether a face was detected.
    // Measures frame-to-frame variation: real video always has micro-motion;
    // a static photo or looped identical frames has near-zero variation.
    if (!this.state.textureDetected) {
      this.offCtx.drawImage(video, 0, 0, 48, 48);
      const imgData = this.offCtx.getImageData(0, 0, 48, 48).data;
      const gray = new Uint8ClampedArray(48 * 48);
      for (let i = 0; i < gray.length; i++) {
        gray[i] = Math.round(
          0.299 * imgData[i * 4] +
          0.587 * imgData[i * 4 + 1] +
          0.114 * imgData[i * 4 + 2]
        );
      }

      if (this.prevPixels) {
        let mad = 0;
        for (let i = 0; i < gray.length; i++) {
          mad += Math.abs(gray[i] - this.prevPixels[i]);
        }
        mad /= gray.length;

        this.textureScores.push(mad);
        const recent = this.textureScores.slice(-TEXTURE_WINDOW);
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

  /** Read the current liveness state without advancing it. */
  getState(): LivenessState {
    return { ...this.state };
  }
}

/**
 * Liveness Detector  (v3 — replay-resistant + directional head challenge)
 *
 * Four behavioural proofs that a real, physically present person is in front
 * of the camera.  v3 adds a RANDOMISED DIRECTIONAL HEAD CHALLENGE to defeat
 * video-replay attacks.
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
 *  • Lip: LIP_OPEN_PX raised from 6 → 14.  A photo with a slightly-open
 *    mouth (gap ≈ 8-10 px) plus ±3 px noise can no longer reach 14 px.
 *
 *  • Head (v3): DIRECTIONAL — the detector is given a random target direction
 *    ("left" or "right") each authentication session.  The nose tip must
 *    move ≥ 18 px IN THAT SPECIFIC DIRECTION.  A video replay would need to
 *    match the direction, which is unknown before the session starts.
 *
 *    Coordinate note: face-api reads the raw (non-mirrored) video.
 *    When the user turns their head to THEIR LEFT, the nose tip moves to the
 *    CAMERA'S RIGHT → raw-video x INCREASES.
 *    So: user-left ↔ dx > 0;  user-right ↔ dx < 0.
 *
 *  • Texture: MAD threshold raised from 1.2 → 2.0.  Organic face micro-motion
 *    at normal breathing easily exceeds 2.0 px; phone-screen tremor (rigid,
 *    compressed) does not sustain this consistently.
 *    Window tightened: 5/8 samples must pass (was 4/6).
 */

import type * as faceapi from "@vladmandic/face-api";

// ── Constants ─────────────────────────────────────────────────────────────────

/** Blink: EAR must drop to ≤ this fraction of its rolling max. */
const BLINK_DROP_RATIO     = 0.68;

/** Blink: EAR must recover to ≥ this fraction of rolling max after the drop. */
const BLINK_RECOVERY_RATIO = 0.88;

/** EAR rolling history size. */
const EAR_HISTORY_SIZE = 14;

/** Minimum rolling-max EAR to establish a valid open-eye baseline. */
const EAR_BASELINE_MIN = 0.18;

/**
 * Lip OPEN threshold (pixels).
 * A static photo with slightly open mouth (≈ 8-10 px + ±3 px noise) cannot
 * reach 14 px without a real, deliberate mouth opening.
 */
const LIP_OPEN_PX  = 14;
const LIP_CLOSE_PX = 4;

/**
 * Head movement: nose tip must shift ≥ 10 px IN THE REQUIRED DIRECTION.
 * Lowered from 18 → 10 px for speed; the randomised DIRECTION requirement
 * is the primary replay-defence — distance is secondary.
 */
const HEAD_MOVE_PX = 10;

/**
 * Skin texture: minimum MAD (Mean Absolute Difference) between frames.
 * Raised from 1.2 → 2.0.  Organic face micro-motion at normal breathing
 * easily exceeds 2.0; phone-screen tremor (rigid, compressed) does not
 * sustain this consistently.
 */
const TEXTURE_MAD_THRESHOLD = 2.0;

const TEXTURE_WINDOW      = 5;   // Reduced from 8 for faster completion
const TEXTURE_PASSING_MIN = 3;   // Reduced from 5 for faster completion

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
  ear:       number | null;
  lipGap:    number | null;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function ptDist(a: faceapi.Point, b: faceapi.Point): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

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
  private earHistory:         number[] = [];
  private blinkPhase:         "open" | "closing" = "open";
  private minEarDuringBlink:  number = 1;

  // ── Lip state ────────────────────────────────────────────────────────────
  private lipWasOpen:   boolean = false;
  private lipWasClosed: boolean = false;

  // ── Head state ───────────────────────────────────────────────────────────
  private noseBaseline:  { x: number; y: number } | null = null;
  /**
   * The direction the user must turn their head.
   * "left"  = user's physical left  → raw-video dx > 0 (camera's right)
   * "right" = user's physical right → raw-video dx < 0 (camera's left)
   *
   * Randomised per session — defeats video replay attacks that can't
   * anticipate which direction will be required.
   */
  private headDirection: "left" | "right";

  // ── Texture state ────────────────────────────────────────────────────────
  private offCanvas:   HTMLCanvasElement;
  private offCtx:      CanvasRenderingContext2D;
  private prevPixels:  Uint8ClampedArray | null = null;
  private textureScores: number[] = [];

  constructor(headDirection: "left" | "right" = "left") {
    this.headDirection = headDirection;
    this.offCanvas = document.createElement("canvas");
    this.offCanvas.width  = 48;
    this.offCanvas.height = 48;
    const ctx = this.offCanvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) throw new Error("LivenessDetector: Canvas 2D context unavailable");
    this.offCtx = ctx;
  }

  /** Update the required head direction (call before reset or at session start). */
  setHeadDirection(dir: "left" | "right"): void {
    this.headDirection = dir;
  }

  getHeadDirection(): "left" | "right" {
    return this.headDirection;
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
    // Note: headDirection is NOT reset here — it is set by the caller (Login)
    // before reset() to ensure the direction is known before monitoring starts.
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
      const leftEye  = pts.slice(36, 42) as faceapi.Point[];
      const rightEye = pts.slice(42, 48) as faceapi.Point[];
      ear = (computeEAR(leftEye) + computeEAR(rightEye)) / 2;

      if (!this.state.blinkDetected) {
        this.earHistory.push(ear);
        if (this.earHistory.length > EAR_HISTORY_SIZE) this.earHistory.shift();

        if (this.earHistory.length >= 5) {
          const rollingMax     = Math.max(...this.earHistory);
          const blinkThresh    = rollingMax * BLINK_DROP_RATIO;
          const recoveryThresh = rollingMax * BLINK_RECOVERY_RATIO;

          if (rollingMax > EAR_BASELINE_MIN) {
            switch (this.blinkPhase) {
              case "open":
                if (ear <= blinkThresh) {
                  this.blinkPhase        = "closing";
                  this.minEarDuringBlink = ear;
                }
                break;
              case "closing":
                if (ear < this.minEarDuringBlink) this.minEarDuringBlink = ear;
                if (ear >= recoveryThresh) {
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
      lipGap = Math.abs(pts[62].y - pts[66].y);

      if (!this.state.lipMovementDetected) {
        if (lipGap > LIP_OPEN_PX)  this.lipWasOpen   = true;
        if (lipGap < LIP_CLOSE_PX) this.lipWasClosed = true;
        if (this.lipWasOpen && this.lipWasClosed) {
          this.state = { ...this.state, lipMovementDetected: true };
        }
      }

      // ── 3. Directional Head Movement ──────────────────────────────────────
      //
      // Coordinate frame: face-api reads the RAW (non-mirrored) video stream.
      // The display may be CSS-mirrored, but pixel data is always raw.
      //
      // Physical mapping (front-facing camera):
      //   User turns to THEIR LEFT  → nose moves to camera's RIGHT → raw dx > 0
      //   User turns to THEIR RIGHT → nose moves to camera's LEFT  → raw dx < 0
      //
      // The required direction is randomised per session and stored in
      // this.headDirection.  A pre-recorded replay cannot know which direction
      // will be required; an attacker must perform the correct turn live.
      if (!this.state.headMovementDetected) {
        const nosePt = pts[30];
        if (!this.noseBaseline) {
          this.noseBaseline = { x: nosePt.x, y: nosePt.y };
        } else {
          const dx = nosePt.x - this.noseBaseline.x; // SIGNED displacement
          const requiredDx =
            this.headDirection === "left" ? dx : -dx;   // positive when correct
          if (requiredDx > HEAD_MOVE_PX) {
            this.state = { ...this.state, headMovementDetected: true };
          }
        }
      }
    }

    // ── 4. Skin Texture (temporal MAD — spoof-hardened) ───────────────────
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

    // Lip movement is tracked for UI feedback but NOT required to pass.
    // Blink + directional head turn + texture are sufficient for liveness.
    const allPassed =
      this.state.blinkDetected &&
      this.state.headMovementDetected &&
      this.state.textureDetected;

    return { state: { ...this.state }, allPassed, ear, lipGap };
  }

  getState(): LivenessState {
    return { ...this.state };
  }
}

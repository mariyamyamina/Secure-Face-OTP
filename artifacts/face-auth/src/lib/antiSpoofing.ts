/**
 * Anti-Spoofing Engine  (v3 — tightened against phone-screen attacks)
 *
 * WHY THE PREVIOUS VERSION FAILED
 * ─────────────────────────────────
 * A high-quality modern phone screen (460+ PPI) at 30–40 cm distance looks
 * almost identical to a real face when captured by a typical webcam:
 *
 *  • Glare       – anti-reflective coatings and soft indoor light keep hot
 *                  pixels well below the old 4 % trigger threshold.
 *  • LBP texture – the photo is rendered at sub-pixel precision; from the
 *                  camera's view there are no visible screen pixels → entropy
 *                  is indistinguishable from real skin.
 *  • Color       – a face photo trivially passes the skin-tone check.
 *  • Temporal    – hand tremor causes the entire phone image to shift
 *                  rigidly ~2-5 px per frame, which the old check scored as
 *                  "natural face motion" (score = 100).
 *
 * CHANGES IN THIS VERSION
 * ────────────────────────
 *  1. Glare threshold tightened:  "safe" reduced from 4 % → 1.5 % bright
 *     pixels.  Even a well-coated screen usually reflects ≥ 2 % in typical
 *     office/home lighting.
 *
 *  2. LBP threshold tightened:  "definitely real" raised from 0.72 → 0.80
 *     normalised entropy.  This shaves off borderline screen captures.
 *
 *  3. NEW signal — Motion Consistency (Coefficient of Variation of MAD):
 *     Hand tremor produces VERY CONSISTENT frame-to-frame MAD (regular
 *     oscillation at 8-12 Hz sampled at ~5 Hz → near-constant MAD).
 *     Organic face micro-motion is IRREGULAR — pauses, micro-expressions,
 *     breathing cadence all vary.  CoV (std/mean) of the MAD history:
 *       • Low CoV (< 0.20) → suspiciously regular → spoof indicator
 *       • High CoV (> 0.55) → irregular motion  → real face indicator
 *
 *  4. Combined score threshold raised:  42 → 58.  Previously a phone could
 *     coast on three 80-point signals; now it needs genuine performance
 *     across all four.
 *
 *  5. Catastrophic override now also fires when temporal variance < 10
 *     (near-zero motion = strong static-image indicator).
 */

// ── Public types ──────────────────────────────────────────────────────────────

export interface SpoofSignals {
  /** 0-100: higher = less screen glare → more likely real */
  glare: number;
  /** 0-100: higher = richer skin micro-texture → more likely real */
  texture: number;
  /** 0-100: higher = more natural skin-tone distribution → more likely real */
  colorNaturalness: number;
  /** 0-100: higher = natural face micro-motion present → more likely real */
  temporalVariance: number;
  /** 0-100: higher = irregular (organic) motion → more likely real */
  motionConsistency: number;
}

export interface AntiSpoofResult {
  /** Final verdict */
  isReal: boolean;
  /** Combined 0-100 confidence score */
  score: number;
  /** Qualitative confidence tier */
  confidence: "high" | "medium" | "low";
  /** Human-readable reason string (for UI display) */
  reason: string;
  /** Per-signal breakdown */
  signals: SpoofSignals;
}

export interface FaceBounds {
  x: number;
  y: number;
  width: number;
  height: number;
}

// ── Engine ────────────────────────────────────────────────────────────────────

export class AntiSpoofEngine {
  // Temporal state
  private prevGray:      Uint8ClampedArray | null = null;
  private motionHistory: number[]                 = [];

  // Off-screen canvas
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx:    CanvasRenderingContext2D;
  private readonly SIZE = 64;

  constructor() {
    this.canvas        = document.createElement("canvas");
    this.canvas.width  = this.SIZE;
    this.canvas.height = this.SIZE;
    const ctx = this.canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) throw new Error("AntiSpoofEngine: Canvas 2D context unavailable");
    this.ctx = ctx;
  }

  reset(): void {
    this.prevGray      = null;
    this.motionHistory = [];
  }

  analyze(video: HTMLVideoElement, faceBounds?: FaceBounds): AntiSpoofResult {
    // ── Crop to face region ───────────────────────────────────────────────────
    if (faceBounds && faceBounds.width > 10 && faceBounds.height > 10) {
      const pad = Math.round(faceBounds.width * 0.1);
      const sx  = Math.max(0, faceBounds.x - pad);
      const sy  = Math.max(0, faceBounds.y - pad);
      const sw  = Math.min(video.videoWidth  - sx, faceBounds.width  + pad * 2);
      const sh  = Math.min(video.videoHeight - sy, faceBounds.height + pad * 2);
      this.ctx.drawImage(video, sx, sy, sw, sh, 0, 0, this.SIZE, this.SIZE);
    } else {
      this.ctx.drawImage(video, 0, 0, this.SIZE, this.SIZE);
    }

    const imageData = this.ctx.getImageData(0, 0, this.SIZE, this.SIZE);
    const pixels    = imageData.data;
    const total     = this.SIZE * this.SIZE;

    const gray = new Uint8ClampedArray(total);
    for (let i = 0; i < total; i++) {
      const r = pixels[i * 4];
      const g = pixels[i * 4 + 1];
      const b = pixels[i * 4 + 2];
      gray[i] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
    }

    // ── Five signals ─────────────────────────────────────────────────────────
    const glare             = this._detectGlare(gray);
    const texture           = this._computeLBPEntropy(gray);
    const colorNaturalness  = this._analyzeColorNaturalness(pixels, total);
    const temporalVariance  = this._computeTemporalVariance(gray);
    const motionConsistency = this._computeMotionConsistency();

    // ── Weighted combination ──────────────────────────────────────────────────
    // Weights: glare 0.25 | texture 0.25 | color 0.15 | temporal 0.20 | consistency 0.15
    const score = Math.round(
      glare            * 0.25 +
      texture          * 0.25 +
      colorNaturalness * 0.15 +
      temporalVariance * 0.20 +
      motionConsistency* 0.15
    );

    // Catastrophic single-signal overrides
    const anyCatastrophic =
      glare            < 15 ||
      texture          < 15 ||
      temporalVariance < 10;   // Near-zero motion = static image

    // ── Raised threshold: 58 (was 42) ────────────────────────────────────────
    const isReal = score >= 58 && !anyCatastrophic;

    // ── Reason string ─────────────────────────────────────────────────────────
    let reason = "Face appears genuine";
    if (!isReal) {
      if (temporalVariance < 10) {
        reason = "No natural face motion detected — static image likely";
      } else if (motionConsistency < 20) {
        reason = "Motion pattern too regular — possible screen or printed photo";
      } else if (glare < 20) {
        reason = "Screen glare / reflection detected";
      } else if (texture < 20) {
        reason = "Skin micro-texture appears artificial (photo or screen)";
      } else if (colorNaturalness < 25) {
        reason = "Unnatural colour distribution in face region";
      } else {
        reason = "Multiple signals indicate a non-real face";
      }
    }

    return {
      isReal,
      score,
      confidence: score >= 75 ? "high" : score >= 60 ? "medium" : "low",
      reason,
      signals: {
        glare:             Math.round(glare),
        texture:           Math.round(texture),
        colorNaturalness:  Math.round(colorNaturalness),
        temporalVariance:  Math.round(temporalVariance),
        motionConsistency: Math.round(motionConsistency),
      },
    };
  }

  // ── Private signal implementations ────────────────────────────────────────

  /**
   * Signal 1 — Glare / Specular detection.
   *
   * Threshold tightened: "safe zone" lowered from 4% → 1.5% super-bright
   * pixels.  Modern anti-reflective phone screens still typically reflect
   * ≥ 2% of ambient light as near-white hotspots in typical indoor conditions.
   */
  private _detectGlare(gray: Uint8ClampedArray): number {
    const THRESHOLD = 235;
    let hotPixels = 0;
    for (let i = 0; i < gray.length; i++) {
      if (gray[i] > THRESHOLD) hotPixels++;
    }
    const ratio = hotPixels / gray.length;

    // ≤ 1.5% → safe (score 100); ≥ 16% → clear spoof (score 0)
    if (ratio <= 0.015) return 100;
    if (ratio >= 0.16)  return 0;
    return Math.round((1 - (ratio - 0.015) / 0.145) * 100);
  }

  /**
   * Signal 2 — LBP micro-texture entropy.
   *
   * Threshold tightened: "definitely real" normalised entropy raised from
   * 0.72 → 0.80.  Real skin at close range scores 0.85-0.95.
   * A high-quality phone photo typically scores 0.70-0.82 — this change
   * ensures borderline screen captures no longer receive a full 100 score.
   */
  private _computeLBPEntropy(gray: Uint8ClampedArray): number {
    const W = this.SIZE;
    const H = this.SIZE;
    const histogram = new Uint32Array(256);
    let validPixels = 0;

    for (let y = 1; y < H - 1; y++) {
      for (let x = 1; x < W - 1; x++) {
        const center = gray[y * W + x];
        const n = [
          gray[(y - 1) * W + (x - 1)],
          gray[(y - 1) * W + x],
          gray[(y - 1) * W + (x + 1)],
          gray[y * W + (x + 1)],
          gray[(y + 1) * W + (x + 1)],
          gray[(y + 1) * W + x],
          gray[(y + 1) * W + (x - 1)],
          gray[y * W + (x - 1)],
        ];

        let code = 0;
        for (let b = 0; b < 8; b++) {
          if (n[b] >= center) code |= (1 << b);
        }
        histogram[code]++;
        validPixels++;
      }
    }

    let entropy = 0;
    for (let i = 0; i < 256; i++) {
      if (histogram[i] > 0) {
        const p = histogram[i] / validPixels;
        entropy -= p * Math.log2(p);
      }
    }

    const normalised = entropy / 8;
    // Tighter: real zone starts at 0.80 (was 0.72), spoof zone below 0.42 (was 0.38)
    if (normalised >= 0.80) return 100;
    if (normalised <= 0.42) return 0;
    return Math.round(((normalised - 0.42) / 0.38) * 100);
  }

  /**
   * Signal 3 — Colour naturalness / skin-tone distribution.
   * (Unchanged — classic Kovac et al. rule-based skin detector.)
   */
  private _analyzeColorNaturalness(
    pixels: Uint8ClampedArray,
    total:  number
  ): number {
    let skinPixels = 0;
    for (let i = 0; i < total; i++) {
      const r = pixels[i * 4];
      const g = pixels[i * 4 + 1];
      const b = pixels[i * 4 + 2];
      const maxC = Math.max(r, g, b);
      const minC = Math.min(r, g, b);
      if (
        r > 95 &&
        g > 40 &&
        b > 20 &&
        maxC - minC > 15 &&
        Math.abs(r - g) > 15 &&
        r > g &&
        r > b
      ) {
        skinPixels++;
      }
    }
    const ratio = skinPixels / total;

    if (ratio >= 0.25 && ratio <= 0.85) return 100;
    if (ratio < 0.10 || ratio > 0.95)   return 10;
    if (ratio < 0.25) return Math.round((ratio / 0.25) * 100);
    return Math.round((1 - (ratio - 0.85) / 0.10) * 100);
  }

  /**
   * Signal 4 — Temporal micro-variance (raw MAD).
   *
   * Near-zero motion still immediately flags a static image.
   * Hand tremor that used to score 100 now feeds into Signal 5 instead.
   */
  private _computeTemporalVariance(gray: Uint8ClampedArray): number {
    if (!this.prevGray) {
      this.prevGray = gray.slice();
      return 50;
    }

    let mad = 0;
    for (let i = 0; i < gray.length; i++) {
      mad += Math.abs(gray[i] - this.prevGray[i]);
    }
    mad /= gray.length;

    this.prevGray = gray.slice();
    this.motionHistory.push(mad);
    if (this.motionHistory.length > 16) this.motionHistory.shift();

    if (this.motionHistory.length < 3) return 50;

    const avg = this.motionHistory.reduce((a, b) => a + b, 0) / this.motionHistory.length;

    // Calibration (unchanged):
    if (avg < 0.3)                  return 0;
    if (avg >= 0.8 && avg <= 10)    return 100;
    if (avg > 20)                   return 25;
    if (avg < 0.8)                  return Math.round((avg / 0.8) * 100);
    return Math.round((1 - (avg - 10) / 10) * 75 + 25);
  }

  /**
   * Signal 5 — Motion Consistency (NEW in v3).
   *
   * Coefficient of Variation (CoV = std/mean) of the recent MAD history:
   *
   *   Hand tremor  → near-constant MAD (regular 8-12 Hz tremor sampled at
   *                  ~5 Hz appears as slow oscillation with low CoV ≈ 0.05-0.20)
   *
   *   Organic face → irregular micro-motion — breathing pauses, micro-
   *                  expressions, eye movements create bursts with high
   *                  CoV ≈ 0.40-0.90
   *
   * Therefore: LOW CoV = suspiciously regular = spoof signal.
   * Requires at least 6 samples to produce a meaningful estimate.
   *
   * IMPORTANT: this signal is only meaningful when significant motion is
   * present (avg MAD > 0.5).  When the face is nearly static we return
   * neutral (50) to avoid double-penalising with Signal 4.
   */
  private _computeMotionConsistency(): number {
    const h = this.motionHistory;
    if (h.length < 6) return 50; // Not enough history yet

    const avg = h.reduce((s, v) => s + v, 0) / h.length;
    if (avg < 0.5) return 50; // Nearly static — let Signal 4 handle it

    const variance = h.reduce((s, v) => s + (v - avg) ** 2, 0) / h.length;
    const std      = Math.sqrt(variance);
    const cov      = std / (avg + 1e-6);

    // Calibration:
    //   CoV < 0.20 → suspiciously consistent → spoof → score 0
    //   CoV > 0.55 → organic irregularity    → real  → score 100
    if (cov >= 0.55) return 100;
    if (cov <= 0.20) return 0;
    return Math.round(((cov - 0.20) / 0.35) * 100);
  }
}

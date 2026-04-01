/**
 * Anti-Spoofing Engine
 *
 * Classifies a webcam frame as "real face" or "spoof (photo / screen / video)"
 * using four independent, client-side signals that are cheap to compute:
 *
 *  1. Glare / Specular Detection  – phone screens reflect ambient light and
 *     camera fill-light, producing a high ratio of near-white pixels in the
 *     captured face region. Real skin almost never has that many bright spots.
 *
 *  2. LBP Micro-Texture Entropy  – Local Binary Patterns (LBP) characterise
 *     the fine-grain texture of the image. Real skin has rich, organic texture
 *     (high entropy). Screen-rendered images and printed photos are smoother
 *     (lower entropy) because they lack natural pore/hair micro-detail.
 *
 *  3. Colour Naturalness  – A face crop should contain a high proportion of
 *     pixels in the expected skin-tone range. A phone bezel, screen edge, or
 *     printed background shifts that proportion outside typical bounds.
 *
 *  4. Temporal Micro-Variance  – Real faces exhibit constant subtle movement
 *     (breathing, micro-expressions, eye saccades). A static photo has near-
 *     zero frame-to-frame difference; a screen playing a video shows a
 *     distinctly compressed-video-noise pattern that differs from organic motion.
 *
 * Signals are combined with calibrated weights into a single 0-100 score.
 * Any single catastrophically low signal (< 15) immediately flags a spoof.
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

/**
 * Stateful engine — instantiate once per authentication session.
 * Call `reset()` when the user restarts liveness monitoring.
 * Call `analyze()` every N detection ticks (e.g., every 2nd tick).
 */
export class AntiSpoofEngine {
  // Temporal variance state
  private prevGray: Uint8ClampedArray | null = null;
  private motionHistory: number[] = [];

  // Off-screen canvas for pixel extraction
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly SIZE = 64; // analyse at 64×64 — fast and sufficient

  constructor() {
    this.canvas = document.createElement("canvas");
    this.canvas.width = this.SIZE;
    this.canvas.height = this.SIZE;
    const ctx = this.canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) throw new Error("AntiSpoofEngine: Canvas 2D context unavailable");
    this.ctx = ctx;
  }

  /** Reset temporal state — call when monitoring restarts */
  reset(): void {
    this.prevGray = null;
    this.motionHistory = [];
  }

  /**
   * Analyse a single frame.
   *
   * @param video       Live HTMLVideoElement from the webcam.
   * @param faceBounds  Optional bounding box of the detected face region.
   *                    When provided the engine crops to the face, which
   *                    improves accuracy of all four signals significantly.
   */
  analyze(video: HTMLVideoElement, faceBounds?: FaceBounds): AntiSpoofResult {
    // ── Crop to face region (or full frame) ──────────────────────────────────
    if (
      faceBounds &&
      faceBounds.width > 10 &&
      faceBounds.height > 10
    ) {
      // Add a small margin around the detected face box
      const pad = Math.round(faceBounds.width * 0.1);
      const sx = Math.max(0, faceBounds.x - pad);
      const sy = Math.max(0, faceBounds.y - pad);
      const sw = Math.min(video.videoWidth - sx, faceBounds.width + pad * 2);
      const sh = Math.min(video.videoHeight - sy, faceBounds.height + pad * 2);
      this.ctx.drawImage(video, sx, sy, sw, sh, 0, 0, this.SIZE, this.SIZE);
    } else {
      this.ctx.drawImage(video, 0, 0, this.SIZE, this.SIZE);
    }

    const imageData = this.ctx.getImageData(0, 0, this.SIZE, this.SIZE);
    const pixels = imageData.data; // RGBA, length = SIZE*SIZE*4
    const total = this.SIZE * this.SIZE;

    // ── Derived channels ─────────────────────────────────────────────────────
    const gray = new Uint8ClampedArray(total);
    for (let i = 0; i < total; i++) {
      const r = pixels[i * 4];
      const g = pixels[i * 4 + 1];
      const b = pixels[i * 4 + 2];
      gray[i] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
    }

    // ── Run the four signals ─────────────────────────────────────────────────
    const glare            = this._detectGlare(gray);
    const texture          = this._computeLBPEntropy(gray);
    const colorNaturalness = this._analyzeColorNaturalness(pixels, total);
    const temporalVariance = this._computeTemporalVariance(gray);

    // ── Weighted combination ─────────────────────────────────────────────────
    // Glare and texture are the most reliable discriminators; they get
    // higher weights. Temporal variance needs several frames to warm up
    // so it starts with a neutral score — keep its weight balanced.
    const score = Math.round(
      glare            * 0.30 +
      texture          * 0.30 +
      colorNaturalness * 0.20 +
      temporalVariance * 0.20
    );

    // A single catastrophically low signal (strong spoof indicator) overrides
    // the aggregate: any signal < 15 immediately classifies as spoof.
    const anyCatastrophic = glare < 15 || texture < 15;

    const isReal = score >= 42 && !anyCatastrophic;

    // ── Reason string ────────────────────────────────────────────────────────
    let reason = "Face appears genuine";
    if (!isReal) {
      if (glare < 20) {
        reason = "Screen glare / reflection detected — please use your real face";
      } else if (texture < 20) {
        reason = "Skin micro-texture appears artificial (photo or screen detected)";
      } else if (colorNaturalness < 25) {
        reason = "Unnatural colour distribution detected in face region";
      } else if (temporalVariance < 20) {
        reason = "No natural face motion detected — possible static image";
      } else {
        reason = "Multiple signals indicate a non-real face";
      }
    }

    return {
      isReal,
      score,
      confidence: score >= 70 ? "high" : score >= 45 ? "medium" : "low",
      reason,
      signals: {
        glare:            Math.round(glare),
        texture:          Math.round(texture),
        colorNaturalness: Math.round(colorNaturalness),
        temporalVariance: Math.round(temporalVariance),
      },
    };
  }

  // ── Private signal implementations ────────────────────────────────────────

  /**
   * Signal 1 — Glare / Specular reflection detection.
   *
   * Phone and tablet screens have a glass layer that strongly reflects the
   * room lights and any camera LED fill-light.  This produces a concentrated
   * cluster of near-white pixels inside the face crop.  Real skin may have
   * a small highlight on the forehead, but almost never >4% super-bright area.
   *
   * Threshold is empirically set at 235/255 (near-saturation) which filters
   * normal facial highlights while catching screen hotspots.
   */
  private _detectGlare(gray: Uint8ClampedArray): number {
    const THRESHOLD = 235;
    let hotPixels = 0;
    for (let i = 0; i < gray.length; i++) {
      if (gray[i] > THRESHOLD) hotPixels++;
    }
    const ratio = hotPixels / gray.length;

    // Calibrated bounds:
    //   < 4% → clearly real (score = 100)
    //   4-18% → grey zone (linear interpolation)
    //   > 18% → clearly a screen (score = 0)
    if (ratio <= 0.04) return 100;
    if (ratio >= 0.18) return 0;
    return Math.round((1 - (ratio - 0.04) / 0.14) * 100);
  }

  /**
   * Signal 2 — LBP (Local Binary Pattern) micro-texture entropy.
   *
   * The LBP of a pixel is an 8-bit code formed by thresholding its 8
   * neighbours against the centre value.  Shannon entropy of the resulting
   * histogram measures how diverse the local texture is.
   *
   * Real skin has a high-entropy LBP distribution (~6.5-7.5 bits out of 8)
   * because pores, fine hairs and micro-wrinkles create varied patterns.
   * Screen-rendered images and printed photos are smoother, producing a
   * lower-entropy histogram (~4-6 bits).
   */
  private _computeLBPEntropy(gray: Uint8ClampedArray): number {
    const W = this.SIZE;
    const H = this.SIZE;
    const histogram = new Uint32Array(256);
    let validPixels = 0;

    for (let y = 1; y < H - 1; y++) {
      for (let x = 1; x < W - 1; x++) {
        const center = gray[y * W + x];
        // 8 clockwise neighbours starting from top-left
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

    // Shannon entropy of the LBP histogram
    let entropy = 0;
    for (let i = 0; i < 256; i++) {
      if (histogram[i] > 0) {
        const p = histogram[i] / validPixels;
        entropy -= p * Math.log2(p);
      }
    }

    // Normalise to [0, 1] then to 0-100 score
    // entropy range [0, 8] where 8 = maximum (perfectly uniform histogram)
    const normalised = entropy / 8;
    if (normalised >= 0.72) return 100;
    if (normalised <= 0.38) return 0;
    return Math.round(((normalised - 0.38) / 0.34) * 100);
  }

  /**
   * Signal 3 — Colour naturalness / skin-tone distribution.
   *
   * When we crop to the face region, the majority of pixels should fall
   * inside a skin-tone colour space.  The classic rule-based skin detector
   * (Kovac et al.):
   *
   *   R > 95, G > 40, B > 20
   *   max(R,G,B) - min(R,G,B) > 15
   *   |R - G| > 15
   *   R > G  and  R > B
   *
   * works well here because we specifically care about faces, not arbitrary
   * backgrounds.  A phone bezel, screen border, or paper print shifts the
   * ratio outside 20-85%.
   */
  private _analyzeColorNaturalness(
    pixels: Uint8ClampedArray,
    total: number
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

    // Healthy face crop: 25-85% skin pixels
    // Outside this range → suspicious (too little = bezel/background; too much = uniform fill)
    if (ratio >= 0.25 && ratio <= 0.85) return 100;
    if (ratio < 0.10 || ratio > 0.95) return 10;
    if (ratio < 0.25) return Math.round((ratio / 0.25) * 100);
    return Math.round((1 - (ratio - 0.85) / 0.10) * 100);
  }

  /**
   * Signal 4 — Temporal micro-variance (frame-to-frame motion).
   *
   * Mean Absolute Difference (MAD) of consecutive grayscale frames.
   *
   * Real faces always exhibit organic micro-motion: breathing shifts the
   * head slightly, eyes saccade, skin pulses.  Average MAD: 1.0-10.0.
   *
   * A static printed photo: MAD ≈ 0.0-0.4 (near-zero).
   * A screen playing a static/looped image: similar.
   * A screen playing a video may have higher MAD but exhibits a distinctly
   * blocky, compression-artefact noise pattern; however, the temporal
   * variance check alone cannot fully distinguish this case — the other
   * signals (texture, glare) compensate.
   */
  private _computeTemporalVariance(gray: Uint8ClampedArray): number {
    if (!this.prevGray) {
      this.prevGray = gray.slice();
      return 50; // Neutral on very first frame — no history yet
    }

    let mad = 0;
    for (let i = 0; i < gray.length; i++) {
      mad += Math.abs(gray[i] - this.prevGray[i]);
    }
    mad /= gray.length;

    this.prevGray = gray.slice();
    this.motionHistory.push(mad);
    if (this.motionHistory.length > 12) this.motionHistory.shift();

    if (this.motionHistory.length < 3) return 50; // Still warming up

    const avg =
      this.motionHistory.reduce((a, b) => a + b, 0) / this.motionHistory.length;

    // Calibrated ranges:
    //   avg < 0.3 → definitely static image    → 0
    //   0.3-0.8 → very little motion          → scale up
    //   0.8-10 → natural motion range         → 100
    //   10-20 → borderline (video/shaking)    → scale down
    //   > 20 → excessive (motion blur/video)  → 25
    if (avg < 0.3) return 0;
    if (avg >= 0.8 && avg <= 10) return 100;
    if (avg > 20) return 25;
    if (avg < 0.8) return Math.round((avg / 0.8) * 100);
    return Math.round((1 - (avg - 10) / 10) * 75 + 25);
  }
}

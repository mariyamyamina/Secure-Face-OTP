/**
 * ServerLivenessClient
 *
 * Thin client that manages a MediaPipe liveness session hosted on the
 * Python anti-spoof service (proxied through Express).
 *
 * Flow:
 *   1. Call start()         → creates session, returns head_direction challenge
 *   2. Call sendFrame()     → sends one base64 frame; returns per-frame state
 *   3. Call verify()        → gets final verdict before login
 *   4. Call deleteSession() → cleanup after login (auto-called by verify)
 *
 * Usage:
 *   const client = new ServerLivenessClient();
 *   const { sessionId, headDirection } = await client.start();
 *   // ... on every Nth detection tick:
 *   const state = await client.sendFrame(base64Image);
 *   // ... on login:
 *   const verdict = await client.verify();
 */

// ── Types ─────────────────────────────────────────────────────────────────────

export interface ServerLivenessChecks {
  blink_detected: boolean;
  lip_moved:      boolean;
  head_moved:     boolean;
  texture_ok:     boolean;
}

export interface ServerLivenessFrameResult {
  frame_count:       number;
  face_detected:     boolean;
  ear:               number | null;
  mar:               number | null;
  pitch:             number;
  yaw:               number;
  roll:              number;
  nose_x:            number;
  checks:            ServerLivenessChecks;
  blink_count:       number;
  liveness_score:    number;
  is_live:           boolean;
  head_direction:    string;
  min_frames_needed: number;
}

export interface ServerLivenessVerifyResult {
  is_live:        boolean;
  frame_count:    number;
  liveness_score: number;
  checks:         ServerLivenessChecks;
  head_direction: string;
  blink_count:    number;
  reason:         string;
}

export interface ServerLivenessStartResult {
  sessionId:     string;
  headDirection: "left" | "right" | "up";
  thresholds: {
    ear_blink_close: number;
    ear_blink_open:  number;
    mar_open:        number;
    mar_close:       number;
    head_nose_disp:  number;
    texture_mad:     number;
    min_frames:      number;
  };
}

// ── Client ─────────────────────────────────────────────────────────────────────

const LIVENESS_API = "";  // empty = relative to current origin (goes through Express)

export class ServerLivenessClient {
  private sessionId:     string | null = null;
  private headDirection: "left" | "right" | "up" = "left";
  private _sendInFlight  = false;
  private _lastFrameResult: ServerLivenessFrameResult | null = null;

  /** Create a new session on the server. Must be called before sendFrame. */
  async start(preferredDirection?: "left" | "right" | "up"): Promise<ServerLivenessStartResult> {
    const res = await fetch(`${LIVENESS_API}/api/liveness/start`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ head_direction: preferredDirection ?? null }),
    });

    if (!res.ok) {
      const body = await res.text();
      throw new Error(`Liveness session start failed (${res.status}): ${body}`);
    }

    const data = await res.json() as {
      session_id: string;
      head_direction: string;
      thresholds: ServerLivenessStartResult["thresholds"];
    };

    this.sessionId     = data.session_id;
    this.headDirection = data.head_direction as "left" | "right" | "up";

    return {
      sessionId:     data.session_id,
      headDirection: this.headDirection,
      thresholds:    data.thresholds,
    };
  }

  /**
   * Send one frame to the server.  Returns null if:
   *  - No session has been started, OR
   *  - A previous frame send is still in-flight (throttle)
   *
   * The base64 string can be a data-URI ("data:image/...;base64,...")
   * or a raw base64 string.
   */
  async sendFrame(imageB64: string): Promise<ServerLivenessFrameResult | null> {
    if (!this.sessionId || this._sendInFlight) return null;

    this._sendInFlight = true;
    try {
      const res = await fetch(`${LIVENESS_API}/api/liveness/frame`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ session_id: this.sessionId, image_b64: imageB64 }),
      });

      if (!res.ok) {
        if (res.status === 404 || res.status === 410) {
          // Session expired or not found
          this.sessionId = null;
        }
        return null;
      }

      const data = await res.json() as ServerLivenessFrameResult;
      this._lastFrameResult = data;
      return data;
    } catch {
      return null;
    } finally {
      this._sendInFlight = false;
    }
  }

  /** Get the final liveness verdict from the server. */
  async verify(): Promise<ServerLivenessVerifyResult | null> {
    if (!this.sessionId) return null;
    try {
      const res = await fetch(`${LIVENESS_API}/api/liveness/verify/${this.sessionId}`);
      if (!res.ok) return null;
      return await res.json() as ServerLivenessVerifyResult;
    } catch {
      return null;
    }
  }

  /** Delete the session (call after login completes). */
  async deleteSession(): Promise<void> {
    if (!this.sessionId) return;
    const sid = this.sessionId;
    this.sessionId = null;
    fetch(`${LIVENESS_API}/api/liveness/session/${sid}`, { method: "DELETE" }).catch(() => {});
  }

  get currentSessionId(): string | null { return this.sessionId; }
  get currentHeadDirection(): "left" | "right" | "up" { return this.headDirection; }
  get lastFrameResult(): ServerLivenessFrameResult | null { return this._lastFrameResult; }
  get hasSession(): boolean { return this.sessionId !== null; }

  /** Reset client state (call on session retry). */
  reset(): void {
    if (this.sessionId) {
      fetch(`${LIVENESS_API}/api/liveness/session/${this.sessionId}`, { method: "DELETE" }).catch(() => {});
    }
    this.sessionId         = null;
    this._sendInFlight     = false;
    this._lastFrameResult  = null;
    this.headDirection     = "left";
  }
}

import { Router, type IRouter } from "express";
import bcrypt from "bcrypt";
import { db, usersTable } from "@workspace/db";
import { RegisterFaceBody, LoginFaceBody } from "@workspace/api-zod";
import { eq } from "drizzle-orm";

const router: IRouter = Router();

const SALT_ROUNDS = 10;

/** Euclidean distance between two face descriptor vectors. */
function euclideanDistance(a: number[], b: number[]): number {
  return Math.sqrt(a.reduce((sum, val, i) => sum + (val - b[i]) ** 2, 0));
}

// ── Anti-Spoof Service Integration ────────────────────────────────────────────

const ANTI_SPOOF_URL = process.env.ANTI_SPOOF_URL ?? "http://localhost:8000";

/**
 * Call the Python anti-spoofing service.
 *
 * Returns the parsed JSON response, or null if the service is unreachable.
 * A null response is treated as SPOOF REJECTED to enforce a strict security
 * posture — the Python service must be running for logins to proceed.
 */
async function checkAntiSpoof(
  imageB64: string,
  faceBounds?: { x: number; y: number; width: number; height: number },
): Promise<{ is_real: boolean; spoof_score: number; reason: string; signals: Record<string, number> } | null> {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 5000); // 5 s timeout

    const res = await fetch(`${ANTI_SPOOF_URL}/analyze`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ image_b64: imageB64, face_bounds: faceBounds ?? null }),
      signal:  controller.signal,
    });
    clearTimeout(timer);

    if (!res.ok) {
      const body = await res.text();
      console.error(`[anti-spoof] Service returned ${res.status}: ${body}`);
      return null;
    }
    return await res.json() as { is_real: boolean; spoof_score: number; reason: string; signals: Record<string, number> };
  } catch (err: unknown) {
    if (err instanceof Error && err.name === "AbortError") {
      console.error("[anti-spoof] Service timed out after 5 s");
    } else {
      console.error("[anti-spoof] Service unreachable:", err instanceof Error ? err.message : err);
    }
    return null;
  }
}

// ── POST /api/register-face ───────────────────────────────────────────────────

router.post("/register-face", async (req, res) => {
  const parsed = RegisterFaceBody.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ error: "Invalid input", details: parsed.error.message });
    return;
  }

  const { email, password, face_descriptor } = parsed.data;
  const name = typeof req.body.name === "string" ? req.body.name.trim() : null;

  try {
    const existing = await db
      .select({ id: usersTable.id })
      .from(usersTable)
      .where(eq(usersTable.email, email))
      .limit(1);

    if (existing.length > 0) {
      res.status(409).json({ error: "Email already registered" });
      return;
    }

    const hashedPassword = await bcrypt.hash(password, SALT_ROUNDS);
    const descriptorJson = JSON.stringify(face_descriptor);

    const [newUser] = await db
      .insert(usersTable)
      .values({ name, email, password: hashedPassword, face_descriptor: descriptorJson })
      .returning({ id: usersTable.id });

    res.status(201).json({
      message: "Registration successful. Please wait for admin approval before logging in.",
      userId: newUser.id,
    });
  } catch (err) {
    console.error("Register error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

// ── Liveness session verifier (calls Python service) ─────────────────────────

async function verifyLivenessSession(
  sessionId: string,
): Promise<{ is_live: boolean; liveness_score: number; reason: string; checks: Record<string, boolean> } | null> {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 5_000);
    const res = await fetch(`${ANTI_SPOOF_URL}/liveness/verify/${sessionId}`, {
      signal: controller.signal,
    });
    clearTimeout(timer);
    if (!res.ok) return null;
    return await res.json() as { is_live: boolean; liveness_score: number; reason: string; checks: Record<string, boolean> };
  } catch {
    return null;
  }
}

// ── POST /api/login-face ──────────────────────────────────────────────────────

router.post("/login-face", async (req, res) => {
  const parsed = LoginFaceBody.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ error: "Invalid input", details: parsed.error.message });
    return;
  }

  const { email, face_descriptor, liveness_passed } = parsed.data;

  // ── Gate 1a: Client-side liveness (legacy flag) ───────────────────────────
  if (!liveness_passed) {
    res.status(401).json({ error: "Liveness verification failed" });
    return;
  }

  // ── Gate 1b: Server-side MediaPipe liveness session (if provided) ─────────
  const livenessSessionId: string | null =
    typeof req.body.liveness_session_id === "string" && req.body.liveness_session_id.length > 10
      ? req.body.liveness_session_id
      : null;

  if (livenessSessionId) {
    const livenessResult = await verifyLivenessSession(livenessSessionId);
    if (livenessResult === null) {
      console.error(`[liveness] Session verify failed for ${email} — service unreachable`);
      res.status(503).json({ error: "Liveness verification service unavailable. Please try again." });
      return;
    }
    if (!livenessResult.is_live) {
      console.warn(
        `[liveness] Session ${livenessSessionId} rejected for ${email}: score=${livenessResult.liveness_score} reason="${livenessResult.reason}"`,
      );
      res.status(401).json({
        error:           "Server-side liveness verification failed. Please complete all checks and try again.",
        liveness_score:  livenessResult.liveness_score,
        reason:          livenessResult.reason,
        checks:          livenessResult.checks,
      });
      return;
    }
    console.log(`[liveness] Session ${livenessSessionId} verified for ${email}: score=${livenessResult.liveness_score}`);

    // Clean up session asynchronously (don't block login)
    fetch(`${ANTI_SPOOF_URL}/liveness/session/${livenessSessionId}`, { method: "DELETE" }).catch(() => {});
  }

  // ── Server-side anti-spoofing (second gate, mandatory) ───────────────────
  // The client must include a base64-encoded webcam frame for server-side
  // analysis.  Without it the request is rejected — this prevents clients
  // that have been tampered with from bypassing the anti-spoof layer.
  const faceImageB64: string | null =
    typeof req.body.face_image_b64 === "string" && req.body.face_image_b64.length > 100
      ? req.body.face_image_b64
      : null;

  const faceBounds =
    req.body.face_bounds &&
    typeof req.body.face_bounds.x === "number"
      ? req.body.face_bounds as { x: number; y: number; width: number; height: number }
      : undefined;

  if (!faceImageB64) {
    console.warn(`[anti-spoof] No face image provided by ${email} — rejecting`);
    res.status(400).json({
      error:  "Face image required for server-side anti-spoofing verification.",
      detail: "The client must include face_image_b64 in the request.",
    });
    return;
  }

  const spoofResult = await checkAntiSpoof(faceImageB64, faceBounds);

  if (spoofResult === null) {
    // Service unreachable — strict policy: reject login
    console.error(`[anti-spoof] Service unavailable — blocking login for ${email}`);
    res.status(503).json({
      error: "Anti-spoofing service temporarily unavailable. Please try again in a moment.",
    });
    return;
  }

  console.log(
    `[anti-spoof] user=${email}  real=${spoofResult.is_real}  score=${spoofResult.spoof_score}` +
    `  signals=${JSON.stringify(spoofResult.signals)}`,
  );

  if (!spoofResult.is_real) {
    res.status(403).json({
      error:       "Spoofing attempt detected (mobile screen / photo / video replay). Access denied.",
      spoof_score: spoofResult.spoof_score,
      reason:      spoofResult.reason,
      signals:     spoofResult.signals,
    });
    return;
  }

  // ── Face recognition (third gate) ────────────────────────────────────────
  try {
    const users = await db
      .select()
      .from(usersTable)
      .where(eq(usersTable.email, email))
      .limit(1);

    if (users.length === 0) {
      res.status(404).json({ error: "User not found. Please register first." });
      return;
    }

    const user = users[0];

    if (!user.is_approved) {
      res.status(403).json({
        error: "Account pending approval. Please wait for an admin to approve your account.",
      });
      return;
    }

    const storedDescriptor: number[] = JSON.parse(user.face_descriptor);

    if (face_descriptor.length !== storedDescriptor.length) {
      res.status(400).json({ error: "Invalid face descriptor format" });
      return;
    }

    const distance = euclideanDistance(face_descriptor, storedDescriptor);
    const THRESHOLD = 0.6;

    console.log(
      `[face-match] user=${email} distance=${distance.toFixed(4)} threshold=${THRESHOLD} matched=${distance <= THRESHOLD}`,
    );

    if (distance > THRESHOLD) {
      res.status(401).json({
        error:    "Face does not match. Please try again.",
        distance: parseFloat(distance.toFixed(4)),
      });
      return;
    }

    const confidence = Math.max(0, Math.round((1 - distance / THRESHOLD) * 100));

    res.status(200).json({
      message:      "Face verified. OTP will be sent to your email.",
      face_matched: true,
      confidence,
      name:         user.name ?? null,
    });
  } catch (err) {
    console.error("Login error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

// ── POST /api/login ───────────────────────────────────────────────────────────

router.post("/login", async (req, res) => {
  const { email, password } = req.body ?? {};
  if (!email || typeof email !== "string" || !password || typeof password !== "string") {
    res.status(400).json({ error: "Invalid input" });
    return;
  }

  try {
    const users = await db
      .select()
      .from(usersTable)
      .where(eq(usersTable.email, email))
      .limit(1);

    if (users.length === 0) {
      res.status(401).json({ error: "Invalid email or password." });
      return;
    }

    const user = users[0];

    if (!user.is_approved) {
      res.status(403).json({
        error: "Account pending approval. Please wait for an admin to approve your account.",
      });
      return;
    }

    const passwordMatch = await bcrypt.compare(password, user.password);
    if (!passwordMatch) {
      res.status(401).json({ error: "Invalid email or password." });
      return;
    }

    res.status(200).json({
      message: "Login successful.",
      email:   user.email,
      name:    user.name ?? null,
    });
  } catch (err) {
    console.error("Password login error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

export default router;

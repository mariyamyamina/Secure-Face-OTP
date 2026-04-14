/**
 * Liveness Proxy Routes
 *
 * Proxies liveness session requests from the frontend to the Python
 * MediaPipe liveness service.  The browser cannot call the Python service
 * directly in production, so all liveness traffic goes through here.
 *
 * Routes:
 *   POST   /api/liveness/start           → Python /liveness/start
 *   POST   /api/liveness/frame           → Python /liveness/frame
 *   GET    /api/liveness/verify/:id      → Python /liveness/verify/{id}
 *   DELETE /api/liveness/session/:id     → Python /liveness/session/{id}
 */

import { Router, type IRouter } from "express";

const router: IRouter = Router();

const ANTI_SPOOF_URL = process.env.ANTI_SPOOF_URL ?? "http://localhost:8000";

/** Generic proxy helper — forwards body as JSON, returns Python response. */
async function proxyToPython(
  path: string,
  method: "GET" | "POST" | "DELETE",
  body?: unknown,
  timeoutMs = 10_000,
): Promise<{ status: number; data: unknown }> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(`${ANTI_SPOOF_URL}${path}`, {
      method,
      headers: body ? { "Content-Type": "application/json" } : {},
      body:   body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });
    clearTimeout(timer);
    const data = await res.json();
    return { status: res.status, data };
  } catch (err: unknown) {
    clearTimeout(timer);
    const msg = err instanceof Error ? err.message : String(err);
    throw new Error(`Python service unreachable: ${msg}`);
  }
}

// ── POST /api/liveness/start ─────────────────────────────────────────────────

router.post("/liveness/start", async (req, res) => {
  try {
    const { status, data } = await proxyToPython(
      "/liveness/start", "POST",
      { head_direction: req.body?.head_direction ?? null },
    );
    res.status(status).json(data);
  } catch (err) {
    console.error("[liveness/start]", err instanceof Error ? err.message : err);
    res.status(503).json({ error: "Liveness service unavailable." });
  }
});

// ── POST /api/liveness/frame ─────────────────────────────────────────────────

router.post("/liveness/frame", async (req, res) => {
  const { session_id, image_b64 } = req.body ?? {};
  if (!session_id || !image_b64) {
    res.status(400).json({ error: "session_id and image_b64 required." });
    return;
  }
  try {
    const { status, data } = await proxyToPython(
      "/liveness/frame", "POST", { session_id, image_b64 },
      8_000,
    );
    res.status(status).json(data);
  } catch (err) {
    console.error("[liveness/frame]", err instanceof Error ? err.message : err);
    res.status(503).json({ error: "Liveness service unavailable." });
  }
});

// ── GET /api/liveness/verify/:sessionId ──────────────────────────────────────

router.get("/liveness/verify/:sessionId", async (req, res) => {
  const { sessionId } = req.params;
  try {
    const { status, data } = await proxyToPython(
      `/liveness/verify/${sessionId}`, "GET",
    );
    res.status(status).json(data);
  } catch (err) {
    console.error("[liveness/verify]", err instanceof Error ? err.message : err);
    res.status(503).json({ error: "Liveness service unavailable." });
  }
});

// ── DELETE /api/liveness/session/:sessionId ───────────────────────────────────

router.delete("/liveness/session/:sessionId", async (req, res) => {
  const { sessionId } = req.params;
  try {
    const { status, data } = await proxyToPython(
      `/liveness/session/${sessionId}`, "DELETE",
    );
    res.status(status).json(data);
  } catch (err) {
    console.error("[liveness/delete]", err instanceof Error ? err.message : err);
    res.status(503).json({ error: "Liveness service unavailable." });
  }
});

// ── GET /api/liveness/health ──────────────────────────────────────────────────

router.get("/liveness/health", async (_req, res) => {
  try {
    const { status, data } = await proxyToPython("/health", "GET");
    res.status(status).json(data);
  } catch {
    res.status(503).json({ ok: false, error: "Liveness service unavailable." });
  }
});

export default router;

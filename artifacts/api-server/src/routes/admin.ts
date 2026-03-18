import { Router, type IRouter, type Request, type Response, type NextFunction } from "express";
import { db, usersTable } from "@workspace/db";
import { eq } from "drizzle-orm";
import crypto from "crypto";

const router: IRouter = Router();

// ── In-memory token store (single admin session) ──────────────────────────────
// For a production app this would be JWTs or a DB-backed session table.
const activeSessions = new Set<string>();

function generateToken(): string {
  return crypto.randomBytes(32).toString("hex");
}

// ── Admin credentials from environment ────────────────────────────────────────
const ADMIN_EMAIL    = process.env.ADMIN_EMAIL    ?? "admin@auraauth.com";
const ADMIN_PASSWORD = process.env.ADMIN_PASSWORD ?? "admin123";

// ── Auth middleware ───────────────────────────────────────────────────────────
function requireAdmin(req: Request, res: Response, next: NextFunction) {
  const auth = req.headers.authorization ?? "";
  const token = auth.replace("Bearer ", "").trim();
  if (!token || !activeSessions.has(token)) {
    res.status(401).json({ error: "Unauthorized" });
    return;
  }
  next();
}

// POST /api/admin/login
router.post("/admin/login", (req, res) => {
  const { email, password } = req.body;
  if (email === ADMIN_EMAIL && password === ADMIN_PASSWORD) {
    const token = generateToken();
    activeSessions.add(token);
    res.status(200).json({ token, message: "Admin login successful" });
  } else {
    res.status(401).json({ error: "Invalid admin credentials" });
  }
});

// POST /api/admin/logout
router.post("/admin/logout", requireAdmin, (req, res) => {
  const token = (req.headers.authorization ?? "").replace("Bearer ", "").trim();
  activeSessions.delete(token);
  res.status(200).json({ message: "Logged out" });
});

// GET /api/admin/users — list all users
router.get("/admin/users", requireAdmin, async (_req, res) => {
  try {
    const users = await db
      .select({
        id: usersTable.id,
        email: usersTable.email,
        is_approved: usersTable.is_approved,
        created_at: usersTable.created_at,
      })
      .from(usersTable)
      .orderBy(usersTable.created_at);

    res.status(200).json({ users });
  } catch (err) {
    console.error("Admin list users error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

// PUT /api/admin/approve-user/:id — approve a user
router.put("/admin/approve-user/:id", requireAdmin, async (req, res) => {
  const id = parseInt(req.params.id, 10);
  if (isNaN(id)) { res.status(400).json({ error: "Invalid user ID" }); return; }
  try {
    await db.update(usersTable).set({ is_approved: true }).where(eq(usersTable.id, id));
    res.status(200).json({ message: "User approved" });
  } catch (err) {
    console.error("Approve user error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

// PUT /api/admin/reject-user/:id — reject (un-approve) a user
router.put("/admin/reject-user/:id", requireAdmin, async (req, res) => {
  const id = parseInt(req.params.id, 10);
  if (isNaN(id)) { res.status(400).json({ error: "Invalid user ID" }); return; }
  try {
    await db.update(usersTable).set({ is_approved: false }).where(eq(usersTable.id, id));
    res.status(200).json({ message: "User rejected" });
  } catch (err) {
    console.error("Reject user error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

// DELETE /api/admin/delete-user/:id — permanently remove a user
router.delete("/admin/delete-user/:id", requireAdmin, async (req, res) => {
  const id = parseInt(req.params.id, 10);
  if (isNaN(id)) { res.status(400).json({ error: "Invalid user ID" }); return; }
  try {
    await db.delete(usersTable).where(eq(usersTable.id, id));
    res.status(200).json({ message: "User deleted" });
  } catch (err) {
    console.error("Delete user error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

export default router;

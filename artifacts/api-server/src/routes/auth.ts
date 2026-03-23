import { Router, type IRouter } from "express";
import bcrypt from "bcrypt";
import { db, usersTable } from "@workspace/db";
import { RegisterFaceBody, LoginFaceBody } from "@workspace/api-zod";
import { eq } from "drizzle-orm";

const router: IRouter = Router();

const SALT_ROUNDS = 10;

// Euclidean distance between two face descriptor vectors
function euclideanDistance(a: number[], b: number[]): number {
  return Math.sqrt(a.reduce((sum, val, i) => sum + (val - b[i]) ** 2, 0));
}

// POST /api/register-face - Register a new user with face descriptor
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

// POST /api/login-face - Authenticate user with face descriptor + liveness check
router.post("/login-face", async (req, res) => {
  const parsed = LoginFaceBody.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ error: "Invalid input", details: parsed.error.message });
    return;
  }

  const { email, face_descriptor, liveness_passed } = parsed.data;

  if (!liveness_passed) {
    res.status(401).json({ error: "Liveness verification failed" });
    return;
  }

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
    // face-api.js recommends 0.6 as the recognition threshold.
    // 0.5 is too strict and rejects real users with slight lighting/angle variation.
    const THRESHOLD = 0.6;

    console.log(`[face-match] user=${email} distance=${distance.toFixed(4)} threshold=${THRESHOLD} matched=${distance <= THRESHOLD}`);

    if (distance > THRESHOLD) {
      res.status(401).json({ error: "Face does not match. Please try again.", distance: parseFloat(distance.toFixed(4)) });
      return;
    }

    const confidence = Math.max(0, Math.round((1 - distance / THRESHOLD) * 100));

    res.status(200).json({
      message: "Face verified. OTP will be sent to your email.",
      face_matched: true,
      confidence,
      name: user.name ?? null,
    });
  } catch (err) {
    console.error("Login error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

// POST /api/login - Email + password login
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
      email: user.email,
      name: user.name ?? null,
    });
  } catch (err) {
    console.error("Password login error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

export default router;

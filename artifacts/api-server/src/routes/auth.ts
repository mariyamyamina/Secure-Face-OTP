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
    res.status(400).json({
      error: "Invalid input",
      details: parsed.error.message,
    });
    return;
  }

  const { email, password, face_descriptor } = parsed.data;

  try {
    // Check for duplicate email
    const existing = await db
      .select({ id: usersTable.id })
      .from(usersTable)
      .where(eq(usersTable.email, email))
      .limit(1);

    if (existing.length > 0) {
      res.status(409).json({ error: "Email already registered" });
      return;
    }

    // Hash the password
    const hashedPassword = await bcrypt.hash(password, SALT_ROUNDS);

    // Serialize the face descriptor array to JSON string
    const descriptorJson = JSON.stringify(face_descriptor);

    // Insert the user into the database
    const [newUser] = await db
      .insert(usersTable)
      .values({
        email,
        password: hashedPassword,
        face_descriptor: descriptorJson,
      })
      .returning({ id: usersTable.id });

    res.status(201).json({
      message: "User registered successfully",
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
    res.status(400).json({
      error: "Invalid input",
      details: parsed.error.message,
    });
    return;
  }

  const { email, face_descriptor, liveness_passed } = parsed.data;

  // Reject immediately if liveness check did not pass on frontend
  if (!liveness_passed) {
    res.status(401).json({ error: "Liveness verification failed" });
    return;
  }

  try {
    // Look up user by email
    const users = await db
      .select()
      .from(usersTable)
      .where(eq(usersTable.email, email))
      .limit(1);

    if (users.length === 0) {
      res.status(404).json({ error: "User not found" });
      return;
    }

    const user = users[0];
    const storedDescriptor: number[] = JSON.parse(user.face_descriptor);

    // Validate descriptor lengths match
    if (face_descriptor.length !== storedDescriptor.length) {
      res.status(400).json({ error: "Invalid face descriptor format" });
      return;
    }

    // Compare face descriptors using euclidean distance
    // Threshold of 0.5 is stricter than the 0.6 used during registration
    const distance = euclideanDistance(face_descriptor, storedDescriptor);
    const THRESHOLD = 0.5;

    if (distance > THRESHOLD) {
      res.status(401).json({ error: "Face does not match" });
      return;
    }

    // Compute a confidence score: 100% at distance=0, 0% at distance=0.5
    const confidence = Math.max(0, Math.round((1 - distance / THRESHOLD) * 100));

    res.status(200).json({
      message: "Login successful",
      matched: true,
      confidence,
    });
  } catch (err) {
    console.error("Login error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

export default router;

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

// POST /api/login-face - Authenticate user with face descriptor
router.post("/login-face", async (req, res) => {
  const parsed = LoginFaceBody.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({
      error: "Invalid input",
      details: parsed.error.message,
    });
    return;
  }

  const { email, face_descriptor } = parsed.data;

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

    // Compare face descriptors using euclidean distance
    const distance = euclideanDistance(face_descriptor, storedDescriptor);
    const THRESHOLD = 0.6; // Standard face-api.js threshold

    if (distance > THRESHOLD) {
      res.status(401).json({ error: "Face not recognized" });
      return;
    }

    res.status(200).json({
      message: "Face matched successfully. OTP would be sent to email.",
      matched: true,
    });
  } catch (err) {
    console.error("Login error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

export default router;

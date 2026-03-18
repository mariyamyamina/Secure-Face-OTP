import { Router, type IRouter } from "express";
import { db } from "@workspace/db";
import { otpVerificationsTable } from "@workspace/db";
import { eq } from "drizzle-orm";
import nodemailer from "nodemailer";

const router: IRouter = Router();

// Generate a random 6-digit OTP
function generateOTP(): string {
  return Math.floor(100000 + Math.random() * 900000).toString();
}

// Build a nodemailer transporter from env vars (falls back to console logging)
function getTransporter() {
  if (process.env.SMTP_HOST && process.env.SMTP_USER && process.env.SMTP_PASS) {
    return nodemailer.createTransport({
      host: process.env.SMTP_HOST,
      port: Number(process.env.SMTP_PORT ?? 587),
      secure: process.env.SMTP_SECURE === "true",
      auth: {
        user: process.env.SMTP_USER,
        pass: process.env.SMTP_PASS,
      },
    });
  }
  return null;
}

// POST /api/send-otp
router.post("/send-otp", async (req, res) => {
  const { email } = req.body;
  if (!email || typeof email !== "string") {
    res.status(400).json({ error: "Email is required" });
    return;
  }

  const otp = generateOTP();
  const now = new Date();
  const expiresAt = new Date(now.getTime() + 5 * 60 * 1000); // 5 minutes

  try {
    // Upsert: delete any existing OTP for this email, then insert new one
    await db.delete(otpVerificationsTable).where(eq(otpVerificationsTable.email, email));
    await db.insert(otpVerificationsTable).values({ email, otp, expires_at: expiresAt });

    // Try to send email
    const transporter = getTransporter();
    if (transporter) {
      await transporter.sendMail({
        from: process.env.SMTP_FROM ?? process.env.SMTP_USER,
        to: email,
        subject: "Your AuraAuth OTP Code",
        html: `
          <div style="font-family:sans-serif;max-width:400px;margin:0 auto;padding:32px;background:#0f0f1a;border-radius:12px;border:1px solid #2d2d5e">
            <h2 style="color:#818cf8;margin:0 0 16px">Your One-Time Password</h2>
            <p style="color:#94a3b8;margin:0 0 24px">Use the code below to complete your login. It expires in 5 minutes.</p>
            <div style="background:#1e1e3a;border:1px solid #4f46e5;border-radius:8px;padding:24px;text-align:center">
              <span style="font-size:36px;font-weight:700;letter-spacing:12px;color:#ffffff">${otp}</span>
            </div>
            <p style="color:#64748b;margin:24px 0 0;font-size:12px">If you didn't request this, ignore this email.</p>
          </div>
        `,
      });
    } else {
      // Development fallback — log OTP to server console
      console.log(`\n[OTP] Email: ${email} | Code: ${otp} | Expires: ${expiresAt.toISOString()}\n`);
    }

    res.status(200).json({
      message: "OTP sent successfully",
      // Return OTP in dev mode (no SMTP configured) so it can be displayed
      ...(process.env.NODE_ENV !== "production" && !getTransporter() ? { dev_otp: otp } : {}),
    });
  } catch (err) {
    console.error("Send OTP error:", err);
    res.status(500).json({ error: "Failed to send OTP" });
  }
});

// POST /api/verify-otp
router.post("/verify-otp", async (req, res) => {
  const { email, otp } = req.body;
  if (!email || !otp) {
    res.status(400).json({ error: "Email and OTP are required" });
    return;
  }

  try {
    const records = await db
      .select()
      .from(otpVerificationsTable)
      .where(eq(otpVerificationsTable.email, email))
      .limit(1);

    if (records.length === 0) {
      res.status(400).json({ error: "No OTP found for this email. Please request a new one." });
      return;
    }

    const record = records[0];

    // Check expiry
    if (new Date() > new Date(record.expires_at)) {
      await db.delete(otpVerificationsTable).where(eq(otpVerificationsTable.email, email));
      res.status(400).json({ error: "OTP has expired. Please request a new one." });
      return;
    }

    // Check OTP value
    if (record.otp !== otp.toString().trim()) {
      res.status(400).json({ error: "Incorrect OTP. Please try again." });
      return;
    }

    // OTP is valid — delete it so it can't be reused
    await db.delete(otpVerificationsTable).where(eq(otpVerificationsTable.email, email));

    res.status(200).json({ message: "OTP verified successfully", verified: true });
  } catch (err) {
    console.error("Verify OTP error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
});

export default router;

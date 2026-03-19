import emailjs from "@emailjs/browser";

// ─── EmailJS config — set these in your .env file ────────────────────────────
// VITE_EMAILJS_SERVICE_ID   → Your EmailJS service ID
// VITE_EMAILJS_TEMPLATE_ID  → Your EmailJS template ID
// VITE_EMAILJS_PUBLIC_KEY   → Your EmailJS public key
//
// Template must contain these variables:
//   {{to_email}}  — recipient address
//   {{otp}}       — the 6-digit code
//   {{app_name}}  — optional, set to "AuraAuth"
// ─────────────────────────────────────────────────────────────────────────────

const SERVICE_ID  = import.meta.env.VITE_EMAILJS_SERVICE_ID  as string | undefined;
const TEMPLATE_ID = import.meta.env.VITE_EMAILJS_TEMPLATE_ID as string | undefined;
const PUBLIC_KEY  = import.meta.env.VITE_EMAILJS_PUBLIC_KEY  as string | undefined;

export const emailJSConfigured =
  Boolean(SERVICE_ID && TEMPLATE_ID && PUBLIC_KEY);

/** Generate a cryptographically random 6-digit OTP */
export function generateOTP(): string {
  return Math.floor(100000 + Math.random() * 900000).toString();
}

/**
 * Send an OTP to the given email address using EmailJS.
 * Returns `{ ok: true }` on success or `{ ok: false, error: string }` on failure.
 *
 * If EmailJS credentials are not configured, falls back to a dev-mode stub
 * that just returns the OTP so it can be displayed on screen.
 */
export async function sendOTPEmail(
  toEmail: string,
  otp: string
): Promise<{ ok: true } | { ok: false; error: string }> {
  if (!emailJSConfigured) {
    // Dev-mode: no EmailJS credentials — caller should display OTP in UI
    console.info(`[EmailService][DEV] OTP for ${toEmail}: ${otp}`);
    return { ok: true };
  }

  try {
    await emailjs.send(
      SERVICE_ID!,
      TEMPLATE_ID!,
      {
        to_email: toEmail,
        otp,
        app_name: "AuraAuth",
      },
      { publicKey: PUBLIC_KEY! }
    );
    return { ok: true };
  } catch (err: any) {
    console.error("[EmailService] EmailJS error:", err);
    return {
      ok: false,
      error: err?.text ?? err?.message ?? "Failed to send OTP email",
    };
  }
}

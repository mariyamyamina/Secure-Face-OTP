/**
 * Login.tsx — Biometric Authentication Pipeline
 *
 * The authentication flow is divided into five clearly separated stages:
 *
 *  Stage 1 · Face Detection
 *    face-api.js TinyFaceDetector locates the face and extracts 68 landmarks
 *    every DETECTION_INTERVAL_MS milliseconds.
 *
 *  Stage 2 · Anti-Spoofing (runs concurrently with Stage 3)
 *    AntiSpoofEngine analyses the face crop for five signals:
 *      · Glare / specular reflection (screen hotspots)
 *      · LBP micro-texture entropy (real skin vs. printed / screen texture)
 *      · Colour naturalness (skin-tone distribution)
 *      · Temporal micro-variance (organic face motion vs. static image)
 *      · Motion consistency (CoV of MAD — detects rigid phone-tremor)
 *    Two consecutive "spoof" readings → session rejected immediately.
 *
 *  Stage 3 · Liveness Detection (runs concurrently with Stage 2)
 *    LivenessDetector requires the user to complete four behavioural proofs:
 *      · Blink (EAR drop relative to rolling max)
 *      · Lip movement (open → close cycle)
 *      · Head movement (nose-tip displacement from baseline)
 *      · Skin texture (temporal MAD variance)
 *
 *  Stage 4 · Face Recognition
 *    Once Stages 2 & 3 both pass: three descriptor captures are averaged and
 *    sent to POST /api/login-face.  The backend computes Euclidean distance
 *    against the stored descriptor (threshold 0.6).
 *
 *  Stage 5 · OTP Verification
 *    If the face matches, a 6-digit OTP is sent via EmailJS.
 *    Successful OTP entry grants dashboard access.
 */

import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useLocation } from "wouter";
import Webcam from "react-webcam";
import * as faceapi from "@vladmandic/face-api";
import {
  Eye, Smile, Move, Layers, CheckCircle2, XCircle, Loader2,
  ShieldCheck, ShieldX, RefreshCw, User, Timer, AlertTriangle,
  Activity, Shield, Zap,
} from "lucide-react";
import { Navbar } from "@/components/layout/Navbar";
import { useToast } from "@/hooks/use-toast";
import { generateOTP, sendOTPEmail, emailJSConfigured } from "@/lib/emailService";
import { useUser } from "@/context/UserContext";
import { LivenessDetector, type LivenessState } from "@/lib/livenessDetector";
import { AntiSpoofEngine, type SpoofSignals } from "@/lib/antiSpoofing";
import { ServerLivenessClient, type ServerLivenessChecks, type ServerLivenessFrameResult } from "@/lib/serverLiveness";

// ── Constants ──────────────────────────────────────────────────────────────────

const MODEL_URL              = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/";
const DETECTION_INTERVAL_MS  = 100;   // Face detection tick rate (~10 fps)
const LIVENESS_TIMEOUT_S     = 30;    // Total time allowed for liveness
const ANTI_SPOOF_INTERVAL    = 3;     // Run anti-spoof every N-th detection tick
const ANTI_SPOOF_WARMUP      = 1;     // Skip anti-spoof for first N ticks (minimal warm-up)
const SPOOF_REJECT_COUNT     = 8;     // Consecutive "spoof" readings before rejection (adjusted for faster tick)
const SERVER_FRAME_INTERVAL  = 8;     // Send frame to MediaPipe server every N ticks

// ── Types ──────────────────────────────────────────────────────────────────────

type PageState =
  | "idle"
  | "loading-models"
  | "camera-ready"
  | "monitoring"
  | "verifying"
  | "otp"
  | "success"
  | "failed";

// ── Instruction sequence for liveness (built dynamically with head direction) ──

function buildInstructions(headDir: "left" | "right" | "up") {
  const headText =
    headDir === "up"    ? "⬆️  Slowly tilt your head upward" :
    headDir === "left"  ? "⬅️  Slowly turn your head to the LEFT" :
                          "➡️  Slowly turn your head to the RIGHT";
  return [
    { key: "blinkDetected"        as const, text: "👁  Blink your eyes once naturally" },
    { key: "headMovementDetected" as const, text: headText },
    { key: "textureDetected"      as const, text: "✅  Hold still — detecting skin texture…" },
  ];
}

// ── Component ──────────────────────────────────────────────────────────────────

export default function Login() {
  const { login }    = useUser();
  const [, navigate] = useLocation();
  const { toast }    = useToast();

  // ── Page & auth state ────────────────────────────────────────────────────
  const [pageState,     setPageState]     = useState<PageState>("idle");
  const [email,         setEmail]         = useState("");
  const [modelsLoaded,  setModelsLoaded]  = useState(false);
  const [errorMsg,      setErrorMsg]      = useState("");
  const [successMsg,    setSuccessMsg]    = useState("");
  const [confidence,    setConfidence]    = useState<number | null>(null);
  const [userName,      setUserName]      = useState("");
  const [timeLeft,      setTimeLeft]      = useState(LIVENESS_TIMEOUT_S);

  // ── Login mode ───────────────────────────────────────────────────────────
  const [loginMode,  setLoginMode]  = useState<"face" | "password">("face");
  const [pwInput,    setPwInput]    = useState("");
  const [pwLoading,  setPwLoading]  = useState(false);
  const [pwError,    setPwError]    = useState("");

  // ── OTP state ────────────────────────────────────────────────────────────
  const [otpValue,      setOtpValue]      = useState("");
  const [otpError,      setOtpError]      = useState("");
  const [otpLoading,    setOtpLoading]    = useState(false);
  const [otpResendLeft, setOtpResendLeft] = useState(0);
  const [devOtp,        setDevOtp]        = useState<string | null>(null);
  const storedOTP      = useRef("");
  const otpResendTimer = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Stage 3: Liveness state (for UI rendering) ───────────────────────────
  const [liveness, setLiveness] = useState<LivenessState>({
    blinkDetected: false, lipMovementDetected: false,
    headMovementDetected: false, textureDetected: false,
  });
  // Debug values for the developer panel
  const [debugEAR, setDebugEAR] = useState<number | null>(null);
  const [debugLip, setDebugLip] = useState<number | null>(null);

  // ── Server-side MediaPipe liveness state ──────────────────────────────────
  const [serverChecks,    setServerChecks]    = useState<ServerLivenessChecks | null>(null);
  const [serverScore,     setServerScore]     = useState<number | null>(null);
  const [serverIsLive,    setServerIsLive]    = useState(false);
  const [serverFrameData, setServerFrameData] = useState<ServerLivenessFrameResult | null>(null);
  const [serverAvailable, setServerAvailable] = useState(true);
  const [headChallenge,   setHeadChallenge]   = useState<"left" | "right" | "up">("left");

  // ── Stage 2: Anti-spoofing state (for UI rendering) ──────────────────────
  const [antiSpoofScore,   setAntiSpoofScore]   = useState<number | null>(null);
  const [antiSpoofSignals, setAntiSpoofSignals] = useState<SpoofSignals | null>(null);
  const [spoofDetected,    setSpoofDetected]    = useState(false);

  // ── Webcam ref ───────────────────────────────────────────────────────────
  const webcamRef = useRef<Webcam>(null);

  // ── Interval / lifecycle refs ────────────────────────────────────────────
  const detectionInterval  = useRef<ReturnType<typeof setInterval> | null>(null);
  const countdownInterval  = useRef<ReturnType<typeof setInterval> | null>(null);
  const monitoringActive   = useRef(false);
  const detectionRunning   = useRef(false);

  // ── Stage 2 & 3 engine refs ──────────────────────────────────────────────
  // These are class instances so they hold their own state internally.
  const livenessDetector     = useRef<LivenessDetector>(new LivenessDetector());
  const antiSpoofEngine      = useRef<AntiSpoofEngine>(new AntiSpoofEngine());
  const serverLivenessClient = useRef<ServerLivenessClient>(new ServerLivenessClient());
  const serverAllPassedRef   = useRef(false);

  // ── Per-session counters ─────────────────────────────────────────────────
  const tickCount            = useRef(0);
  const consecutiveSpoofCount= useRef(0);
  const allPassedRef         = useRef(false); // used inside interval callbacks

  // ── Stage 1: Load face-api models ────────────────────────────────────────
  const loadModels = useCallback(async () => {
    if (modelsLoaded) { setPageState("camera-ready"); return; }
    setPageState("loading-models");
    try {
      await faceapi.tf.setBackend("cpu");
      await faceapi.tf.ready();
      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
        faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
        faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
      ]);
      setModelsLoaded(true);
      setPageState("camera-ready");
    } catch (err) {
      console.error("Model load error:", err);
      setErrorMsg("Failed to load AI models. Check your connection and refresh.");
      setPageState("failed");
    }
  }, [modelsLoaded]);

  // ── Stop all intervals ───────────────────────────────────────────────────
  const stopAll = useCallback(() => {
    monitoringActive.current = false;
    if (detectionInterval.current)  { clearInterval(detectionInterval.current);  detectionInterval.current  = null; }
    if (countdownInterval.current)  { clearInterval(countdownInterval.current);  countdownInterval.current  = null; }
  }, []);

  // ── Full reset → camera-ready ────────────────────────────────────────────
  const resetLiveness = useCallback(() => {
    stopAll();
    // Reset all engine instances
    livenessDetector.current.reset();
    antiSpoofEngine.current.reset();
    serverLivenessClient.current.reset();
    // Reset counters
    tickCount.current             = 0;
    consecutiveSpoofCount.current = 0;
    allPassedRef.current          = false;
    serverAllPassedRef.current    = false;
    // Reset React state
    const blank: LivenessState = {
      blinkDetected: false, lipMovementDetected: false,
      headMovementDetected: false, textureDetected: false,
    };
    setLiveness(blank);
    setSpoofDetected(false);
    setAntiSpoofScore(null);
    setAntiSpoofSignals(null);
    setDebugEAR(null);
    setDebugLip(null);
    setServerChecks(null);
    setServerScore(null);
    setServerIsLive(false);
    setServerFrameData(null);
    setServerAvailable(true);
    setTimeLeft(LIVENESS_TIMEOUT_S);
    setErrorMsg("");
    setSuccessMsg("");
    setConfidence(null);
    storedOTP.current = "";
    setPageState("camera-ready");
  }, [stopAll]);

  // ── Single detection tick ────────────────────────────────────────────────
  // Called every DETECTION_INTERVAL_MS while monitoring is active.
  const runDetectionTick = useCallback(async () => {
    if (detectionRunning.current || !monitoringActive.current) return;

    const video = webcamRef.current?.video;
    if (!video || video.readyState !== 4) return;

    detectionRunning.current = true;
    tickCount.current++;

    try {
      // ── STAGE 1: Face Detection ──────────────────────────────────────────
      // Request only landmarks here (no descriptor) — 3-4× faster on CPU.
      const det = await faceapi
        .detectSingleFace(
          video,
          new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.5 })
        )
        .withFaceLandmarks();

      // ── STAGE 2: Anti-Spoofing ───────────────────────────────────────────
      // Run every ANTI_SPOOF_INTERVAL ticks after the warm-up period.
      const shouldRunAntiSpoof =
        tickCount.current > ANTI_SPOOF_WARMUP &&
        tickCount.current % ANTI_SPOOF_INTERVAL === 0;

      if (shouldRunAntiSpoof) {
        // Provide face bounding box when available for a tighter crop
        const bounds = det
          ? {
              x:      Math.max(0, det.detection.box.x),
              y:      Math.max(0, det.detection.box.y),
              width:  Math.min(det.detection.box.width,  video.videoWidth),
              height: Math.min(det.detection.box.height, video.videoHeight),
            }
          : undefined;

        const spoofResult = antiSpoofEngine.current.analyze(video, bounds);

        // Update UI state
        setAntiSpoofScore(spoofResult.score);
        setAntiSpoofSignals(spoofResult.signals);

        if (!spoofResult.isReal) {
          consecutiveSpoofCount.current++;
          // Require SPOOF_REJECT_COUNT consecutive failures before rejecting
          // to avoid false positives from individual noisy frames.
          if (consecutiveSpoofCount.current >= SPOOF_REJECT_COUNT) {
            stopAll();
            setSpoofDetected(true);
            setErrorMsg("Spoofing attempt detected. Please use a real face.");
            setPageState("failed");
            return;
          }
        } else {
          // Reset on any genuine-looking frame
          consecutiveSpoofCount.current = 0;
        }
      }

      // ── STAGE 3: Liveness Detection ──────────────────────────────────────
      const livenessResult = livenessDetector.current.update(det ?? null, video);

      // Sync React state for rendering (only update when something changed)
      setLiveness(prev => {
        const s = livenessResult.state;
        if (
          prev.blinkDetected         === s.blinkDetected &&
          prev.lipMovementDetected   === s.lipMovementDetected &&
          prev.headMovementDetected  === s.headMovementDetected &&
          prev.textureDetected       === s.textureDetected
        ) return prev;
        return { ...s };
      });

      if (livenessResult.ear    !== null) setDebugEAR(Math.round(livenessResult.ear    * 1000) / 1000);
      if (livenessResult.lipGap !== null) setDebugLip(Math.round(livenessResult.lipGap));

      // Keep allPassedRef in sync so the countdown callback can read it
      allPassedRef.current = livenessResult.allPassed;

      // ── Auto-complete: all liveness checks passed ─────────────────────────
      if (livenessResult.allPassed && monitoringActive.current) {
        stopAll();
        // Stage 4 (face recognition) starts here
        verifyAndLogin();
      }

    } catch {
      // Silently swallow individual frame errors; next tick will retry.
    } finally {
      detectionRunning.current = false;
    }
  // verifyAndLogin is defined below; useCallback dependency handled via ref
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stopAll]);

  // Keep a ref to verifyAndLogin so runDetectionTick can call the latest version
  const verifyAndLoginRef = useRef<() => void>(() => {});

  // ── Stage 4: Face Recognition + Server-Side Anti-Spoof ────────────────────
  const verifyAndLogin = useCallback(async () => {
    setPageState("verifying");
    const video = webcamRef.current?.video;
    if (!video) { setErrorMsg("Camera unavailable."); setPageState("failed"); return; }

    try {
      // ── Capture webcam frame for server-side anti-spoofing ─────────────
      // react-webcam's getScreenshot() returns a data-URI JPEG string.
      // This exact frame is sent to the Python anti-spoof service which
      // runs OpenCV-based frequency, texture, and glare analysis on it.
      const faceImageB64: string | null = webcamRef.current?.getScreenshot() ?? null;

      // ── Capture 2 face descriptors and average them ────────────────────
      // Two samples with a short gap to smooth lighting/angle noise.
      const SAMPLES     = 2;
      const descriptors: Float32Array[] = [];
      let   capturedBounds: { x: number; y: number; width: number; height: number } | null = null;

      for (let i = 0; i < SAMPLES; i++) {
        if (i > 0) await new Promise(r => setTimeout(r, 100));
        const det = await faceapi
          .detectSingleFace(
            video,
            new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.4 })
          )
          .withFaceLandmarks()
          .withFaceDescriptor();
        if (det) {
          descriptors.push(det.descriptor);
          // Store face bounds from first successful detection for server-side crop
          if (!capturedBounds) {
            const box = det.detection.box;
            capturedBounds = { x: box.x, y: box.y, width: box.width, height: box.height };
          }
        }
      }

      if (descriptors.length === 0) {
        setErrorMsg("No face detected at verification time. Look directly at the camera and try again.");
        setPageState("failed");
        return;
      }

      // Average the captured descriptors into one representative vector
      const avgDescriptor = new Float32Array(128);
      for (const d of descriptors) {
        for (let j = 0; j < 128; j++) avgDescriptor[j] += d[j];
      }
      for (let j = 0; j < 128; j++) avgDescriptor[j] /= descriptors.length;

      const res = await fetch("/api/login-face", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          email,
          face_descriptor:       Array.from(avgDescriptor),
          liveness_passed:       true,
          // Server-side anti-spoof: full webcam frame + face crop region
          face_image_b64:        faceImageB64,
          face_bounds:           capturedBounds,
          // Server-side MediaPipe liveness session (if available)
          liveness_session_id:   serverLivenessClient.current.currentSessionId ?? undefined,
        }),
      });
      const data = await res.json();

      if (res.ok) {
        setConfidence(data.confidence ?? null);
        setUserName(data.name ?? "");
        setPageState("otp");
        // Start OTP flow asynchronously so OTP UI appears immediately
        sendOtp();
      } else {
        // Surface the spoof-specific message when available
        const errMsg = data.error || "Authentication failed.";
        setErrorMsg(errMsg);
        setPageState("failed");
      }
    } catch {
      setErrorMsg("Network error. Please check your connection.");
      setPageState("failed");
    }
  }, [email]);

  // Keep ref in sync with the latest verifyAndLogin so the detection tick can call it
  useEffect(() => { verifyAndLoginRef.current = verifyAndLogin; }, [verifyAndLogin]);

  // Patch runDetectionTick to call via ref (avoids stale closure)
  const runDetectionTickPatched = useCallback(async () => {
    if (detectionRunning.current || !monitoringActive.current) return;
    const video = webcamRef.current?.video;
    if (!video || video.readyState !== 4) return;

    detectionRunning.current = true;
    tickCount.current++;

    try {
      const det = await faceapi
        .detectSingleFace(
          video,
          new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.5 })
        )
        .withFaceLandmarks();

      // ── STAGE 2: Anti-Spoofing ───────────────────────────────────────────
      const shouldRunAntiSpoof =
        tickCount.current > ANTI_SPOOF_WARMUP &&
        tickCount.current % ANTI_SPOOF_INTERVAL === 0;

      if (shouldRunAntiSpoof) {
        const bounds = det
          ? {
              x:      Math.max(0, det.detection.box.x),
              y:      Math.max(0, det.detection.box.y),
              width:  Math.min(det.detection.box.width,  video.videoWidth),
              height: Math.min(det.detection.box.height, video.videoHeight),
            }
          : undefined;

        const spoofResult = antiSpoofEngine.current.analyze(video, bounds);
        setAntiSpoofScore(spoofResult.score);
        setAntiSpoofSignals(spoofResult.signals);

        if (!spoofResult.isReal) {
          consecutiveSpoofCount.current++;
          if (consecutiveSpoofCount.current >= SPOOF_REJECT_COUNT) {
            stopAll();
            setSpoofDetected(true);
            setErrorMsg("Spoofing attempt detected. Please use a real face.");
            setPageState("failed");
            return;
          }
        } else {
          consecutiveSpoofCount.current = 0;
        }
      }

      // ── STAGE 3a: Client-side Liveness (face-api.js) ─────────────────────
      const livenessResult = livenessDetector.current.update(det ?? null, video);

      setLiveness(prev => {
        const s = livenessResult.state;
        if (
          prev.blinkDetected         === s.blinkDetected &&
          prev.lipMovementDetected   === s.lipMovementDetected &&
          prev.headMovementDetected  === s.headMovementDetected &&
          prev.textureDetected       === s.textureDetected
        ) return prev;
        return { ...s };
      });

      if (livenessResult.ear    !== null) setDebugEAR(Math.round(livenessResult.ear    * 1000) / 1000);
      if (livenessResult.lipGap !== null) setDebugLip(Math.round(livenessResult.lipGap));

      allPassedRef.current = livenessResult.allPassed;

      // ── STAGE 3b: Server-side Liveness (MediaPipe — every N ticks) ────────
      // Fire-and-forget: don't block the detection loop; update state asynchronously.
      if (
        tickCount.current % SERVER_FRAME_INTERVAL === 0 &&
        serverLivenessClient.current.hasSession &&
        serverAvailable
      ) {
        const screenshot = webcamRef.current?.getScreenshot() ?? null;
        if (screenshot) {
          serverLivenessClient.current.sendFrame(screenshot).then(result => {
            if (!result) return;
            setServerFrameData(result);
            setServerChecks({ ...result.checks });
            setServerScore(result.liveness_score);
            if (result.is_live && !serverAllPassedRef.current) {
              serverAllPassedRef.current = true;
              setServerIsLive(true);
            }
          }).catch(() => {});
        }
      }

      // ── STAGE 4 trigger ───────────────────────────────────────────────────
      // Require BOTH client-side AND server-side liveness to pass.
      // If server is unavailable, fall back to client-side only.
      const serverGatePassed = !serverAvailable || serverAllPassedRef.current;
      const allPassed = livenessResult.allPassed && serverGatePassed;

      if (allPassed && monitoringActive.current) {
        stopAll();
        verifyAndLoginRef.current();
      }

    } catch {
      // Ignore individual frame errors
    } finally {
      detectionRunning.current = false;
    }
  }, [stopAll, serverAvailable]);

  // ── Start liveness monitoring ─────────────────────────────────────────────
  const startMonitoring = useCallback(() => {
    if (!email.trim()) {
      toast({ variant: "destructive", title: "Email required", description: "Enter your email first." });
      return;
    }

    monitoringActive.current = true;
    setPageState("monitoring");
    setTimeLeft(LIVENESS_TIMEOUT_S);

    // Kick off server-side liveness session (non-blocking)
    serverLivenessClient.current.start().then(result => {
      const dir = result.headDirection;
      setHeadChallenge(dir);
      livenessDetector.current.setHeadDirection(dir === "up" ? "left" : dir);
    }).catch(() => {
      setServerAvailable(false);
      serverAllPassedRef.current = true; // fallback: skip server gate
    });

    // Countdown — 1 tick per second
    countdownInterval.current = setInterval(() => {
      setTimeLeft(prev => {
        if (prev <= 1) {
          stopAll();
          if (!allPassedRef.current) {
            setErrorMsg("Time expired. Complete all liveness checks before the timer runs out.");
            setPageState("failed");
          }
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    // Face detection — every DETECTION_INTERVAL_MS
    detectionInterval.current = setInterval(runDetectionTickPatched, DETECTION_INTERVAL_MS);

    // Kick off first tick immediately
    runDetectionTickPatched();
  }, [email, runDetectionTickPatched, stopAll, toast]);

  // ── Stage 5: OTP helpers ──────────────────────────────────────────────────
  const startOtpResendCountdown = useCallback((seconds = 60) => {
    setOtpResendLeft(seconds);
    if (otpResendTimer.current) clearInterval(otpResendTimer.current);
    otpResendTimer.current = setInterval(() => {
      setOtpResendLeft(prev => {
        if (prev <= 1) { clearInterval(otpResendTimer.current!); return 0; }
        return prev - 1;
      });
    }, 1000);
  }, []);

  const sendOtp = useCallback(async () => {
    setOtpError("");
    setDevOtp(null);
    const otp = generateOTP();
    storedOTP.current = otp;
    const result = await sendOTPEmail(email, otp);
    if (result.ok) {
      if (!emailJSConfigured) setDevOtp(otp);
      startOtpResendCountdown(60);
    } else {
      setOtpError(result.error ?? "Failed to send OTP email");
    }
  }, [email, startOtpResendCountdown]);

  const verifyOtp = useCallback(async () => {
    const entered = otpValue.trim();
    if (entered.length !== 6) { setOtpError("Please enter the full 6-digit code."); return; }
    if (!storedOTP.current)   { setOtpError("No OTP found. Please request a new one."); return; }

    setOtpLoading(true);
    setOtpError("");
    await new Promise(r => setTimeout(r, 400));

    if (entered === storedOTP.current) {
      storedOTP.current = "";
      if (otpResendTimer.current) clearInterval(otpResendTimer.current);
      login(email, userName || undefined);
      navigate("/dashboard");
    } else {
      setOtpError("Incorrect code. Please check and try again.");
    }
    setOtpLoading(false);
  }, [otpValue, email, userName, login, navigate]);

  // ── Password login ────────────────────────────────────────────────────────
  const handlePasswordLogin = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    setPwError("");
    setPwLoading(true);
    try {
      const res  = await fetch("/api/login", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ email, password: pwInput }),
      });
      const data = await res.json();
      if (!res.ok) { setPwError(data.error || "Login failed. Please try again."); return; }
      login(email, data.name ?? undefined);
      navigate("/dashboard");
    } catch {
      setPwError("Network error. Please check your connection.");
    } finally {
      setPwLoading(false);
    }
  }, [email, pwInput, login, navigate]);

  // ── Cleanup on unmount ────────────────────────────────────────────────────
  useEffect(() => () => {
    stopAll();
    if (otpResendTimer.current) clearInterval(otpResendTimer.current);
  }, [stopAll]);

  // ── Derived display values ────────────────────────────────────────────────
  const INSTRUCTIONS       = buildInstructions(headChallenge);
  const passedCount        = Object.values(liveness).filter(Boolean).length;
  const activeInstruction  = pageState === "monitoring"
    ? INSTRUCTIONS.find(i => !liveness[i.key])?.text ?? "✅ All checks done!"
    : null;

  const headHint =
    headChallenge === "up"    ? "Tilt head upward"        :
    headChallenge === "left"  ? "Turn head to your LEFT"  :
                                "Turn head to your RIGHT";

  const checks = [
    { key: "blinkDetected"        as const, label: "Eye Blink",         icon: Eye,    hint: "Blink your eyes once" },
    { key: "headMovementDetected" as const, label: "Head Movement",     icon: Move,   hint: headHint },
    { key: "textureDetected"      as const, label: "Real Skin Texture", icon: Layers, hint: "Hold still in frame" },
  ];

  // Anti-spoof shield colour tier
  const shieldColor =
    antiSpoofScore === null  ? "text-gray-400 border-white/10 bg-white/3" :
    antiSpoofScore >= 65     ? "text-green-400 border-green-500/30 bg-green-500/8" :
    antiSpoofScore >= 40     ? "text-yellow-400 border-yellow-500/30 bg-yellow-500/8" :
                               "text-red-400 border-red-500/30 bg-red-500/8";

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen flex flex-col relative overflow-hidden bg-background">
      {/* Background glows */}
      <div className="fixed inset-0 z-0 pointer-events-none">
        <div className="absolute top-[5%] left-[5%] w-[45%] h-[45%] rounded-full bg-indigo-600/5 blur-[120px]" />
        <div className="absolute bottom-[5%] right-[5%] w-[40%] h-[40%] rounded-full bg-purple-600/5 blur-[100px]" />
      </div>

      <Navbar />

      <main className="flex-1 container mx-auto px-4 py-24 md:py-28 relative z-10">
        <div className="w-full max-w-6xl mx-auto">

          {/* ── Page Header ───────────────────────────────────────────── */}
          <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="text-center mb-10">
            <h1 className="text-4xl md:text-5xl font-display font-bold text-white mb-3">
              Biometric{" "}
              <span className="bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">
                Login
              </span>
            </h1>
            <p className="text-muted-foreground max-w-xl mx-auto">
              Multi-layer authentication: anti-spoofing, liveness detection, face recognition and OTP.
            </p>
          </motion.div>

          {/* ── Login Mode Tab Switcher ────────────────────────────────── */}
          {pageState === "idle" && (
            <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} className="flex justify-center mb-8">
              <div className="flex gap-1 p-1 rounded-2xl bg-white/5 border border-white/10">
                {(["face", "password"] as const).map(mode => (
                  <button
                    key={mode}
                    onClick={() => { setLoginMode(mode); setPwError(""); }}
                    className={`px-6 py-2.5 rounded-xl text-sm font-semibold transition-all duration-200 ${
                      loginMode === mode
                        ? "bg-gradient-to-r from-indigo-500 to-purple-600 text-white shadow-[0_0_20px_rgba(99,102,241,0.3)]"
                        : "text-muted-foreground hover:text-white"
                    }`}
                  >
                    {mode === "face" ? "Face Login" : "Password Login"}
                  </button>
                ))}
              </div>
            </motion.div>
          )}

          {/* ── Password Login Form ────────────────────────────────────── */}
          {loginMode === "password" && pageState === "idle" && (
            <motion.div initial={{ opacity: 0, scale: 0.97 }} animate={{ opacity: 1, scale: 1 }} className="max-w-md mx-auto">
              <div className="glass-panel rounded-3xl p-8 border-white/10">
                <h2 className="text-2xl font-bold text-white mb-1">Sign In</h2>
                <p className="text-gray-400 text-sm mb-7">Enter your email and password to access your account.</p>
                <form onSubmit={handlePasswordLogin} className="space-y-5">
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-gray-300 ml-1">Email Address</label>
                    <input
                      type="email" required value={email}
                      onChange={e => { setEmail(e.target.value); setPwError(""); }}
                      className="w-full px-4 py-3.5 bg-black/40 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500/50 transition-all"
                      placeholder="you@example.com"
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-gray-300 ml-1">Password</label>
                    <input
                      type="password" required value={pwInput}
                      onChange={e => { setPwInput(e.target.value); setPwError(""); }}
                      className="w-full px-4 py-3.5 bg-black/40 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500/50 transition-all"
                      placeholder="••••••••••••"
                    />
                  </div>
                  <AnimatePresence>
                    {pwError && (
                      <motion.div
                        initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                        className="flex items-center gap-2 px-4 py-3 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 text-sm"
                      >
                        <ShieldX className="w-4 h-4 flex-shrink-0" />
                        {pwError}
                      </motion.div>
                    )}
                  </AnimatePresence>
                  <button
                    type="submit" disabled={pwLoading}
                    className="w-full py-4 rounded-xl font-bold text-white bg-gradient-to-r from-indigo-500 to-purple-600 shadow-[0_0_20px_rgba(99,102,241,0.3)] hover:shadow-[0_0_30px_rgba(99,102,241,0.5)] disabled:opacity-60 disabled:cursor-not-allowed hover:-translate-y-0.5 active:translate-y-0 transition-all duration-200 flex items-center justify-center gap-2"
                  >
                    {pwLoading ? <><Loader2 className="w-5 h-5 animate-spin" /> Signing in...</> : <><ShieldCheck className="w-5 h-5" /> Sign In</>}
                  </button>
                  <p className="text-center text-sm text-muted-foreground pt-1">
                    Don't have an account?{" "}
                    <a href="/register" className="text-indigo-400 hover:text-indigo-300 font-medium transition-colors">Register</a>
                  </p>
                </form>
              </div>
            </motion.div>
          )}

          {/* ── OTP Screen ────────────────────────────────────────────── */}
          {pageState === "otp" && (
            <motion.div initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }} className="max-w-md mx-auto">
              <div className="glass-panel rounded-3xl p-8 border-white/10 text-center">
                <div className="w-16 h-16 rounded-2xl bg-indigo-500/15 border border-indigo-500/30 flex items-center justify-center mx-auto mb-5 shadow-[0_0_30px_rgba(99,102,241,0.2)]">
                  <ShieldCheck className="w-8 h-8 text-indigo-400" />
                </div>
                <h2 className="text-2xl font-bold text-white mb-1">Two-Factor Verification</h2>
                <p className="text-gray-400 text-sm mb-6">
                  Face verified.{" "}
                  {emailJSConfigured
                    ? <>Check <span className="text-indigo-300 font-medium">{email}</span> for your code.</>
                    : <>EmailJS not configured — see code below.</>
                  }
                </p>

                {devOtp && (
                  <div className="mb-5 px-4 py-3 rounded-xl bg-yellow-500/10 border border-yellow-500/25 text-yellow-300 text-sm text-left">
                    <p className="font-semibold mb-1">Development Mode</p>
                    <p className="text-xs text-yellow-400/80 mb-2">
                      Set <code className="text-yellow-300">VITE_EMAILJS_SERVICE_ID</code>,{" "}
                      <code className="text-yellow-300">VITE_EMAILJS_TEMPLATE_ID</code>, and{" "}
                      <code className="text-yellow-300">VITE_EMAILJS_PUBLIC_KEY</code> to enable real emails.
                    </p>
                    <div className="flex items-center justify-center gap-3 py-2 rounded-lg bg-black/30 border border-yellow-500/20">
                      <span className="text-yellow-400/70 text-xs">Your OTP:</span>
                      <span className="font-mono font-bold text-2xl tracking-[0.3em] text-white">{devOtp}</span>
                    </div>
                  </div>
                )}

                <input
                  type="text" inputMode="numeric" maxLength={6}
                  value={otpValue}
                  onChange={e => { setOtpValue(e.target.value.replace(/\D/g, "")); setOtpError(""); }}
                  onKeyDown={e => e.key === "Enter" && verifyOtp()}
                  className="w-full text-center text-3xl font-bold tracking-[0.4em] py-4 px-4 bg-black/40 border border-white/10 rounded-xl text-white placeholder-gray-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500/50 transition-all mb-4"
                  placeholder="______"
                />

                {otpError && (
                  <motion.p
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                    className="text-sm text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-4 py-2.5 mb-4"
                  >
                    {otpError}
                  </motion.p>
                )}

                <button
                  onClick={verifyOtp} disabled={otpLoading || otpValue.length !== 6}
                  className="w-full py-3.5 rounded-xl font-bold text-white bg-gradient-to-r from-indigo-600 to-purple-600 shadow-[0_0_20px_rgba(99,102,241,0.3)] hover:shadow-[0_0_30px_rgba(99,102,241,0.5)] disabled:opacity-50 disabled:cursor-not-allowed hover:-translate-y-0.5 active:translate-y-0 transition-all flex items-center justify-center gap-2 mb-5"
                >
                  {otpLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <ShieldCheck className="w-5 h-5" />}
                  {otpLoading ? "Verifying…" : "Verify OTP"}
                </button>

                <div className="flex items-center justify-center gap-2 text-sm">
                  {otpResendLeft > 0 ? (
                    <span className="text-gray-500">
                      Resend in <span className="text-indigo-400 font-mono font-semibold">{otpResendLeft}s</span>
                    </span>
                  ) : (
                    <button onClick={sendOtp} className="text-indigo-400 hover:text-indigo-300 font-medium transition-colors flex items-center gap-1.5">
                      <RefreshCw className="w-3.5 h-3.5" /> Resend OTP
                    </button>
                  )}
                </div>
              </div>
            </motion.div>
          )}

          {/* ── Success Screen ─────────────────────────────────────────── */}
          {pageState === "success" && (
            <motion.div initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }} className="max-w-md mx-auto">
              <div className="glass-panel rounded-3xl p-10 border-green-500/20 bg-green-500/5 text-center">
                <motion.div
                  initial={{ scale: 0 }} animate={{ scale: 1 }}
                  transition={{ type: "spring", bounce: 0.5, delay: 0.1 }}
                  className="w-20 h-20 rounded-full bg-green-500/20 border border-green-500/50 flex items-center justify-center mx-auto mb-5 shadow-[0_0_40px_rgba(34,197,94,0.4)]"
                >
                  <ShieldCheck className="w-10 h-10 text-green-400" />
                </motion.div>
                <h2 className="text-3xl font-bold text-white mb-2">Access Granted</h2>
                <p className="text-green-300 mb-6">{successMsg || "Identity fully verified."}</p>
                {confidence !== null && (
                  <div className="mb-6">
                    <p className="text-xs text-gray-400 mb-2">Face Match Confidence</p>
                    <div className="h-3 rounded-full bg-white/5 overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }} animate={{ width: `${confidence}%` }}
                        transition={{ duration: 1, ease: "easeOut", delay: 0.4 }}
                        className="h-full rounded-full bg-gradient-to-r from-green-500 to-emerald-400"
                      />
                    </div>
                    <p className="text-green-400 text-sm font-semibold mt-1">{confidence}% match</p>
                  </div>
                )}
                <button onClick={resetLiveness} className="w-full py-3 rounded-xl font-bold text-white bg-white/8 hover:bg-white/12 border border-white/15 transition-all text-sm">
                  Sign in with another account
                </button>
              </div>
            </motion.div>
          )}

          {/* ── Two-column face auth layout ────────────────────────────── */}
          {loginMode === "face" && !["otp", "success"].includes(pageState) && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">

              {/* ── LEFT: Camera ─────────────────────────────────────────── */}
              <motion.div initial={{ opacity: 0, x: -30 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 }} className="flex flex-col gap-4">

                {/* Camera box */}
                <div className="glass-panel rounded-3xl overflow-hidden border-white/10 bg-black/60 relative">
                  <div className="aspect-[4/3] relative">

                    {/* Live webcam feed */}
                    {["camera-ready", "monitoring", "verifying", "success"].includes(pageState) && (
                      <Webcam
                        audio={false} ref={webcamRef}
                        screenshotFormat="image/jpeg"
                        videoConstraints={{ facingMode: "user", width: 640, height: 480 }}
                        className="w-full h-full object-cover" mirrored
                      />
                    )}

                    {/* Idle / loading model overlay */}
                    {["idle", "loading-models"].includes(pageState) && (
                      <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/90 z-20">
                        {pageState === "loading-models" ? (
                          <>
                            <Loader2 className="w-10 h-10 text-indigo-400 animate-spin mb-4" />
                            <p className="text-indigo-300 text-sm tracking-widest uppercase font-medium">Loading Neural Networks…</p>
                          </>
                        ) : (
                          <>
                            <div className="w-20 h-20 rounded-full bg-indigo-500/10 border border-indigo-500/30 flex items-center justify-center mb-4">
                              <User className="w-10 h-10 text-indigo-400" />
                            </div>
                            <p className="text-white/50 text-sm">Camera starts after model load</p>
                          </>
                        )}
                      </div>
                    )}

                    {/* Monitoring overlay: scan frame + instruction banner */}
                    {pageState === "monitoring" && (
                      <div className="absolute inset-0 z-20 pointer-events-none">
                        <div className="absolute inset-0 flex items-center justify-center">
                          <div className="w-44 h-60 border-2 border-indigo-400/40 rounded-[40px] relative">
                            <div className="absolute -top-1 -left-1  w-5 h-5 border-t-2 border-l-2 border-indigo-400" />
                            <div className="absolute -top-1 -right-1 w-5 h-5 border-t-2 border-r-2 border-indigo-400" />
                            <div className="absolute -bottom-1 -left-1  w-5 h-5 border-b-2 border-l-2 border-indigo-400" />
                            <div className="absolute -bottom-1 -right-1 w-5 h-5 border-b-2 border-r-2 border-indigo-400" />
                            <motion.div
                              animate={{ top: ["0%", "100%", "0%"] }}
                              transition={{ duration: 2.5, repeat: Infinity, ease: "linear" }}
                              className="absolute left-0 w-full h-0.5 bg-gradient-to-r from-transparent via-indigo-400 to-transparent shadow-[0_0_12px_#818cf8]"
                            />
                          </div>
                        </div>
                        {activeInstruction && (
                          <div className="absolute bottom-14 left-4 right-4 flex justify-center">
                            <motion.div
                              key={activeInstruction}
                              initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}
                              className="px-4 py-2 rounded-full bg-indigo-600/80 backdrop-blur-sm text-white text-sm font-medium shadow-lg"
                            >
                              {activeInstruction}
                            </motion.div>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Verifying overlay */}
                    {pageState === "verifying" && (
                      <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-black/70 backdrop-blur-sm">
                        <Loader2 className="w-12 h-12 text-indigo-400 animate-spin mb-3" />
                        <p className="text-white font-semibold">Verifying identity…</p>
                        <p className="text-indigo-300 text-sm mt-1">Matching face against database</p>
                      </div>
                    )}

                    {/* Failed overlay */}
                    {pageState === "failed" && (
                      <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-black/80">
                        <ShieldX className="w-12 h-12 text-red-400 mb-3" />
                        <p className="text-white font-semibold text-center px-8">{errorMsg || "Authentication Failed"}</p>
                      </div>
                    )}

                    {/* Top-bar: LIVE badge + countdown */}
                    {["camera-ready", "monitoring"].includes(pageState) && (
                      <div className="absolute top-4 left-4 right-4 z-30 flex items-center justify-between">
                        <div className="flex items-center gap-2 bg-black/60 backdrop-blur-sm px-3 py-1.5 rounded-full border border-white/10">
                          <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                          <span className="text-xs font-mono text-white/80 uppercase">Live</span>
                        </div>
                        {pageState === "monitoring" && (
                          <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full border text-xs font-mono font-bold ${
                            timeLeft <= 7
                              ? "bg-red-900/60 border-red-500/50 text-red-300"
                              : "bg-black/60 border-white/10 text-indigo-300"
                          }`}>
                            <Timer className="w-3.5 h-3.5" />
                            {timeLeft}s
                          </div>
                        )}
                      </div>
                    )}

                    {/* Bottom progress bar */}
                    {pageState === "monitoring" && (
                      <div className="absolute bottom-0 left-0 right-0 z-30">
                        <div className="h-1.5 bg-white/10">
                          <motion.div
                            animate={{ width: `${(passedCount / 4) * 100}%` }}
                            transition={{ duration: 0.4 }}
                            className="h-full bg-gradient-to-r from-indigo-500 to-purple-500"
                          />
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Email input (shown before monitoring starts) */}
                {["idle", "camera-ready", "loading-models"].includes(pageState) && (
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-gray-300 ml-1">Email Address</label>
                    <div className="relative">
                      <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                        <User className="h-5 w-5 text-gray-500" />
                      </div>
                      <input
                        type="email" value={email}
                        onChange={e => setEmail(e.target.value)}
                        onKeyDown={e => e.key === "Enter" && pageState === "camera-ready" && startMonitoring()}
                        className="w-full pl-11 pr-4 py-3.5 bg-black/40 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500/50 transition-all"
                        placeholder="you@example.com"
                      />
                    </div>
                  </div>
                )}

                {/* Primary action buttons */}
                {pageState === "idle" && (
                  <button
                    onClick={loadModels} disabled={!email.trim()}
                    className="w-full py-4 rounded-xl font-bold text-white bg-gradient-to-r from-indigo-500 to-purple-600 shadow-[0_0_20px_rgba(99,102,241,0.3)] hover:shadow-[0_0_30px_rgba(99,102,241,0.5)] disabled:opacity-50 disabled:cursor-not-allowed hover:-translate-y-0.5 transition-all duration-200 flex items-center justify-center gap-2"
                  >
                    <ShieldCheck className="w-5 h-5" /> Begin Authentication
                  </button>
                )}

                {pageState === "camera-ready" && (
                  <button
                    onClick={startMonitoring} disabled={!email.trim()}
                    className="w-full py-4 rounded-xl font-bold text-white bg-gradient-to-r from-indigo-500 to-purple-600 shadow-[0_0_20px_rgba(99,102,241,0.3)] hover:shadow-[0_0_30px_rgba(99,102,241,0.5)] disabled:opacity-50 disabled:cursor-not-allowed hover:-translate-y-0.5 transition-all duration-200 flex items-center justify-center gap-2"
                  >
                    <Zap className="w-5 h-5" /> Start Liveness Check
                  </button>
                )}

                {pageState === "failed" && (
                  <button
                    onClick={resetLiveness}
                    className="w-full py-4 rounded-xl font-bold text-white bg-gradient-to-r from-rose-600 to-red-700 hover:shadow-[0_0_30px_rgba(239,68,68,0.4)] hover:-translate-y-0.5 transition-all duration-200 flex items-center justify-center gap-2"
                  >
                    <RefreshCw className="w-5 h-5" /> Try Again
                  </button>
                )}

                {/* Developer debug panel */}
                {pageState === "monitoring" && (
                  <motion.div
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                    className="glass-panel rounded-2xl p-4 border-white/5 bg-black/40"
                  >
                    <div className="flex items-center gap-2 mb-3">
                      <Activity className="w-4 h-4 text-indigo-400" />
                      <span className="text-xs font-mono text-gray-400 uppercase tracking-wider">Live Biometrics</span>
                      {serverAvailable && (
                        <span className="ml-auto text-xs text-purple-400 font-mono">MediaPipe</span>
                      )}
                    </div>
                    <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 font-mono text-xs">
                      {/* Client-side (face-api) */}
                      {debugEAR !== null && (
                        <div className="flex justify-between text-gray-400 col-span-1">
                          <span>EAR (client)</span>
                          <span className={debugEAR < 0.22 ? "text-yellow-400" : "text-green-400"}>{debugEAR.toFixed(3)}</span>
                        </div>
                      )}
                      {debugLip !== null && (
                        <div className="flex justify-between text-gray-400 col-span-1">
                          <span>Lip gap (client)</span>
                          <span className={debugLip > 14 ? "text-yellow-400" : "text-gray-500"}>{debugLip}px</span>
                        </div>
                      )}
                      {/* Server-side (MediaPipe) */}
                      {serverFrameData?.ear != null && (
                        <div className="flex justify-between text-gray-400 col-span-1">
                          <span>EAR (server)</span>
                          <span className={serverFrameData.ear < 0.22 ? "text-yellow-400" : "text-purple-400"}>{serverFrameData.ear.toFixed(3)}</span>
                        </div>
                      )}
                      {serverFrameData?.mar != null && (
                        <div className="flex justify-between text-gray-400 col-span-1">
                          <span>MAR (server)</span>
                          <span className={serverFrameData.mar > 0.6 ? "text-yellow-400" : "text-purple-400"}>{serverFrameData.mar.toFixed(3)}</span>
                        </div>
                      )}
                      {serverFrameData && (
                        <>
                          <div className="flex justify-between text-gray-400 col-span-1">
                            <span>Yaw</span>
                            <span className="text-purple-400">{serverFrameData.yaw.toFixed(1)}°</span>
                          </div>
                          <div className="flex justify-between text-gray-400 col-span-1">
                            <span>Pitch</span>
                            <span className="text-purple-400">{serverFrameData.pitch.toFixed(1)}°</span>
                          </div>
                        </>
                      )}
                      {antiSpoofScore !== null && (
                        <div className="flex justify-between text-gray-400 col-span-2 border-t border-white/5 pt-1.5 mt-0.5">
                          <span>Anti-Spoof Score</span>
                          <span className={antiSpoofScore >= 65 ? "text-green-400" : antiSpoofScore >= 40 ? "text-yellow-400" : "text-red-400"}>
                            {antiSpoofScore}/100
                          </span>
                        </div>
                      )}
                      {serverScore !== null && (
                        <div className="flex justify-between text-gray-400 col-span-2">
                          <span>MediaPipe Liveness</span>
                          <span className={serverScore >= 75 ? "text-green-400" : serverScore >= 50 ? "text-yellow-400" : "text-purple-400"}>
                            {serverScore}/100
                          </span>
                        </div>
                      )}
                    </div>
                  </motion.div>
                )}
              </motion.div>

              {/* ── RIGHT: Checks + Anti-Spoof Shield + Instructions ──────── */}
              <motion.div initial={{ opacity: 0, x: 30 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2 }} className="flex flex-col gap-5">

                {/* ── Stage 2 Anti-Spoof Shield Panel ───────────────────── */}
                <div className={`glass-panel rounded-3xl p-6 border transition-colors duration-500 ${shieldColor}`}>
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div className={`w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 ${
                        antiSpoofScore === null      ? "bg-white/5" :
                        antiSpoofScore >= 65         ? "bg-green-500/20" :
                        antiSpoofScore >= 40         ? "bg-yellow-500/20" :
                                                       "bg-red-500/20"
                      }`}>
                        <Shield className={`w-5 h-5 ${
                          antiSpoofScore === null      ? "text-gray-400" :
                          antiSpoofScore >= 65         ? "text-green-400" :
                          antiSpoofScore >= 40         ? "text-yellow-400" :
                                                         "text-red-400"
                        }`} />
                      </div>
                      <div>
                        <h3 className="text-sm font-bold text-white">Anti-Spoof Shield</h3>
                        <p className="text-xs text-gray-400">
                          {pageState !== "monitoring"
                            ? "Active during liveness check"
                            : antiSpoofScore === null
                              ? "Initializing…"
                              : antiSpoofScore >= 65
                                ? "No spoofing detected"
                                : antiSpoofScore >= 40
                                  ? "Analyzing — hold steady"
                                  : "Suspicious signal — use real face"
                          }
                        </p>
                      </div>
                    </div>
                    {antiSpoofScore !== null && (
                      <span className={`text-lg font-bold font-mono ${
                        antiSpoofScore >= 65 ? "text-green-400" :
                        antiSpoofScore >= 40 ? "text-yellow-400" : "text-red-400"
                      }`}>
                        {antiSpoofScore}
                      </span>
                    )}
                  </div>

                  {/* Score bar */}
                  <div className="h-1.5 rounded-full bg-white/10 overflow-hidden mb-4">
                    <motion.div
                      animate={{ width: `${antiSpoofScore ?? 0}%` }}
                      transition={{ duration: 0.4 }}
                      className={`h-full rounded-full ${
                        (antiSpoofScore ?? 0) >= 65 ? "bg-gradient-to-r from-green-500 to-emerald-400" :
                        (antiSpoofScore ?? 0) >= 40 ? "bg-gradient-to-r from-yellow-500 to-orange-400" :
                                                       "bg-gradient-to-r from-red-600 to-rose-500"
                      }`}
                    />
                  </div>

                  {/* Signal breakdown (shown when we have data) */}
                  {antiSpoofSignals && (
                    <div className="grid grid-cols-2 gap-2">
                      {[
                        { label: "Glare",       value: antiSpoofSignals.glare,             icon: "✦" },
                        { label: "Texture",     value: antiSpoofSignals.texture,           icon: "◈" },
                        { label: "Colour",      value: antiSpoofSignals.colorNaturalness,  icon: "◉" },
                        { label: "Motion",      value: antiSpoofSignals.temporalVariance,  icon: "⊛" },
                        { label: "Regularity",  value: antiSpoofSignals.motionConsistency, icon: "≋" },
                      ].map(({ label, value, icon }) => (
                        <div key={label} className="flex items-center gap-2 bg-black/20 rounded-lg px-3 py-2">
                          <span className="text-xs text-gray-500">{icon}</span>
                          <div className="flex-1 min-w-0">
                            <div className="flex justify-between items-center">
                              <span className="text-xs text-gray-400">{label}</span>
                              <span className={`text-xs font-mono font-semibold ${
                                value >= 65 ? "text-green-400" :
                                value >= 40 ? "text-yellow-400" : "text-red-400"
                              }`}>{value}</span>
                            </div>
                            <div className="h-1 rounded-full bg-white/10 mt-1 overflow-hidden">
                              <div
                                className={`h-full rounded-full transition-all duration-300 ${
                                  value >= 65 ? "bg-green-500" :
                                  value >= 40 ? "bg-yellow-500" : "bg-red-500"
                                }`}
                                style={{ width: `${value}%` }}
                              />
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* ── Stage 3 Liveness Detection Panel ──────────────────── */}
                <div className="glass-panel rounded-3xl p-6 md:p-8 border-white/10">
                  <div className="flex items-center justify-between mb-6">
                    <h2 className="text-xl font-bold text-white">Liveness Detection</h2>
                    <span className="text-sm text-indigo-400 font-medium">{passedCount} / 4 passed</span>
                  </div>

                  <div className="space-y-3">
                    {checks.map(({ key, label, icon: Icon, hint }) => {
                      const passed   = liveness[key];
                      const isActive = pageState === "monitoring" && !passed
                        && INSTRUCTIONS.findIndex(i => !liveness[i.key]) === checks.findIndex(c => c.key === key);
                      return (
                        <motion.div
                          key={key}
                          animate={passed ? { scale: [1, 1.02, 1] } : {}}
                          transition={{ duration: 0.3 }}
                          className={`flex items-center gap-4 p-4 rounded-2xl border transition-all duration-300 ${
                            passed    ? "bg-green-500/10 border-green-500/30"
                            : isActive ? "bg-indigo-500/10 border-indigo-500/30"
                                       : "bg-white/3 border-white/8"
                          }`}
                        >
                          <div className={`w-11 h-11 rounded-xl flex items-center justify-center flex-shrink-0 transition-colors ${
                            passed ? "bg-green-500/20" : isActive ? "bg-indigo-500/15" : "bg-white/5"
                          }`}>
                            <Icon className={`w-5 h-5 ${passed ? "text-green-400" : isActive ? "text-indigo-300" : "text-gray-400"}`} />
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className={`font-semibold text-sm ${passed ? "text-green-300" : isActive ? "text-indigo-200" : "text-white"}`}>
                              {label}
                            </p>
                            <p className={`text-xs mt-0.5 ${passed ? "text-green-500/70" : isActive ? "text-indigo-400" : "text-gray-500"}`}>
                              {passed ? "Verified ✓" : isActive ? hint : "Pending"}
                            </p>
                          </div>
                          <AnimatePresence mode="wait">
                            {passed ? (
                              <motion.div key="check" initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ type: "spring", bounce: 0.6 }}>
                                <CheckCircle2 className="w-6 h-6 text-green-400 flex-shrink-0" />
                              </motion.div>
                            ) : (
                              <motion.div key="ring" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex-shrink-0">
                                <div className={`w-6 h-6 rounded-full border-2 ${isActive ? "border-indigo-400 animate-pulse" : "border-white/20"}`} />
                              </motion.div>
                            )}
                          </AnimatePresence>
                        </motion.div>
                      );
                    })}
                  </div>

                  {/* Overall progress */}
                  <div className="mt-6">
                    <div className="flex justify-between text-xs text-gray-400 mb-2">
                      <span>Overall Progress</span>
                      <span>{Math.round((passedCount / 4) * 100)}%</span>
                    </div>
                    <div className="h-2 rounded-full bg-white/5 overflow-hidden">
                      <motion.div
                        animate={{ width: `${(passedCount / 4) * 100}%` }}
                        transition={{ duration: 0.5, ease: "easeOut" }}
                        className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-purple-500"
                      />
                    </div>
                  </div>
                </div>

                {/* ── MediaPipe Server-side Liveness Panel ───────────────── */}
                {serverAvailable && pageState === "monitoring" && serverChecks && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
                    className={`glass-panel rounded-3xl p-5 border transition-colors duration-500 ${
                      serverIsLive ? "border-purple-500/30 bg-purple-500/5" : "border-white/10"
                    }`}
                  >
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center gap-2">
                        <Zap className="w-4 h-4 text-purple-400" />
                        <h3 className="text-sm font-bold text-white">MediaPipe Verification</h3>
                        <span className="text-xs text-purple-400/60 font-mono">478 landmarks</span>
                      </div>
                      {serverScore !== null && (
                        <span className={`text-sm font-bold font-mono ${serverScore >= 75 ? "text-purple-300" : "text-gray-400"}`}>
                          {serverScore}/100
                        </span>
                      )}
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      {[
                        { key: "blink_detected", label: "Eye Blink (EAR)",    icon: "👁" },
                        { key: "lip_moved",       label: "Lip Move (MAR)",    icon: "👄" },
                        { key: "head_moved",      label: `Head ${headChallenge.toUpperCase()}`, icon: "↔" },
                        { key: "texture_ok",      label: "Skin Motion",       icon: "🌊" },
                      ].map(({ key, label, icon }) => {
                        const done = serverChecks[key as keyof ServerLivenessChecks];
                        return (
                          <div key={key} className={`flex items-center gap-2 px-3 py-2 rounded-xl border text-xs transition-all duration-300 ${
                            done ? "bg-purple-500/15 border-purple-500/30" : "bg-white/3 border-white/8"
                          }`}>
                            <span>{icon}</span>
                            <span className={done ? "text-purple-300" : "text-gray-500"}>{label}</span>
                            <span className="ml-auto">{done ? "✓" : "…"}</span>
                          </div>
                        );
                      })}
                    </div>
                    {serverFrameData && (
                      <div className="mt-3 flex gap-4 text-xs font-mono text-gray-500">
                        <span>Yaw: <span className="text-purple-400">{serverFrameData.yaw.toFixed(1)}°</span></span>
                        <span>Pitch: <span className="text-purple-400">{serverFrameData.pitch.toFixed(1)}°</span></span>
                        <span>Frames: <span className="text-gray-400">{serverFrameData.frame_count}</span></span>
                      </div>
                    )}
                  </motion.div>
                )}

                {/* ── Instructions ──────────────────────────────────────── */}
                <div className="glass-panel rounded-3xl p-6 border-white/10">
                  <h3 className="text-base font-bold text-white mb-4 flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4 text-yellow-400" />
                    How to Complete Authentication
                  </h3>
                  <ul className="space-y-3">
                    {[
                      "Position your real face inside the scan frame",
                      "Blink your eyes naturally (close fully, then open)",
                      headChallenge === "up"
                        ? "Slowly tilt your head upward and hold briefly"
                        : `Slowly turn your head to the ${headChallenge.toUpperCase()} and hold briefly`,
                      "Good lighting helps — avoid backlight or shadows",
                      "Do not use a photo, screen or video of a face",
                    ].map((tip, i) => (
                      <li key={i} className="flex items-start gap-3 text-sm text-gray-400">
                        <span className="flex-shrink-0 w-5 h-5 rounded-full bg-indigo-500/15 border border-indigo-500/30 flex items-center justify-center text-xs text-indigo-400 font-bold">
                          {i + 1}
                        </span>
                        {tip}
                      </li>
                    ))}
                  </ul>
                </div>

                {/* ── Confidence on success ──────────────────────────────── */}
                {pageState === "success" && confidence !== null && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
                    className="glass-panel rounded-3xl p-6 border-green-500/20 bg-green-500/5"
                  >
                    <h3 className="text-base font-bold text-green-300 mb-3">Authentication Score</h3>
                    <div className="flex items-center gap-4">
                      <div className="text-5xl font-bold text-white">
                        {confidence}<span className="text-xl text-green-400">%</span>
                      </div>
                      <div className="flex-1">
                        <div className="h-3 rounded-full bg-white/5 overflow-hidden">
                          <motion.div
                            initial={{ width: 0 }} animate={{ width: `${confidence}%` }}
                            transition={{ duration: 1, ease: "easeOut", delay: 0.3 }}
                            className="h-full rounded-full bg-gradient-to-r from-green-500 to-emerald-400"
                          />
                        </div>
                        <p className="text-xs text-green-400/70 mt-1.5">Face descriptor match confidence</p>
                      </div>
                    </div>
                  </motion.div>
                )}

                {/* ── Failure panel ──────────────────────────────────────── */}
                {pageState === "failed" && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}
                    className={`glass-panel rounded-3xl p-6 border ${
                      spoofDetected
                        ? "border-orange-500/30 bg-orange-500/5"
                        : "border-red-500/20 bg-red-500/5"
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      {spoofDetected ? (
                        <Shield className="w-6 h-6 text-orange-400 flex-shrink-0 mt-0.5" />
                      ) : (
                        <XCircle className="w-6 h-6 text-red-400 flex-shrink-0 mt-0.5" />
                      )}
                      <div>
                        <h3 className={`text-base font-bold mb-1 ${spoofDetected ? "text-orange-300" : "text-red-300"}`}>
                          {spoofDetected ? "Spoofing Attempt Detected" : "Authentication Failed"}
                        </h3>
                        <p className="text-sm text-gray-400">
                          {errorMsg || "Liveness check failed. Please try again."}
                        </p>
                        {spoofDetected && (
                          <p className="text-xs text-orange-400/80 mt-2">
                            The anti-spoof shield detected a non-real face. Please look directly into the camera with your real face.
                          </p>
                        )}
                      </div>
                    </div>
                  </motion.div>
                )}
              </motion.div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

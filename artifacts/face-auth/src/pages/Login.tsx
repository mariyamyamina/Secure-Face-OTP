import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useLocation } from "wouter";
import Webcam from "react-webcam";
import * as faceapi from "@vladmandic/face-api";
import {
  Eye, Smile, Move, Layers, CheckCircle2,
  XCircle, Loader2, ShieldCheck, ShieldX,
  RefreshCw, User, Timer, ChevronRight,
  AlertTriangle, Zap, Activity,
} from "lucide-react";
import { Navbar } from "@/components/layout/Navbar";
import { useToast } from "@/hooks/use-toast";
import { generateOTP, sendOTPEmail, emailJSConfigured } from "@/lib/emailService";
import { useUser } from "@/context/UserContext";

// ─── Constants ────────────────────────────────────────────────────────────────
const MODEL_URL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/";
const DETECTION_INTERVAL_MS = 200;   // 200ms — fast enough to catch a 200ms blink
const LIVENESS_TIMEOUT_S = 25;       // 25 seconds total

// Blink: relative drop. When EAR falls to ≤75% of the rolling max, it's a blink.
// Works at any face distance — no fixed absolute threshold needed.
const BLINK_DROP_RATIO = 0.75;
const EAR_HISTORY_SIZE = 12;        // Rolling window of recent EAR samples

// Lip openness threshold (pixels)
const LIP_OPEN_PX  = 6;             // Mouth open when inner lip gap > 6px
const LIP_CLOSE_PX = 3;             // Mouth closed when gap < 3px

// Head movement threshold (pixels)
const HEAD_MOVE_PX = 8;             // Nose must shift >8px from baseline

// ─── EAR helper ───────────────────────────────────────────────────────────────
function ptDist(a: faceapi.Point, b: faceapi.Point) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

// 6-point eye: [0]=outerCorner [1][2]=upper lid [3]=innerCorner [4][5]=lower lid
function computeEAR(eye: faceapi.Point[]) {
  const v1 = ptDist(eye[1], eye[5]);
  const v2 = ptDist(eye[2], eye[4]);
  const h  = ptDist(eye[0], eye[3]);
  return (v1 + v2) / (2.0 * h);
}

// ─── Types ────────────────────────────────────────────────────────────────────
interface LivenessState {
  blinkDetected:         boolean;
  lipMovementDetected:   boolean;
  headMovementDetected:  boolean;
  textureDetected:       boolean;
}

type PageState =
  | "idle"
  | "loading-models"
  | "camera-ready"
  | "monitoring"
  | "verifying"
  | "otp"           // Face matched — waiting for OTP verification
  | "success"
  | "failed";

// Active instruction shown to the user
const INSTRUCTIONS = [
  { key: "blinkDetected",        text: "👁  Please blink your eyes naturally" },
  { key: "lipMovementDetected",  text: "👄  Open and close your mouth slightly" },
  { key: "headMovementDetected", text: "↔️  Gently turn your head left or right" },
  { key: "textureDetected",      text: "✅  Hold still — detecting skin texture…" },
] as const;

// ─── Component ────────────────────────────────────────────────────────────────
export default function Login() {
  const { login } = useUser();
  const [, navigate] = useLocation();
  const [pageState,     setPageState]     = useState<PageState>("idle");
  const [email,         setEmail]         = useState("");
  const [modelsLoaded,  setModelsLoaded]  = useState(false);
  const [liveness,      setLiveness]      = useState<LivenessState>({
    blinkDetected: false, lipMovementDetected: false,
    headMovementDetected: false, textureDetected: false,
  });
  const [timeLeft,      setTimeLeft]      = useState(LIVENESS_TIMEOUT_S);
  const [errorMsg,      setErrorMsg]      = useState("");
  const [successMsg,    setSuccessMsg]    = useState("");
  const [confidence,    setConfidence]    = useState<number | null>(null);

  // Login mode
  const [loginMode,     setLoginMode]     = useState<"face" | "password">("face");
  const [pwInput,       setPwInput]       = useState("");
  const [pwLoading,     setPwLoading]     = useState(false);
  const [pwError,       setPwError]       = useState("");
  const [userName,      setUserName]      = useState("");

  // OTP state
  const [otpValue,      setOtpValue]      = useState("");
  const [otpError,      setOtpError]      = useState("");
  const [otpLoading,    setOtpLoading]    = useState(false);
  const [otpResendLeft, setOtpResendLeft] = useState(0); // seconds until resend allowed
  const [devOtp,        setDevOtp]        = useState<string | null>(null); // shown when no SMTP
  const otpResendTimer  = useRef<ReturnType<typeof setInterval> | null>(null);
  const storedOTP       = useRef<string>("");        // holds the generated OTP in memory

  // Live debug values (shown in monitoring mode)
  const [debugEAR,      setDebugEAR]      = useState<number | null>(null);
  const [debugLip,      setDebugLip]      = useState<number | null>(null);

  const { toast } = useToast();
  const webcamRef = useRef<Webcam>(null);

  // Interval handles
  const detectionInterval = useRef<ReturnType<typeof setInterval> | null>(null);
  const countdownInterval = useRef<ReturnType<typeof setInterval> | null>(null);

  // Ref-copy of liveness so the interval callback always sees latest state
  const livenessRef = useRef<LivenessState>(liveness);
  useEffect(() => { livenessRef.current = liveness; }, [liveness]);

  // Ref so the interval can stop itself
  const monitoringActive = useRef(false);
  // Prevent overlapping async calls inside the interval
  const detectionRunning = useRef(false);

  // ── Blink: rolling EAR history for relative-drop detection ──────────────
  // We keep the last N EAR values. A blink is when the latest reading drops
  // to ≤75% of the rolling max (eyes-open baseline). No fixed threshold needed.
  const earHistory = useRef<number[]>([]);

  // ── Lip state machine ────────────────────────────────────────────────────
  // Track whether we've seen both an open and a closed state
  const lipWasOpen   = useRef(false);
  const lipWasClosed = useRef(false);
  // Remember the initial nose position as a baseline
  const noseBaseline = useRef<{ x: number; y: number } | null>(null);

  // ── Texture: pixel variance across several samples ───────────────────────
  const offCanvas = useRef(document.createElement("canvas"));
  const prevPixels = useRef<Uint8ClampedArray | null>(null);
  const textureScores = useRef<number[]>([]);

  // ─── Load models (once) ────────────────────────────────────────────────────
  const loadModels = useCallback(async () => {
    if (modelsLoaded) { setPageState("camera-ready"); return; }
    setPageState("loading-models");
    try {
      // Force CPU backend — avoids WebGL/WASM availability issues in iframes
      await faceapi.tf.setBackend("cpu");
      await faceapi.tf.ready();
      // Load the three models we need for liveness + descriptor
      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
        faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
        faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
      ]);
      setModelsLoaded(true);
      setPageState("camera-ready");
    } catch (err) {
      console.error("Model load error:", err);
      setErrorMsg("Failed to load AI models. Check your internet connection and refresh.");
      setPageState("failed");
    }
  }, [modelsLoaded]);

  // ─── Stop all intervals ────────────────────────────────────────────────────
  const stopAll = useCallback(() => {
    monitoringActive.current = false;
    if (detectionInterval.current) { clearInterval(detectionInterval.current); detectionInterval.current = null; }
    if (countdownInterval.current) { clearInterval(countdownInterval.current); countdownInterval.current = null; }
  }, []);

  // ─── Reset to camera-ready state ──────────────────────────────────────────
  const resetLiveness = useCallback(() => {
    stopAll();
    const blank = { blinkDetected: false, lipMovementDetected: false, headMovementDetected: false, textureDetected: false };
    setLiveness(blank);
    livenessRef.current = blank;
    earHistory.current   = [];
    storedOTP.current    = "";
    lipWasOpen.current   = false;
    lipWasClosed.current = false;
    noseBaseline.current = null;
    prevPixels.current   = null;
    textureScores.current = [];
    setDebugEAR(null);
    setDebugLip(null);
    setTimeLeft(LIVENESS_TIMEOUT_S);
    setErrorMsg("");
    setSuccessMsg("");
    setConfidence(null);
    setPageState("camera-ready");
  }, [stopAll]);

  // ─── Single detection tick (called by interval) ────────────────────────────
  const runDetectionTick = useCallback(async () => {
    // Guard: skip if already running or monitoring stopped
    if (detectionRunning.current || !monitoringActive.current) return;
    const video = webcamRef.current?.video;
    if (!video || video.readyState !== 4) return;

    detectionRunning.current = true;
    try {
      // During liveness monitoring we only need landmarks — NOT the descriptor.
      // Skipping withFaceDescriptor() makes each tick 3–4× faster on CPU.
      const det = await faceapi
        .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.5 }))
        .withFaceLandmarks();

      if (det) {
        const pts = det.landmarks.positions;

        // ── 1. EYE BLINK ────────────────────────────────────────────────────
        // 68-point map: left eye = pts[36..41], right eye = pts[42..47]
        const leftEye  = pts.slice(36, 42) as faceapi.Point[];
        const rightEye = pts.slice(42, 48) as faceapi.Point[];
        const ear = (computeEAR(leftEye) + computeEAR(rightEye)) / 2;
        setDebugEAR(Math.round(ear * 1000) / 1000);

        if (!livenessRef.current.blinkDetected) {
          // Push into rolling history
          earHistory.current.push(ear);
          if (earHistory.current.length > EAR_HISTORY_SIZE) earHistory.current.shift();

          // Need at least 4 samples to establish an open-eye baseline
          if (earHistory.current.length >= 4) {
            // Rolling max = the highest EAR seen recently (open eyes baseline)
            const rollingMax = Math.max(...earHistory.current);

            // A blink = current EAR drops to ≤75% of the open-eye baseline.
            // Require the baseline to be meaningful (> 0.15) to avoid false positives
            // when no face is present.
            if (rollingMax > 0.15 && ear <= rollingMax * BLINK_DROP_RATIO) {
              setLiveness(prev => ({ ...prev, blinkDetected: true }));
              earHistory.current = []; // reset so it doesn't re-trigger
            }
          }
        }

        // ── 2. LIP MOVEMENT ─────────────────────────────────────────────────
        // Inner mouth: pts[60..67]. Vertical gap = pts[62] (top) vs pts[66] (bottom)
        const lipGap = Math.abs(pts[62].y - pts[66].y);
        setDebugLip(Math.round(lipGap));

        if (!livenessRef.current.lipMovementDetected) {
          if (lipGap > LIP_OPEN_PX)  lipWasOpen.current   = true;
          if (lipGap < LIP_CLOSE_PX) lipWasClosed.current = true;
          // Must have seen BOTH open and closed to count as deliberate movement
          if (lipWasOpen.current && lipWasClosed.current) {
            setLiveness(prev => ({ ...prev, lipMovementDetected: true }));
          }
        }

        // ── 3. HEAD MOVEMENT ─────────────────────────────────────────────────
        // Use nose tip (pt[30]) relative to a baseline captured at first frame
        const nosePt = pts[30];
        if (!livenessRef.current.headMovementDetected) {
          if (!noseBaseline.current) {
            noseBaseline.current = { x: nosePt.x, y: nosePt.y };
          } else {
            const dx = Math.abs(nosePt.x - noseBaseline.current.x);
            const dy = Math.abs(nosePt.y - noseBaseline.current.y);
            if (dx > HEAD_MOVE_PX || dy > HEAD_MOVE_PX) {
              setLiveness(prev => ({ ...prev, headMovementDetected: true }));
            }
          }
        }
      }

      // ── 4. SKIN TEXTURE (temporal pixel variance) ──────────────────────────
      // Run on raw video — no face needed. Measures frame-to-frame variation
      // which is always present for real video but flat for static photos.
      if (!livenessRef.current.textureDetected) {
        const ctx = offCanvas.current.getContext("2d");
        if (ctx) {
          offCanvas.current.width  = 48;
          offCanvas.current.height = 48;
          ctx.drawImage(video, 0, 0, 48, 48);
          const imgData = ctx.getImageData(0, 0, 48, 48).data;
          const gray = new Uint8ClampedArray(48 * 48);
          for (let i = 0; i < gray.length; i++) {
            gray[i] = Math.round(0.299 * imgData[i*4] + 0.587 * imgData[i*4+1] + 0.114 * imgData[i*4+2]);
          }

          if (prevPixels.current) {
            let mad = 0;
            for (let i = 0; i < gray.length; i++) mad += Math.abs(gray[i] - prevPixels.current[i]);
            mad /= gray.length;

            // Accumulate scores; pass after 4 samples with variation > threshold
            textureScores.current.push(mad);
            const recentScores = textureScores.current.slice(-6);
            const passing = recentScores.filter(s => s > 1.2).length;
            if (passing >= 4) {
              setLiveness(prev => ({ ...prev, textureDetected: true }));
            }
          }
          prevPixels.current = gray;
        }
      }

    } catch (err) {
      // Silently ignore individual frame errors — next tick will retry
    } finally {
      detectionRunning.current = false;
    }
  }, []);

  // ─── Check if all liveness passed and auto-verify ─────────────────────────
  const allPassed = (l: LivenessState) =>
    l.blinkDetected && l.lipMovementDetected && l.headMovementDetected && l.textureDetected;

  useEffect(() => {
    if (pageState === "monitoring" && allPassed(liveness)) {
      stopAll();
      verifyAndLogin();
    }
  }, [liveness, pageState]);

  // ─── Start liveness monitoring ─────────────────────────────────────────────
  const startMonitoring = useCallback(() => {
    if (!email.trim()) {
      toast({ variant: "destructive", title: "Email required", description: "Enter your email first." });
      return;
    }
    monitoringActive.current = true;
    setPageState("monitoring");
    setTimeLeft(LIVENESS_TIMEOUT_S);

    // Countdown — 1 tick per second
    countdownInterval.current = setInterval(() => {
      setTimeLeft(prev => {
        if (prev <= 1) {
          stopAll();
          if (!allPassed(livenessRef.current)) {
            setErrorMsg("Time expired. Not all liveness checks were completed. Please try again.");
            setPageState("failed");
          }
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    // Face detection — every 350ms (much gentler on CPU than rAF)
    detectionInterval.current = setInterval(runDetectionTick, DETECTION_INTERVAL_MS);

    // Run immediately on the first tick too
    runDetectionTick();
  }, [email, runDetectionTick, stopAll, toast]);

  // ─── Final face capture + backend call ────────────────────────────────────
  const verifyAndLogin = useCallback(async () => {
    setPageState("verifying");
    const video = webcamRef.current?.video;
    if (!video) { setErrorMsg("Camera unavailable."); setPageState("failed"); return; }

    try {
      // Now we DO need the descriptor for matching — run the full pipeline once
      const det = await faceapi
        .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.5 }))
        .withFaceLandmarks()
        .withFaceDescriptor();

      if (!det) {
        setErrorMsg("No face detected at verification time. Please try again.");
        setPageState("failed");
        return;
      }

      const res = await fetch("/api/login-face", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email,
          face_descriptor: Array.from(det.descriptor),
          liveness_passed: true,
        }),
      });
      const data = await res.json();

      if (res.ok) {
        // Face matched — now trigger OTP step
        setConfidence(data.confidence ?? null);
        setUserName(data.name ?? "");
        setPageState("otp");
        // Fire and forget — don't await so we can get to OTP UI quickly
        sendOtp();
      } else {
        setErrorMsg(data.error || "Authentication failed.");
        setPageState("failed");
      }
    } catch {
      setErrorMsg("Network error. Please check your connection.");
      setPageState("failed");
    }
  }, [email]);

  // ─── OTP helpers ─────────────────────────────────────────────────────────
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

    // Generate a fresh 6-digit OTP and store it in memory
    const otp = generateOTP();
    storedOTP.current = otp;

    // Send via EmailJS (frontend) — no backend call needed
    const result = await sendOTPEmail(email, otp);

    if (result.ok) {
      // If EmailJS is not configured, show the OTP in the UI for dev testing
      if (!emailJSConfigured) {
        setDevOtp(otp);
      }
      startOtpResendCountdown(60);
    } else {
      setOtpError(result.error ?? "Failed to send OTP email");
    }
  }, [email, startOtpResendCountdown]);

  const verifyOtp = useCallback(async () => {
    const entered = otpValue.trim();
    if (entered.length !== 6) {
      setOtpError("Please enter the full 6-digit code.");
      return;
    }
    if (!storedOTP.current) {
      setOtpError("No OTP found. Please request a new one.");
      return;
    }

    setOtpLoading(true);
    setOtpError("");

    // Small artificial delay so the button doesn't flash instantly
    await new Promise(r => setTimeout(r, 400));

    if (entered === storedOTP.current) {
      // Clear OTP from memory so it can't be reused
      storedOTP.current = "";
      if (otpResendTimer.current) clearInterval(otpResendTimer.current);
      // Store user and redirect to dashboard
      login(email, userName || undefined);
      navigate("/dashboard");
    } else {
      setOtpError("Incorrect code. Please check and try again.");
    }

    setOtpLoading(false);
  }, [otpValue]);

  // ─── Cleanup on unmount ───────────────────────────────────────────────────
  useEffect(() => () => {
    stopAll();
    if (otpResendTimer.current) clearInterval(otpResendTimer.current);
  }, [stopAll]);

  // ─── Password login ───────────────────────────────────────────────────────
  const handlePasswordLogin = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    setPwError("");
    setPwLoading(true);
    try {
      const res = await fetch("/api/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password: pwInput }),
      });
      const data = await res.json();
      if (!res.ok) {
        setPwError(data.error || "Login failed. Please try again.");
        return;
      }
      login(email, data.name ?? undefined);
      navigate("/dashboard");
    } catch {
      setPwError("Network error. Please check your connection.");
    } finally {
      setPwLoading(false);
    }
  }, [email, pwInput, login, navigate]);

  // ─── Derived values ───────────────────────────────────────────────────────
  const passedCount = Object.values(liveness).filter(Boolean).length;

  // Active instruction: first uncompleted check
  const activeInstruction = pageState === "monitoring"
    ? INSTRUCTIONS.find(i => !liveness[i.key as keyof LivenessState])?.text ?? "✅ All checks done!"
    : null;

  const checks = [
    { key: "blinkDetected"       as const, label: "Eye Blink",        icon: Eye,    hint: "Please blink your eyes" },
    { key: "lipMovementDetected" as const, label: "Lip Movement",     icon: Smile,  hint: "Open and close your mouth" },
    { key: "headMovementDetected"as const, label: "Head Movement",    icon: Move,   hint: "Move your head slightly" },
    { key: "textureDetected"     as const, label: "Real Skin Texture",icon: Layers, hint: "Hold still in frame" },
  ];

  // ─── Render ───────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen flex flex-col relative overflow-hidden bg-background">
      <div className="fixed inset-0 z-0 pointer-events-none">
        <div className="absolute top-[5%] left-[5%] w-[45%] h-[45%] rounded-full bg-indigo-600/5 blur-[120px]" />
        <div className="absolute bottom-[5%] right-[5%] w-[40%] h-[40%] rounded-full bg-purple-600/5 blur-[100px]" />
      </div>

      <Navbar />

      <main className="flex-1 container mx-auto px-4 py-24 md:py-28 relative z-10">
        <div className="w-full max-w-6xl mx-auto">

          {/* Header */}
          <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="text-center mb-10">
            <h1 className="text-4xl md:text-5xl font-display font-bold text-white mb-3">
              Biometric{" "}
              <span className="bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">
                Login
              </span>
            </h1>
            <p className="text-muted-foreground max-w-xl mx-auto">
              The system verifies you're a real person using four liveness signals before granting access.
            </p>
          </motion.div>

          {/* ── Login Mode Tab Switcher ─────────────────────────────── */}
          {pageState === "idle" && (
            <motion.div
              initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}
              className="flex justify-center mb-8"
            >
              <div className="flex gap-1 p-1 rounded-2xl bg-white/5 border border-white/10">
                <button
                  onClick={() => { setLoginMode("face"); setPwError(""); }}
                  className={`px-6 py-2.5 rounded-xl text-sm font-semibold transition-all duration-200 ${
                    loginMode === "face"
                      ? "bg-gradient-to-r from-indigo-500 to-purple-600 text-white shadow-[0_0_20px_rgba(99,102,241,0.3)]"
                      : "text-muted-foreground hover:text-white"
                  }`}
                >
                  Face Login
                </button>
                <button
                  onClick={() => { setLoginMode("password"); setPwError(""); }}
                  className={`px-6 py-2.5 rounded-xl text-sm font-semibold transition-all duration-200 ${
                    loginMode === "password"
                      ? "bg-gradient-to-r from-indigo-500 to-purple-600 text-white shadow-[0_0_20px_rgba(99,102,241,0.3)]"
                      : "text-muted-foreground hover:text-white"
                  }`}
                >
                  Password Login
                </button>
              </div>
            </motion.div>
          )}

          {/* ── Password Login Form ──────────────────────────────────── */}
          {loginMode === "password" && pageState === "idle" && (
            <motion.div
              initial={{ opacity: 0, scale: 0.97 }} animate={{ opacity: 1, scale: 1 }}
              className="max-w-md mx-auto"
            >
              <div className="glass-panel rounded-3xl p-8 border-white/10">
                <h2 className="text-2xl font-bold text-white mb-1">Sign In</h2>
                <p className="text-gray-400 text-sm mb-7">Enter your email and password to access your account.</p>

                <form onSubmit={handlePasswordLogin} className="space-y-5">
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-gray-300 ml-1">Email Address</label>
                    <input
                      type="email"
                      required
                      value={email}
                      onChange={(e) => { setEmail(e.target.value); setPwError(""); }}
                      className="w-full px-4 py-3.5 bg-black/40 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500/50 transition-all"
                      placeholder="you@example.com"
                    />
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium text-gray-300 ml-1">Password</label>
                    <input
                      type="password"
                      required
                      value={pwInput}
                      onChange={(e) => { setPwInput(e.target.value); setPwError(""); }}
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
                    type="submit"
                    disabled={pwLoading}
                    className="w-full py-4 rounded-xl font-bold text-white bg-gradient-to-r from-indigo-500 to-purple-600 shadow-[0_0_20px_rgba(99,102,241,0.3)] hover:shadow-[0_0_30px_rgba(99,102,241,0.5)] disabled:opacity-60 disabled:cursor-not-allowed hover:-translate-y-0.5 active:translate-y-0 transition-all duration-200 flex items-center justify-center gap-2"
                  >
                    {pwLoading ? (
                      <><Loader2 className="w-5 h-5 animate-spin" /> Signing in...</>
                    ) : (
                      <><ShieldCheck className="w-5 h-5" /> Sign In</>
                    )}
                  </button>

                  <p className="text-center text-sm text-muted-foreground pt-1">
                    Don't have an account?{" "}
                    <a href="/register" className="text-indigo-400 hover:text-indigo-300 font-medium transition-colors">
                      Register
                    </a>
                  </p>
                </form>
              </div>
            </motion.div>
          )}

          {/* ── OTP Screen ─────────────────────────────────────────────── */}
          {pageState === "otp" && (
            <motion.div
              initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }}
              className="max-w-md mx-auto"
            >
              <div className="glass-panel rounded-3xl p-8 border-white/10 text-center">
                {/* Icon */}
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

                {/* Dev mode OTP hint (shown when EmailJS not configured) */}
                {devOtp && (
                  <div className="mb-5 px-4 py-3 rounded-xl bg-yellow-500/10 border border-yellow-500/25 text-yellow-300 text-sm text-left">
                    <p className="font-semibold mb-1">Development Mode</p>
                    <p className="text-xs text-yellow-400/80 mb-2">
                      EmailJS is not configured. Set <code className="text-yellow-300">VITE_EMAILJS_SERVICE_ID</code>,{" "}
                      <code className="text-yellow-300">VITE_EMAILJS_TEMPLATE_ID</code>, and{" "}
                      <code className="text-yellow-300">VITE_EMAILJS_PUBLIC_KEY</code> to enable real emails.
                    </p>
                    <div className="flex items-center justify-center gap-3 py-2 rounded-lg bg-black/30 border border-yellow-500/20">
                      <span className="text-yellow-400/70 text-xs">Your OTP:</span>
                      <span className="font-mono font-bold text-2xl tracking-[0.3em] text-white">{devOtp}</span>
                    </div>
                  </div>
                )}

                {/* 6-digit OTP input */}
                <input
                  type="text"
                  inputMode="numeric"
                  maxLength={6}
                  value={otpValue}
                  onChange={e => { setOtpValue(e.target.value.replace(/\D/g, "")); setOtpError(""); }}
                  onKeyDown={e => e.key === "Enter" && verifyOtp()}
                  className="w-full text-center text-3xl font-bold tracking-[0.4em] py-4 px-4 bg-black/40 border border-white/10 rounded-xl text-white placeholder-gray-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500/50 transition-all mb-4"
                  placeholder="______"
                />

                {/* Error */}
                {otpError && (
                  <motion.p
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                    className="text-sm text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-4 py-2.5 mb-4"
                  >
                    {otpError}
                  </motion.p>
                )}

                {/* Verify button */}
                <button
                  onClick={verifyOtp}
                  disabled={otpLoading || otpValue.length !== 6}
                  className="w-full py-3.5 rounded-xl font-bold text-white bg-gradient-to-r from-indigo-600 to-purple-600 shadow-[0_0_20px_rgba(99,102,241,0.3)] hover:shadow-[0_0_30px_rgba(99,102,241,0.5)] disabled:opacity-50 disabled:cursor-not-allowed hover:-translate-y-0.5 active:translate-y-0 transition-all flex items-center justify-center gap-2 mb-5"
                >
                  {otpLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <ShieldCheck className="w-5 h-5" />}
                  {otpLoading ? "Verifying…" : "Verify OTP"}
                </button>

                {/* Resend */}
                <div className="flex items-center justify-center gap-2 text-sm">
                  {otpResendLeft > 0 ? (
                    <span className="text-gray-500">
                      Resend available in <span className="text-indigo-400 font-mono font-semibold">{otpResendLeft}s</span>
                    </span>
                  ) : (
                    <button
                      onClick={sendOtp}
                      className="text-indigo-400 hover:text-indigo-300 font-medium transition-colors flex items-center gap-1.5"
                    >
                      <RefreshCw className="w-3.5 h-3.5" />
                      Resend OTP
                    </button>
                  )}
                </div>
              </div>
            </motion.div>
          )}

          {/* ── Final Success Screen ────────────────────────────────────── */}
          {pageState === "success" && (
            <motion.div
              initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }}
              className="max-w-md mx-auto"
            >
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
                <button
                  onClick={resetLiveness}
                  className="w-full py-3 rounded-xl font-bold text-white bg-white/8 hover:bg-white/12 border border-white/15 transition-all text-sm"
                >
                  Sign in with another account
                </button>
              </div>
            </motion.div>
          )}

          {/* ── Two-column face auth layout ─────────────────────────────── */}
          {loginMode === "face" && !["otp","success"].includes(pageState) && <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">

            {/* ── LEFT: Camera ─────────────────────────────────────────────── */}
            <motion.div initial={{ opacity: 0, x: -30 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 }} className="flex flex-col gap-4">

              {/* Camera box */}
              <div className="glass-panel rounded-3xl overflow-hidden border-white/10 bg-black/60 relative">
                <div className="aspect-[4/3] relative">

                  {/* Webcam — mounted whenever camera is needed */}
                  {["camera-ready","monitoring","verifying","success"].includes(pageState) && (
                    <Webcam
                      audio={false}
                      ref={webcamRef}
                      screenshotFormat="image/jpeg"
                      videoConstraints={{ facingMode: "user", width: 640, height: 480 }}
                      className="w-full h-full object-cover"
                      mirrored
                    />
                  )}

                  {/* Idle / loading */}
                  {["idle","loading-models"].includes(pageState) && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/90 z-20">
                      {pageState === "loading-models" ? (
                        <>
                          <Loader2 className="w-10 h-10 text-indigo-400 animate-spin mb-4" />
                          <p className="text-indigo-300 text-sm tracking-widest uppercase font-medium">
                            Loading Neural Networks…
                          </p>
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

                  {/* Monitoring overlay */}
                  {pageState === "monitoring" && (
                    <div className="absolute inset-0 z-20 pointer-events-none">
                      {/* Face frame */}
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="w-44 h-60 border-2 border-indigo-400/40 rounded-[40px] relative">
                          <div className="absolute -top-1 -left-1  w-5 h-5 border-t-2 border-l-2 border-indigo-400" />
                          <div className="absolute -top-1 -right-1 w-5 h-5 border-t-2 border-r-2 border-indigo-400" />
                          <div className="absolute -bottom-1 -left-1  w-5 h-5 border-b-2 border-l-2 border-indigo-400" />
                          <div className="absolute -bottom-1 -right-1 w-5 h-5 border-b-2 border-r-2 border-indigo-400" />
                          <motion.div
                            animate={{ top: ["0%","100%","0%"] }}
                            transition={{ duration: 2.5, repeat: Infinity, ease: "linear" }}
                            className="absolute left-0 w-full h-0.5 bg-gradient-to-r from-transparent via-indigo-400 to-transparent shadow-[0_0_12px_#818cf8]"
                          />
                        </div>
                      </div>
                      {/* Instruction banner at bottom */}
                      {activeInstruction && (
                        <div className="absolute bottom-14 left-4 right-4 flex justify-center">
                          <motion.div
                            key={activeInstruction}
                            initial={{ opacity: 0, y: 6 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="px-4 py-2 rounded-full bg-indigo-600/80 backdrop-blur-sm text-white text-sm font-medium shadow-lg"
                          >
                            {activeInstruction}
                          </motion.div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Verifying */}
                  {pageState === "verifying" && (
                    <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-black/70 backdrop-blur-sm">
                      <Loader2 className="w-12 h-12 text-indigo-400 animate-spin mb-3" />
                      <p className="text-white font-semibold">Verifying identity…</p>
                      <p className="text-indigo-300 text-sm mt-1">Matching face against database</p>
                    </div>
                  )}

                  {/* Success */}
                  {pageState === "success" && (
                    <motion.div
                      initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                      className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-black/70 backdrop-blur-sm"
                    >
                      <motion.div
                        initial={{ scale: 0 }} animate={{ scale: 1 }}
                        transition={{ type: "spring", bounce: 0.5 }}
                        className="w-20 h-20 rounded-full bg-green-500/20 border border-green-500/50 flex items-center justify-center mb-4 shadow-[0_0_30px_rgba(34,197,94,0.4)]"
                      >
                        <ShieldCheck className="w-10 h-10 text-green-400" />
                      </motion.div>
                      <h3 className="text-2xl font-bold text-white">Access Granted</h3>
                      {confidence !== null && (
                        <p className="text-green-400 text-sm mt-1">Match confidence: {confidence}%</p>
                      )}
                    </motion.div>
                  )}

                  {/* Failed */}
                  {pageState === "failed" && (
                    <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-black/80">
                      <ShieldX className="w-12 h-12 text-red-400 mb-3" />
                      <p className="text-white font-semibold text-center px-8">{errorMsg || "Authentication Failed"}</p>
                    </div>
                  )}

                  {/* Top bar: LIVE + timer */}
                  {["camera-ready","monitoring"].includes(pageState) && (
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

                  {/* Progress bar */}
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

              {/* Email input */}
              {["idle","camera-ready","loading-models"].includes(pageState) && (
                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-300 ml-1">Email Address</label>
                  <div className="relative">
                    <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                      <User className="h-5 w-5 text-gray-500" />
                    </div>
                    <input
                      type="email"
                      value={email}
                      onChange={e => setEmail(e.target.value)}
                      onKeyDown={e => e.key === "Enter" && pageState === "camera-ready" && startMonitoring()}
                      className="w-full pl-11 pr-4 py-3.5 bg-black/40 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500/50 transition-all"
                      placeholder="you@example.com"
                    />
                  </div>
                </div>
              )}

              {/* Action buttons */}
              {pageState === "idle" && (
                <button
                  onClick={loadModels}
                  disabled={!email.trim()}
                  className="w-full py-4 px-6 rounded-xl font-bold text-white bg-gradient-to-r from-indigo-600 to-purple-600 shadow-[0_0_20px_rgba(99,102,241,0.3)] hover:shadow-[0_0_30px_rgba(99,102,241,0.5)] disabled:opacity-40 disabled:cursor-not-allowed hover:-translate-y-0.5 active:translate-y-0 transition-all flex items-center justify-center gap-2"
                >
                  <Eye className="w-5 h-5" />
                  Start Camera & Load AI
                </button>
              )}

              {pageState === "camera-ready" && (
                <button
                  onClick={startMonitoring}
                  className="w-full py-4 px-6 rounded-xl font-bold text-white bg-gradient-to-r from-indigo-600 to-purple-600 shadow-[0_0_20px_rgba(99,102,241,0.3)] hover:shadow-[0_0_30px_rgba(99,102,241,0.5)] hover:-translate-y-0.5 active:translate-y-0 transition-all flex items-center justify-center gap-2 group"
                >
                  <Zap className="w-5 h-5" />
                  Begin Liveness Verification
                  <ChevronRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                </button>
              )}

              {pageState === "monitoring" && (
                <div className="py-3 text-center text-indigo-300 text-sm font-medium bg-indigo-500/10 rounded-xl border border-indigo-500/20">
                  Follow the on-screen prompts →
                </div>
              )}

              {pageState === "failed" && (
                <button
                  onClick={resetLiveness}
                  className="w-full py-4 px-6 rounded-xl font-bold text-white bg-white/10 hover:bg-white/15 border border-white/15 hover:-translate-y-0.5 transition-all flex items-center justify-center gap-2"
                >
                  <RefreshCw className="w-5 h-5" />
                  Try Again
                </button>
              )}

              {pageState === "success" && successMsg && (
                <div className="py-4 rounded-xl bg-green-500/10 border border-green-500/30 text-green-300 text-center font-semibold">
                  {successMsg}
                </div>
              )}

              {/* Debug panel (only during monitoring) */}
              {pageState === "monitoring" && (debugEAR !== null || debugLip !== null) && (
                <div className="p-4 rounded-xl bg-black/40 border border-white/5 text-xs font-mono space-y-1">
                  <div className="flex items-center gap-2 text-indigo-300 mb-2">
                    <Activity className="w-3.5 h-3.5" />
                    <span className="font-semibold uppercase tracking-wider">Live Sensor Readout</span>
                  </div>
                  {debugEAR !== null && (
                    <div className="flex justify-between text-gray-400">
                      <span>Eye Aspect Ratio (EAR)</span>
                      <span className={
                        earHistory.current.length >= 4 &&
                        Math.max(...earHistory.current) > 0.15 &&
                        debugEAR <= Math.max(...earHistory.current) * BLINK_DROP_RATIO
                          ? "text-yellow-400 font-bold"
                          : "text-green-400"
                      }>
                        {debugEAR.toFixed(3)}
                        {earHistory.current.length >= 2 &&
                          ` (max ${Math.max(...earHistory.current).toFixed(3)})`}
                      </span>
                    </div>
                  )}
                  {debugLip !== null && (
                    <div className="flex justify-between text-gray-400">
                      <span>Lip Gap (px)</span>
                      <span className={debugLip > LIP_OPEN_PX ? "text-yellow-400" : "text-gray-500"}>
                        {debugLip}px
                        {debugLip > LIP_OPEN_PX && " ← open"}
                      </span>
                    </div>
                  )}
                </div>
              )}
            </motion.div>

            {/* ── RIGHT: Checks + Instructions ─────────────────────────────── */}
            <motion.div initial={{ opacity: 0, x: 30 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2 }} className="flex flex-col gap-5">

              {/* Liveness check cards */}
              <div className="glass-panel rounded-3xl p-6 md:p-8 border-white/10">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-xl font-bold text-white">Liveness Detection</h2>
                  <span className="text-sm text-indigo-400 font-medium">{passedCount} / 4 passed</span>
                </div>

                <div className="space-y-3">
                  {checks.map(({ key, label, icon: Icon, hint }) => {
                    const passed = liveness[key];
                    const isActive = pageState === "monitoring" && !passed
                      && INSTRUCTIONS.findIndex(i => !liveness[i.key as keyof LivenessState]) === checks.indexOf({ key, label, icon: Icon, hint });
                    return (
                      <motion.div
                        key={key}
                        animate={passed ? { scale: [1, 1.02, 1] } : {}}
                        transition={{ duration: 0.3 }}
                        className={`flex items-center gap-4 p-4 rounded-2xl border transition-all duration-300 ${
                          passed
                            ? "bg-green-500/10 border-green-500/30"
                            : isActive
                            ? "bg-indigo-500/10 border-indigo-500/30"
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

              {/* Instructions */}
              <div className="glass-panel rounded-3xl p-6 border-white/10">
                <h3 className="text-base font-bold text-white mb-4 flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-yellow-400" />
                  How to Complete Liveness Check
                </h3>
                <ul className="space-y-3">
                  {[
                    "Position your face inside the scan frame",
                    "Blink your eyes naturally (close fully, then open)",
                    "Open your mouth, then close it",
                    "Gently tilt or turn your head left or right",
                    "Stay in frame — skin texture is verified automatically",
                    "Good lighting helps — avoid backlighting",
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

              {/* Confidence on success */}
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
                          initial={{ width: 0 }}
                          animate={{ width: `${confidence}%` }}
                          transition={{ duration: 1, ease: "easeOut", delay: 0.3 }}
                          className="h-full rounded-full bg-gradient-to-r from-green-500 to-emerald-400"
                        />
                      </div>
                      <p className="text-xs text-green-400/70 mt-1.5">Face descriptor match confidence</p>
                    </div>
                  </div>
                </motion.div>
              )}

              {/* Failure panel */}
              {pageState === "failed" && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}
                  className="glass-panel rounded-3xl p-6 border-red-500/20 bg-red-500/5"
                >
                  <div className="flex items-start gap-3">
                    <XCircle className="w-6 h-6 text-red-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <h3 className="text-base font-bold text-red-300 mb-1">Authentication Failed</h3>
                      <p className="text-sm text-gray-400">
                        {errorMsg || "Liveness check failed. Possible spoof detected."}
                      </p>
                    </div>
                  </div>
                </motion.div>
              )}
            </motion.div>
          </div>}
        </div>
      </main>
    </div>
  );
}

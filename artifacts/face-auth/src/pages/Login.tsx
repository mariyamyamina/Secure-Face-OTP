import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Webcam from "react-webcam";
import * as faceapi from "@vladmandic/face-api";
import {
  Eye, Smile, Move, Layers, CheckCircle2, XCircle,
  Loader2, ShieldCheck, ShieldX, RefreshCw, User, Timer,
  ChevronRight, AlertTriangle, Zap
} from "lucide-react";
import { Navbar } from "@/components/layout/Navbar";
import { useToast } from "@/hooks/use-toast";

const MODEL_URL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/";
const LIVENESS_TIMEOUT_MS = 15000; // 15 seconds

// ─── Liveness helpers ─────────────────────────────────────────────────────────

// Eye Aspect Ratio: ratio of eye height to eye width
// Uses 6 facial landmarks per eye (from face-api 68-point model)
function computeEAR(eyePoints: faceapi.Point[]): number {
  // Vertical distances
  const v1 = dist(eyePoints[1], eyePoints[5]);
  const v2 = dist(eyePoints[2], eyePoints[4]);
  // Horizontal distance
  const h = dist(eyePoints[0], eyePoints[3]);
  return (v1 + v2) / (2.0 * h);
}

function dist(a: faceapi.Point, b: faceapi.Point): number {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

// Compute centroid of a set of points
function centroid(points: faceapi.Point[]): { x: number; y: number } {
  const x = points.reduce((s, p) => s + p.x, 0) / points.length;
  const y = points.reduce((s, p) => s + p.y, 0) / points.length;
  return { x, y };
}

// Capture a grayscale pixel array from video element for texture analysis
function getGrayscalePixels(video: HTMLVideoElement, canvas: HTMLCanvasElement): Uint8ClampedArray | null {
  try {
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    canvas.width = 64;
    canvas.height = 64;
    ctx.drawImage(video, 0, 0, 64, 64);
    const data = ctx.getImageData(0, 0, 64, 64).data;
    const gray = new Uint8ClampedArray(64 * 64);
    for (let i = 0; i < gray.length; i++) {
      gray[i] = Math.round(0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2]);
    }
    return gray;
  } catch {
    return null;
  }
}

// Mean absolute difference between two pixel arrays — measures temporal variation
function pixelMAD(a: Uint8ClampedArray, b: Uint8ClampedArray): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]);
  return sum / a.length;
}

// ─── Types ────────────────────────────────────────────────────────────────────

interface LivenessState {
  blinkDetected: boolean;
  lipMovementDetected: boolean;
  headMovementDetected: boolean;
  textureVariationDetected: boolean;
}

type PageState = "idle" | "loading-models" | "camera-ready" | "monitoring" | "verifying" | "success" | "failed";

// ─── Component ────────────────────────────────────────────────────────────────

export default function Login() {
  const [pageState, setPageState] = useState<PageState>("idle");
  const [email, setEmail] = useState("");
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [liveness, setLiveness] = useState<LivenessState>({
    blinkDetected: false,
    lipMovementDetected: false,
    headMovementDetected: false,
    textureVariationDetected: false,
  });
  const [timeLeft, setTimeLeft] = useState(LIVENESS_TIMEOUT_MS / 1000);
  const [errorMessage, setErrorMessage] = useState("");
  const [successMessage, setSuccessMessage] = useState("");
  const [confidence, setConfidence] = useState<number | null>(null);

  const webcamRef = useRef<Webcam>(null);
  const offscreenCanvas = useRef<HTMLCanvasElement>(document.createElement("canvas"));
  const animFrameRef = useRef<number>(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const livenessRef = useRef<LivenessState>(liveness);
  const monitoringRef = useRef(false);

  // Rolling state for detection logic
  const prevEAR = useRef<number | null>(null);
  const blinkFrames = useRef(0);
  const prevMouthOpenness = useRef<number | null>(null);
  const lipFrames = useRef(0);
  const prevNosePos = useRef<{ x: number; y: number } | null>(null);
  const headFrames = useRef(0);
  const prevPixels = useRef<Uint8ClampedArray | null>(null);
  const textureFrames = useRef(0);

  const { toast } = useToast();

  // Keep ref in sync with state
  useEffect(() => { livenessRef.current = liveness; }, [liveness]);

  const allLivenessPassed = (l: LivenessState) =>
    l.blinkDetected && l.lipMovementDetected && l.headMovementDetected && l.textureVariationDetected;

  // ─── Load Models ─────────────────────────────────────────────────────────────
  const loadModels = useCallback(async () => {
    if (modelsLoaded) return;
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
      setErrorMessage("Failed to load AI models. Please refresh and try again.");
      setPageState("failed");
    }
  }, [modelsLoaded]);

  // ─── Reset state for retry ────────────────────────────────────────────────────
  const resetLiveness = useCallback(() => {
    setLiveness({ blinkDetected: false, lipMovementDetected: false, headMovementDetected: false, textureVariationDetected: false });
    livenessRef.current = { blinkDetected: false, lipMovementDetected: false, headMovementDetected: false, textureVariationDetected: false };
    setTimeLeft(LIVENESS_TIMEOUT_MS / 1000);
    setErrorMessage("");
    setSuccessMessage("");
    setConfidence(null);
    prevEAR.current = null;
    blinkFrames.current = 0;
    prevMouthOpenness.current = null;
    lipFrames.current = 0;
    prevNosePos.current = null;
    headFrames.current = 0;
    prevPixels.current = null;
    textureFrames.current = 0;
    monitoringRef.current = false;
    cancelAnimationFrame(animFrameRef.current);
    if (timerRef.current) clearInterval(timerRef.current);
    setPageState("camera-ready");
  }, []);

  // ─── Liveness monitor loop ────────────────────────────────────────────────────
  const runMonitorFrame = useCallback(async () => {
    if (!monitoringRef.current) return;

    const video = webcamRef.current?.video;
    if (!video || video.readyState !== 4) {
      animFrameRef.current = requestAnimationFrame(runMonitorFrame);
      return;
    }

    try {
      const detection = await faceapi
        .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.5 }))
        .withFaceLandmarks()
        .withFaceDescriptor();

      if (detection) {
        const { landmarks } = detection;
        const pts = landmarks.positions;

        // ── 1. Eye Blink (EAR) ─────────────────────────────────────────────────
        // 68-point model: left eye = 36-41, right eye = 42-47
        const leftEye = pts.slice(36, 42) as faceapi.Point[];
        const rightEye = pts.slice(42, 48) as faceapi.Point[];
        const ear = (computeEAR(leftEye) + computeEAR(rightEye)) / 2;

        const EAR_CLOSED = 0.21;
        const EAR_OPEN = 0.25;

        if (!livenessRef.current.blinkDetected) {
          if (prevEAR.current !== null) {
            // Detect a blink: previously open → now closed → then back open
            if (prevEAR.current > EAR_OPEN && ear < EAR_CLOSED) {
              blinkFrames.current++;
            } else if (blinkFrames.current > 0 && ear > EAR_OPEN) {
              // Completed a blink
              setLiveness(prev => ({ ...prev, blinkDetected: true }));
            }
          }
          prevEAR.current = ear;
        }

        // ── 2. Lip Movement (mouth openness) ───────────────────────────────────
        // Mouth inner points: 60-67 (68-point model)
        // Use vertical distance of top and bottom lip points (pts 62 vs 66)
        const topLip = pts[62];
        const bottomLip = pts[66];
        const mouthH = Math.abs(topLip.y - bottomLip.y);

        if (!livenessRef.current.lipMovementDetected) {
          if (prevMouthOpenness.current !== null) {
            const delta = Math.abs(mouthH - prevMouthOpenness.current);
            if (delta > 3) lipFrames.current++;
            else lipFrames.current = Math.max(0, lipFrames.current - 1);
            if (lipFrames.current >= 3) {
              setLiveness(prev => ({ ...prev, lipMovementDetected: true }));
            }
          }
          prevMouthOpenness.current = mouthH;
        }

        // ── 3. Head Movement (nose tip position) ──────────────────────────────
        // Nose tip: pt[30]
        const noseTip = pts[30];

        if (!livenessRef.current.headMovementDetected) {
          if (prevNosePos.current !== null) {
            const dx = Math.abs(noseTip.x - prevNosePos.current.x);
            const dy = Math.abs(noseTip.y - prevNosePos.current.y);
            if (dx > 4 || dy > 4) headFrames.current++;
            else headFrames.current = Math.max(0, headFrames.current - 1);
            if (headFrames.current >= 3) {
              setLiveness(prev => ({ ...prev, headMovementDetected: true }));
            }
          }
          prevNosePos.current = { x: noseTip.x, y: noseTip.y };
        }
      }
    } catch {
      // ignore individual frame errors
    }

    // ── 4. Skin Texture / Temporal Variation ──────────────────────────────────
    if (!livenessRef.current.textureVariationDetected) {
      const video = webcamRef.current?.video;
      if (video) {
        const currentPixels = getGrayscalePixels(video, offscreenCanvas.current);
        if (currentPixels && prevPixels.current) {
          const mad = pixelMAD(currentPixels, prevPixels.current);
          if (mad > 1.5) textureFrames.current++;
          if (textureFrames.current >= 5) {
            setLiveness(prev => ({ ...prev, textureVariationDetected: true }));
          }
        }
        prevPixels.current = currentPixels;
      }
    }

    if (monitoringRef.current) {
      animFrameRef.current = requestAnimationFrame(runMonitorFrame);
    }
  }, []);

  // ─── Start monitoring ─────────────────────────────────────────────────────────
  const startMonitoring = useCallback(() => {
    if (!email.trim()) {
      toast({ variant: "destructive", title: "Email required", description: "Please enter your email address." });
      return;
    }
    monitoringRef.current = true;
    setPageState("monitoring");
    setTimeLeft(LIVENESS_TIMEOUT_MS / 1000);

    // Countdown timer
    timerRef.current = setInterval(() => {
      setTimeLeft(prev => {
        if (prev <= 1) {
          if (timerRef.current) clearInterval(timerRef.current);
          monitoringRef.current = false;
          cancelAnimationFrame(animFrameRef.current);
          if (!allLivenessPassed(livenessRef.current)) {
            setErrorMessage("Time's up! Liveness check failed. Please try again.");
            setPageState("failed");
          }
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    animFrameRef.current = requestAnimationFrame(runMonitorFrame);
  }, [email, runMonitorFrame, toast]);

  // Auto-verify when all checks pass
  useEffect(() => {
    if (allLivenessPassed(liveness) && pageState === "monitoring") {
      monitoringRef.current = false;
      cancelAnimationFrame(animFrameRef.current);
      if (timerRef.current) clearInterval(timerRef.current);
      verifyLogin();
    }
  }, [liveness, pageState]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      monitoringRef.current = false;
      cancelAnimationFrame(animFrameRef.current);
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  // ─── Capture and verify ───────────────────────────────────────────────────────
  const verifyLogin = useCallback(async () => {
    setPageState("verifying");

    const video = webcamRef.current?.video;
    if (!video) {
      setErrorMessage("Camera not available.");
      setPageState("failed");
      return;
    }

    try {
      const detection = await faceapi
        .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.5 }))
        .withFaceLandmarks()
        .withFaceDescriptor();

      if (!detection) {
        setErrorMessage("No face detected during verification. Please try again.");
        setPageState("failed");
        return;
      }

      const descriptor = Array.from(detection.descriptor);

      const response = await fetch("/api/login-face", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email,
          face_descriptor: descriptor,
          liveness_passed: true,
        }),
      });

      const data = await response.json();

      if (response.ok) {
        setConfidence(data.confidence ?? null);
        setSuccessMessage(data.message || "Login successful!");
        setPageState("success");
      } else {
        setErrorMessage(data.error || "Authentication failed.");
        setPageState("failed");
      }
    } catch (err: any) {
      console.error("Login error:", err);
      setErrorMessage("Network error. Please check your connection.");
      setPageState("failed");
    }
  }, [email]);

  // ─── Liveness check items ─────────────────────────────────────────────────────
  const checks = [
    { key: "blinkDetected", label: "Eye Blink", icon: Eye, hint: "Blink naturally" },
    { key: "lipMovementDetected", label: "Lip Movement", icon: Smile, hint: "Open/close mouth slightly" },
    { key: "headMovementDetected", label: "Head Movement", icon: Move, hint: "Gently tilt head" },
    { key: "textureVariationDetected", label: "Real Skin Texture", icon: Layers, hint: "Stay in frame" },
  ] as const;

  const passedCount = Object.values(liveness).filter(Boolean).length;

  // ─── Render ───────────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen flex flex-col relative overflow-hidden bg-background">
      {/* Background glow */}
      <div className="fixed inset-0 z-0 pointer-events-none">
        <div className="absolute top-[5%] left-[5%] w-[45%] h-[45%] rounded-full bg-indigo-600/5 blur-[120px]" />
        <div className="absolute bottom-[5%] right-[5%] w-[40%] h-[40%] rounded-full bg-purple-600/5 blur-[100px]" />
      </div>

      <Navbar />

      <main className="flex-1 container mx-auto px-4 py-24 md:py-28 relative z-10 flex items-start justify-center">
        <div className="w-full max-w-6xl">

          {/* ── Header ───────────────────────────────────────────────────── */}
          <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="text-center mb-10">
            <h1 className="text-4xl md:text-5xl font-display font-bold text-white mb-3">
              Biometric <span className="bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">Login</span>
            </h1>
            <p className="text-muted-foreground max-w-xl mx-auto">
              Complete liveness detection to securely access your account. The system verifies you're a real person in real-time.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">

            {/* ── Left: Camera + Controls ───────────────────────────────── */}
            <motion.div initial={{ opacity: 0, x: -30 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 }} className="flex flex-col gap-5">

              {/* Camera panel */}
              <div className="glass-panel rounded-3xl overflow-hidden border-white/10 relative bg-black/60">
                <div className="absolute inset-0 z-10 pointer-events-none opacity-10 bg-[linear-gradient(rgba(255,255,255,0.08)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.08)_1px,transparent_1px)] bg-[size:24px_24px]" />

                <div className="aspect-[4/3] relative">
                  {/* Webcam feed — always mounted after models load */}
                  {(pageState === "camera-ready" || pageState === "monitoring" || pageState === "verifying" || pageState === "success") && (
                    <Webcam
                      audio={false}
                      ref={webcamRef}
                      screenshotFormat="image/jpeg"
                      videoConstraints={{ facingMode: "user", width: 640, height: 480 }}
                      className="w-full h-full object-cover"
                      mirrored
                    />
                  )}

                  {/* Idle / loading overlay */}
                  {(pageState === "idle" || pageState === "loading-models") && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/90 z-20">
                      {pageState === "loading-models" ? (
                        <>
                          <Loader2 className="w-10 h-10 text-indigo-400 animate-spin mb-4" />
                          <p className="text-indigo-300 font-medium tracking-widest text-sm uppercase">Initialising Neural Networks</p>
                        </>
                      ) : (
                        <>
                          <div className="w-20 h-20 rounded-full bg-indigo-500/10 border border-indigo-500/30 flex items-center justify-center mb-4">
                            <User className="w-10 h-10 text-indigo-400" />
                          </div>
                          <p className="text-white/60 text-sm">Camera will start after model load</p>
                        </>
                      )}
                    </div>
                  )}

                  {/* Face scan overlay during monitoring */}
                  {pageState === "monitoring" && (
                    <div className="absolute inset-0 z-20 pointer-events-none flex items-center justify-center">
                      <div className="w-44 h-60 border-2 border-indigo-400/40 rounded-[40px] relative">
                        <div className="absolute -top-1 -left-1 w-5 h-5 border-t-2 border-l-2 border-indigo-400" />
                        <div className="absolute -top-1 -right-1 w-5 h-5 border-t-2 border-r-2 border-indigo-400" />
                        <div className="absolute -bottom-1 -left-1 w-5 h-5 border-b-2 border-l-2 border-indigo-400" />
                        <div className="absolute -bottom-1 -right-1 w-5 h-5 border-b-2 border-r-2 border-indigo-400" />
                        {/* Animated scan beam */}
                        <motion.div
                          animate={{ top: ["0%", "100%", "0%"] }}
                          transition={{ duration: 2.5, repeat: Infinity, ease: "linear" }}
                          className="absolute left-0 w-full h-0.5 bg-gradient-to-r from-transparent via-indigo-400 to-transparent shadow-[0_0_12px_#818cf8]"
                        />
                      </div>
                    </div>
                  )}

                  {/* Verifying overlay */}
                  {pageState === "verifying" && (
                    <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-black/70 backdrop-blur-sm">
                      <Loader2 className="w-12 h-12 text-indigo-400 animate-spin mb-3" />
                      <p className="text-white font-semibold">Verifying identity…</p>
                      <p className="text-indigo-300 text-sm mt-1">Comparing face against database</p>
                    </div>
                  )}

                  {/* Success overlay */}
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
                        <p className="text-green-400 text-sm mt-1">Confidence: {confidence}%</p>
                      )}
                    </motion.div>
                  )}

                  {/* Failed overlay */}
                  {pageState === "failed" && (
                    <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-black/80">
                      <ShieldX className="w-12 h-12 text-red-400 mb-3" />
                      <p className="text-white font-semibold text-center px-6">{errorMessage || "Authentication Failed"}</p>
                    </div>
                  )}

                  {/* Top bar: LIVE badge + timer */}
                  {(pageState === "monitoring" || pageState === "camera-ready") && (
                    <div className="absolute top-4 left-4 right-4 z-30 flex items-center justify-between">
                      <div className="flex items-center gap-2 bg-black/60 backdrop-blur-sm px-3 py-1.5 rounded-full border border-white/10">
                        <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                        <span className="text-xs font-mono text-white/80 uppercase">Live</span>
                      </div>
                      {pageState === "monitoring" && (
                        <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full border text-xs font-mono font-bold ${timeLeft <= 5 ? "bg-red-900/60 border-red-500/50 text-red-300" : "bg-black/60 border-white/10 text-indigo-300"}`}>
                          <Timer className="w-3.5 h-3.5" />
                          {timeLeft}s
                        </div>
                      )}
                    </div>
                  )}

                  {/* Progress bar at bottom during monitoring */}
                  {pageState === "monitoring" && (
                    <div className="absolute bottom-0 left-0 right-0 z-30">
                      <div className="h-1 bg-white/10">
                        <motion.div
                          className="h-full bg-gradient-to-r from-indigo-500 to-purple-500"
                          style={{ width: `${(passedCount / 4) * 100}%` }}
                          transition={{ duration: 0.3 }}
                        />
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Email input */}
              {(pageState === "idle" || pageState === "camera-ready" || pageState === "loading-models") && (
                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-300 ml-1">Email Address</label>
                  <div className="relative">
                    <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                      <User className="h-5 w-5 text-gray-500" />
                    </div>
                    <input
                      type="email"
                      required
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      className="w-full pl-11 pr-4 py-3.5 bg-black/40 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500/50 transition-all"
                      placeholder="you@example.com"
                    />
                  </div>
                </div>
              )}

              {/* Action buttons */}
              <div className="flex flex-col gap-3">
                {pageState === "idle" && (
                  <button
                    onClick={loadModels}
                    disabled={!email.trim()}
                    className="w-full py-4 px-6 rounded-xl font-bold text-white bg-gradient-to-r from-indigo-600 to-purple-600 shadow-[0_0_20px_rgba(99,102,241,0.3)] hover:shadow-[0_0_30px_rgba(99,102,241,0.5)] disabled:opacity-50 disabled:cursor-not-allowed hover:-translate-y-0.5 active:translate-y-0 transition-all duration-200 flex items-center justify-center gap-2"
                  >
                    <Eye className="w-5 h-5" />
                    Start Camera & Load AI
                  </button>
                )}

                {pageState === "camera-ready" && (
                  <button
                    onClick={startMonitoring}
                    className="w-full py-4 px-6 rounded-xl font-bold text-white bg-gradient-to-r from-indigo-600 to-purple-600 shadow-[0_0_20px_rgba(99,102,241,0.3)] hover:shadow-[0_0_30px_rgba(99,102,241,0.5)] hover:-translate-y-0.5 active:translate-y-0 transition-all duration-200 flex items-center justify-center gap-2 group"
                  >
                    <Zap className="w-5 h-5" />
                    Begin Liveness Verification
                    <ChevronRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                  </button>
                )}

                {pageState === "monitoring" && (
                  <div className="text-center text-indigo-300 text-sm font-medium py-3 bg-indigo-500/10 rounded-xl border border-indigo-500/20">
                    Analysing live feed… Follow the prompts on the right →
                  </div>
                )}

                {(pageState === "failed") && (
                  <button
                    onClick={resetLiveness}
                    className="w-full py-4 px-6 rounded-xl font-bold text-white bg-white/10 hover:bg-white/15 border border-white/15 hover:-translate-y-0.5 active:translate-y-0 transition-all duration-200 flex items-center justify-center gap-2"
                  >
                    <RefreshCw className="w-5 h-5" />
                    Try Again
                  </button>
                )}

                {pageState === "success" && (
                  <div className="py-4 rounded-xl bg-green-500/10 border border-green-500/30 text-green-300 text-center font-semibold">
                    {successMessage}
                  </div>
                )}
              </div>
            </motion.div>

            {/* ── Right: Liveness checks panel ──────────────────────────── */}
            <motion.div initial={{ opacity: 0, x: 30 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2 }} className="flex flex-col gap-5">

              {/* Check list */}
              <div className="glass-panel rounded-3xl p-6 md:p-8 border-white/10">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-xl font-bold text-white">Liveness Detection</h2>
                  <span className="text-sm text-indigo-400 font-medium">{passedCount}/4 passed</span>
                </div>

                <div className="space-y-4">
                  {checks.map(({ key, label, icon: Icon, hint }) => {
                    const passed = liveness[key];
                    return (
                      <motion.div
                        key={key}
                        animate={passed ? { scale: [1, 1.02, 1] } : {}}
                        transition={{ duration: 0.3 }}
                        className={`flex items-center gap-4 p-4 rounded-2xl border transition-all duration-300 ${
                          passed
                            ? "bg-green-500/10 border-green-500/30"
                            : pageState === "monitoring"
                            ? "bg-indigo-500/5 border-indigo-500/20 animate-pulse"
                            : "bg-white/3 border-white/8"
                        }`}
                      >
                        <div className={`w-11 h-11 rounded-xl flex items-center justify-center flex-shrink-0 transition-colors duration-300 ${passed ? "bg-green-500/20" : "bg-white/5"}`}>
                          <Icon className={`w-5 h-5 ${passed ? "text-green-400" : "text-gray-400"}`} />
                        </div>
                        <div className="flex-1">
                          <p className={`font-semibold text-sm ${passed ? "text-green-300" : "text-white"}`}>{label}</p>
                          {!passed && pageState === "monitoring" && (
                            <p className="text-xs text-indigo-400 mt-0.5">{hint}</p>
                          )}
                          {passed && (
                            <p className="text-xs text-green-500/80 mt-0.5">Verified ✓</p>
                          )}
                        </div>
                        <AnimatePresence mode="wait">
                          {passed ? (
                            <motion.div key="check" initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ type: "spring", bounce: 0.6 }}>
                              <CheckCircle2 className="w-6 h-6 text-green-400" />
                            </motion.div>
                          ) : (
                            <motion.div key="circle" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                              <div className={`w-6 h-6 rounded-full border-2 ${pageState === "monitoring" ? "border-indigo-500/50" : "border-white/20"}`} />
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </motion.div>
                    );
                  })}
                </div>

                {/* Overall status bar */}
                <div className="mt-6">
                  <div className="flex items-center justify-between text-xs text-gray-400 mb-2">
                    <span>Overall Progress</span>
                    <span>{Math.round((passedCount / 4) * 100)}%</span>
                  </div>
                  <div className="h-2 rounded-full bg-white/5 overflow-hidden">
                    <motion.div
                      className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-purple-500"
                      animate={{ width: `${(passedCount / 4) * 100}%` }}
                      transition={{ duration: 0.5, ease: "easeOut" }}
                    />
                  </div>
                </div>
              </div>

              {/* Instructions / How it works */}
              <div className="glass-panel rounded-3xl p-6 border-white/10">
                <h3 className="text-base font-bold text-white mb-4 flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-yellow-400" />
                  Anti-Spoof Instructions
                </h3>
                <ul className="space-y-3">
                  {[
                    "Position your face inside the detection frame",
                    "Blink naturally at least once",
                    "Open and close your mouth slightly",
                    "Gently turn your head left or right",
                    "Ensure good lighting, no glasses if possible",
                    "Complete all checks within the time limit",
                  ].map((tip, i) => (
                    <li key={i} className="flex items-start gap-3 text-sm text-gray-400">
                      <span className="flex-shrink-0 w-5 h-5 rounded-full bg-indigo-500/15 border border-indigo-500/30 flex items-center justify-center text-xs text-indigo-400 font-bold">{i + 1}</span>
                      {tip}
                    </li>
                  ))}
                </ul>
              </div>

              {/* Confidence score on success */}
              {pageState === "success" && confidence !== null && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
                  className="glass-panel rounded-3xl p-6 border-green-500/20 bg-green-500/5"
                >
                  <h3 className="text-base font-bold text-green-300 mb-3">Authentication Score</h3>
                  <div className="flex items-center gap-4">
                    <div className="text-5xl font-bold text-white">{confidence}<span className="text-xl text-green-400">%</span></div>
                    <div className="flex-1">
                      <div className="h-3 rounded-full bg-white/5 overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${confidence}%` }}
                          transition={{ duration: 1, ease: "easeOut", delay: 0.3 }}
                          className="h-full rounded-full bg-gradient-to-r from-green-500 to-emerald-400"
                        />
                      </div>
                      <p className="text-xs text-green-400/70 mt-1.5">Face match confidence</p>
                    </div>
                  </div>
                </motion.div>
              )}

              {/* Spoof failure message */}
              {pageState === "failed" && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}
                  className="glass-panel rounded-3xl p-6 border-red-500/20 bg-red-500/5"
                >
                  <div className="flex items-start gap-3">
                    <XCircle className="w-6 h-6 text-red-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <h3 className="text-base font-bold text-red-300 mb-1">Authentication Failed</h3>
                      <p className="text-sm text-gray-400">{errorMessage || "Liveness check failed. Possible spoof detected."}</p>
                    </div>
                  </div>
                </motion.div>
              )}
            </motion.div>
          </div>
        </div>
      </main>
    </div>
  );
}

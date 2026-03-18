import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Link } from "wouter";
import Webcam from "react-webcam";
import * as faceapi from "@vladmandic/face-api";
import { 
  ShieldCheck, AlertCircle, Scan, Fingerprint, 
  Loader2, CheckCircle2, ChevronRight, User, Key 
} from "lucide-react";
import { Navbar } from "@/components/layout/Navbar";
import { useRegisterFace } from "@workspace/api-client-react";
import { useToast } from "@/hooks/use-toast";

const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/';

export default function Register() {
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [modelError, setModelError] = useState<string | null>(null);
  
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  
  const [isCapturing, setIsCapturing] = useState(false);
  const [descriptor, setDescriptor] = useState<number[] | null>(null);
  const [scanStatus, setScanStatus] = useState<'idle' | 'scanning' | 'success' | 'error'>('idle');
  const [scanMessage, setScanMessage] = useState("");

  const webcamRef = useRef<Webcam>(null);
  const { toast } = useToast();
  
  const registerMutation = useRegisterFace();

  useEffect(() => {
    const loadModels = async () => {
      try {
        // Wait for TensorFlow.js backend to be ready before loading models
        await faceapi.tf.setBackend('cpu');
        await faceapi.tf.ready();
        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
          faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
          faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
        ]);
        setModelsLoaded(true);
      } catch (err) {
        console.error("Failed to load face-api models", err);
        setModelError("Failed to load AI models. Please check network connection.");
      }
    };
    loadModels();
  }, []);

  const captureFace = useCallback(async () => {
    if (!webcamRef.current) return;
    
    setScanStatus('scanning');
    setIsCapturing(true);
    setScanMessage("Analyzing facial geometry...");

    const imageSrc = webcamRef.current.getScreenshot();
    
    if (!imageSrc) {
      setScanStatus('error');
      setScanMessage("Failed to capture image. Please check camera permissions.");
      setIsCapturing(false);
      return;
    }

    try {
      // Create HTML Image Element from base64
      const img = new Image();
      img.src = imageSrc;
      
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
      });

      const detections = await faceapi.detectAllFaces(
        img, 
        new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.5 })
      )
      .withFaceLandmarks()
      .withFaceDescriptors();

      if (detections.length === 0) {
        throw new Error("No face detected. Please face the camera directly.");
      }
      if (detections.length > 1) {
        throw new Error("Multiple faces detected. Ensure only you are in frame.");
      }

      const faceDescriptor = Array.from(detections[0].descriptor);
      setDescriptor(faceDescriptor);
      setScanStatus('success');
      setScanMessage("Identity matrix extracted successfully.");
      
      toast({
        title: "Scan Successful",
        description: "Your facial geometry has been processed securely.",
      });

    } catch (err: any) {
      setScanStatus('error');
      setScanMessage(err.message || "An error occurred during facial extraction.");
      setDescriptor(null);
    } finally {
      setIsCapturing(false);
    }
  }, [toast]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!descriptor) {
      toast({
        variant: "destructive",
        title: "Missing Biometrics",
        description: "Please capture your face before submitting.",
      });
      return;
    }

    if (password.length < 6) {
      toast({
        variant: "destructive",
        title: "Weak Password",
        description: "Password must be at least 6 characters.",
      });
      return;
    }

    registerMutation.mutate(
      { data: { email, password, face_descriptor: descriptor } },
      {
        onSuccess: () => {
          toast({
            title: "Registration Complete",
            description: "Your identity has been securely vaulted. Redirecting...",
          });
          // In a real app, redirect to login after a delay
          setTimeout(() => {
            window.location.href = "/login";
          }, 2000);
        },
        onError: (err) => {
          toast({
            variant: "destructive",
            title: "Registration Failed",
            description: err.response?.data?.error || "An unexpected error occurred.",
          });
        }
      }
    );
  };

  return (
    <div className="min-h-screen flex flex-col relative bg-background overflow-x-hidden">
      <div className="fixed inset-0 z-0 pointer-events-none">
        <div className="absolute top-[10%] -left-[10%] w-[50%] h-[50%] rounded-full bg-blue-500/5 blur-[120px]" />
        <div className="absolute bottom-[10%] -right-[10%] w-[50%] h-[50%] rounded-full bg-purple-500/5 blur-[120px]" />
      </div>

      <Navbar />

      <main className="flex-1 container mx-auto px-4 md:px-6 py-24 md:py-32 relative z-10 flex items-center justify-center">
        
        <div className="w-full max-w-6xl grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
          
          {/* Left Column: Form */}
          <motion.div 
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6 }}
            className="glass-panel p-8 md:p-10 rounded-3xl border-white/10 flex flex-col h-full"
          >
            <div className="mb-8">
              <h2 className="text-3xl font-display font-bold text-white mb-2">Initialize Identity</h2>
              <p className="text-muted-foreground text-sm">Create your biometric vault to secure your access.</p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-6 flex-1 flex flex-col">
              <div className="space-y-4">
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
                      className="w-full pl-11 pr-4 py-3.5 bg-black/40 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary/50 transition-all"
                      placeholder="operative@aura.sys"
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium text-gray-300 ml-1">Secure Passphrase</label>
                  <div className="relative">
                    <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                      <Key className="h-5 w-5 text-gray-500" />
                    </div>
                    <input
                      type="password"
                      required
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      className="w-full pl-11 pr-4 py-3.5 bg-black/40 border border-white/10 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary/50 transition-all"
                      placeholder="••••••••••••"
                    />
                  </div>
                </div>
              </div>

              {/* Status Indicator */}
              <div className="mt-8 p-4 rounded-xl bg-black/30 border border-white/5 flex items-center gap-4">
                <div className={`p-2 rounded-lg ${descriptor ? 'bg-green-500/20 text-green-400' : 'bg-white/5 text-gray-400'}`}>
                  <Fingerprint className="w-6 h-6" />
                </div>
                <div>
                  <h4 className="text-sm font-medium text-white">Biometric Status</h4>
                  <p className={`text-xs ${descriptor ? 'text-green-400' : 'text-gray-400'}`}>
                    {descriptor ? 'Descriptor acquired and validated.' : 'Awaiting facial scan...'}
                  </p>
                </div>
              </div>

              <div className="mt-auto pt-8">
                <button
                  type="submit"
                  disabled={!descriptor || registerMutation.isPending}
                  className="w-full py-4 px-6 rounded-xl font-bold text-white bg-gradient-to-r from-primary to-purple-600 shadow-[0_0_20px_rgba(99,102,241,0.3)] hover:shadow-[0_0_30px_rgba(99,102,241,0.5)] disabled:opacity-50 disabled:cursor-not-allowed disabled:shadow-none hover:-translate-y-0.5 active:translate-y-0 transition-all duration-200 flex items-center justify-center gap-2 group"
                >
                  {registerMutation.isPending ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      Encrypting...
                    </>
                  ) : (
                    <>
                      Initialize Sequence
                      <ChevronRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                    </>
                  )}
                </button>
              </div>
            </form>
          </motion.div>

          {/* Right Column: Scanner */}
          <motion.div 
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="flex flex-col h-full"
          >
            <div className="glass-panel p-2 rounded-3xl border-white/10 overflow-hidden relative shadow-2xl bg-black/60">
              
              {/* Overlay grid for aesthetics */}
              <div className="absolute inset-0 z-10 pointer-events-none opacity-20 bg-[linear-gradient(rgba(255,255,255,0.1)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.1)_1px,transparent_1px)] bg-[size:20px_20px]" />
              
              {!modelsLoaded ? (
                <div className="aspect-[4/3] w-full bg-black/80 rounded-2xl flex flex-col items-center justify-center relative z-20">
                  {modelError ? (
                    <div className="text-destructive flex flex-col items-center gap-2 px-6 text-center">
                      <AlertCircle className="w-10 h-10 mb-2" />
                      <p className="font-medium">{modelError}</p>
                    </div>
                  ) : (
                    <>
                      <Loader2 className="w-10 h-10 text-primary animate-spin mb-4" />
                      <p className="text-primary font-medium tracking-widest text-sm uppercase">LOADING NEURAL NETWORKS</p>
                    </>
                  )}
                </div>
              ) : (
                <div className="relative aspect-[4/3] w-full rounded-2xl overflow-hidden bg-black group">
                  <Webcam
                    audio={false}
                    ref={webcamRef}
                    screenshotFormat="image/jpeg"
                    videoConstraints={{ facingMode: "user" }}
                    className={`w-full h-full object-cover transition-opacity duration-500 ${descriptor ? 'opacity-50 grayscale blur-[2px]' : 'opacity-100'}`}
                  />
                  
                  {/* Scanner Overlay UI */}
                  {!descriptor && (
                    <div className="absolute inset-0 z-20 pointer-events-none flex items-center justify-center">
                      <div className="w-48 h-64 border-2 border-primary/30 rounded-[40px] relative">
                        {/* Corner markers */}
                        <div className="absolute -top-1 -left-1 w-4 h-4 border-t-2 border-l-2 border-primary" />
                        <div className="absolute -top-1 -right-1 w-4 h-4 border-t-2 border-r-2 border-primary" />
                        <div className="absolute -bottom-1 -left-1 w-4 h-4 border-b-2 border-l-2 border-primary" />
                        <div className="absolute -bottom-1 -right-1 w-4 h-4 border-b-2 border-r-2 border-primary" />
                        
                        {/* Scanning beam */}
                        {scanStatus === 'scanning' && (
                          <motion.div 
                            initial={{ top: 0, opacity: 0 }}
                            animate={{ top: ['0%', '100%', '0%'], opacity: [0, 1, 0] }}
                            transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                            className="absolute left-0 w-full h-1 bg-primary shadow-[0_0_15px_#3b82f6]" 
                          />
                        )}
                      </div>
                    </div>
                  )}

                  {/* Success Overlay */}
                  <AnimatePresence>
                    {descriptor && (
                      <motion.div 
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className="absolute inset-0 z-30 flex flex-col items-center justify-center bg-black/60 backdrop-blur-sm"
                      >
                        <motion.div 
                          initial={{ scale: 0 }}
                          animate={{ scale: 1 }}
                          transition={{ type: "spring", bounce: 0.5 }}
                          className="w-20 h-20 bg-green-500/20 rounded-full flex items-center justify-center mb-4 border border-green-500/50 shadow-[0_0_30px_rgba(34,197,94,0.3)]"
                        >
                          <CheckCircle2 className="w-10 h-10 text-green-400" />
                        </motion.div>
                        <h3 className="text-xl font-bold text-white mb-1">Identity Verified</h3>
                        <p className="text-sm text-green-400">Descriptor locked & ready</p>
                        
                        <button 
                          onClick={() => setDescriptor(null)}
                          className="mt-6 px-4 py-2 text-sm text-white/70 hover:text-white border border-white/10 hover:bg-white/10 rounded-lg transition-colors"
                        >
                          Retake Scan
                        </button>
                      </motion.div>
                    )}
                  </AnimatePresence>

                  {/* Top Status Bar */}
                  <div className="absolute top-4 left-4 right-4 z-20 flex justify-between items-center">
                    <div className="flex items-center gap-2 bg-black/50 backdrop-blur-md px-3 py-1.5 rounded-full border border-white/10">
                      <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                      <span className="text-xs font-mono text-white/80">LIVE</span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="mt-6 flex flex-col items-center">
              <button
                onClick={captureFace}
                disabled={!modelsLoaded || isCapturing || descriptor !== null}
                className="group relative w-20 h-20 flex items-center justify-center rounded-full disabled:opacity-50 disabled:cursor-not-allowed transition-transform active:scale-95"
              >
                <div className="absolute inset-0 bg-white/10 rounded-full border border-white/20 group-hover:border-primary/50 transition-colors" />
                <div className="absolute inset-2 bg-white/20 rounded-full group-hover:bg-primary/40 transition-colors flex items-center justify-center shadow-[0_0_20px_rgba(255,255,255,0.1)] group-hover:shadow-[0_0_30px_rgba(99,102,241,0.4)]">
                  <Scan className="w-8 h-8 text-white" />
                </div>
              </button>
              
              <div className="mt-6 h-8 text-center">
                <AnimatePresence mode="wait">
                  <motion.p 
                    key={scanStatus}
                    initial={{ opacity: 0, y: 5 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -5 }}
                    className={`text-sm font-medium ${
                      scanStatus === 'error' ? 'text-destructive' :
                      scanStatus === 'success' ? 'text-green-400' :
                      'text-muted-foreground'
                    }`}
                  >
                    {scanMessage || (modelsLoaded ? "Align face in frame and press to extract" : "Waking up visual sensors...")}
                  </motion.p>
                </AnimatePresence>
              </div>
            </div>
            
          </motion.div>
        </div>
      </main>
    </div>
  );
}

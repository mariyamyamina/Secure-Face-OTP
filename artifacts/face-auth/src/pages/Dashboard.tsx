import { useEffect } from "react";
import { useLocation } from "wouter";
import { motion } from "framer-motion";
import {
  Shield, ShieldCheck, LogOut, User, Mail,
  ScanFace, Activity, Lock, RefreshCw, BadgeCheck,
} from "lucide-react";
import { Navbar } from "@/components/layout/Navbar";
import { useUser } from "@/context/UserContext";

export default function Dashboard() {
  const { user, logout } = useUser();
  const [, navigate] = useLocation();

  useEffect(() => {
    if (!user) {
      navigate("/login");
    }
  }, [user, navigate]);

  if (!user) return null;

  const displayName = user.name || user.email.split("@")[0];

  function handleLogout() {
    logout();
    navigate("/login");
  }

  return (
    <div className="min-h-screen bg-background">
      <Navbar />

      <main className="pt-28 pb-16 px-4">
        <div className="max-w-4xl mx-auto space-y-8">

          {/* ── Welcome Banner ────────────────────────────────────────────── */}
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="glass-panel rounded-3xl p-8 border-green-500/20 bg-gradient-to-br from-green-500/10 to-emerald-500/5"
          >
            <div className="flex flex-col sm:flex-row items-start sm:items-center gap-5">
              {/* Avatar */}
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ type: "spring", bounce: 0.5, delay: 0.15 }}
                className="relative flex-shrink-0"
              >
                <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-[0_0_40px_rgba(99,102,241,0.4)]">
                  <User className="w-10 h-10 text-white" />
                </div>
                <div className="absolute -bottom-1 -right-1 w-6 h-6 rounded-full bg-green-500 border-2 border-background flex items-center justify-center">
                  <BadgeCheck className="w-3.5 h-3.5 text-white" />
                </div>
              </motion.div>

              {/* Welcome text */}
              <div className="flex-1">
                <p className="text-sm text-green-400 font-semibold tracking-widest uppercase mb-1">
                  Authentication Successful
                </p>
                <h1 className="text-3xl sm:text-4xl font-bold text-white mb-1">
                  Welcome, <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400">{displayName}</span>
                </h1>
                <p className="text-muted-foreground text-sm">{user.email}</p>
              </div>

              {/* Security badge */}
              <div className="flex items-center gap-2 px-4 py-2 rounded-xl bg-green-500/15 border border-green-500/30 text-green-300 text-sm font-medium">
                <ShieldCheck className="w-4 h-4" />
                Verified
              </div>
            </div>
          </motion.div>

          {/* ── Profile Card + Status ──────────────────────────────────────── */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

            {/* Profile Card */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
              className="glass-panel rounded-3xl p-6 border-white/10"
            >
              <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-widest mb-5 flex items-center gap-2">
                <User className="w-4 h-4" /> Profile
              </h2>
              <div className="space-y-4">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-xl bg-blue-500/15 flex items-center justify-center">
                    <User className="w-5 h-5 text-blue-400" />
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Name</p>
                    <p className="text-sm font-semibold text-white">{displayName}</p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-xl bg-purple-500/15 flex items-center justify-center">
                    <Mail className="w-5 h-5 text-purple-400" />
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Email</p>
                    <p className="text-sm font-semibold text-white break-all">{user.email}</p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-xl bg-green-500/15 flex items-center justify-center">
                    <ShieldCheck className="w-5 h-5 text-green-400" />
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Status</p>
                    <p className="text-sm font-semibold text-green-400">Authenticated Successfully</p>
                  </div>
                </div>
              </div>
            </motion.div>

            {/* Security Status */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.25 }}
              className="glass-panel rounded-3xl p-6 border-white/10"
            >
              <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-widest mb-5 flex items-center gap-2">
                <Shield className="w-4 h-4" /> Security Checks Passed
              </h2>
              <div className="space-y-3">
                {[
                  { icon: <ScanFace className="w-4 h-4 text-blue-400" />, label: "Face Recognition", color: "bg-blue-500/15" },
                  { icon: <Activity className="w-4 h-4 text-purple-400" />, label: "Liveness Detection", color: "bg-purple-500/15" },
                  { icon: <Lock className="w-4 h-4 text-pink-400" />, label: "OTP Verification", color: "bg-pink-500/15" },
                ].map(({ icon, label, color }) => (
                  <div key={label} className="flex items-center gap-3 p-3 rounded-xl bg-white/5">
                    <div className={`w-8 h-8 rounded-lg ${color} flex items-center justify-center`}>
                      {icon}
                    </div>
                    <span className="text-sm text-white flex-1">{label}</span>
                    <div className="flex items-center gap-1.5 text-green-400 text-xs font-semibold">
                      <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                      Passed
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
          </div>

          {/* ── System Description ────────────────────────────────────────── */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.35 }}
            className="glass-panel rounded-3xl p-6 border-white/10"
          >
            <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-widest mb-3 flex items-center gap-2">
              <Shield className="w-4 h-4" /> About Your Session
            </h2>
            <p className="text-gray-300 text-sm leading-relaxed">
              You are logged in using a <span className="text-white font-semibold">secure multi-layer authentication system</span> including
              face recognition, liveness detection, and OTP verification. Your biometric data is processed
              locally on your device and never stored in plain form. Each login session requires a fresh
              liveness check, ensuring no replay attacks or spoofing attempts can succeed.
            </p>
          </motion.div>

          {/* ── Actions ───────────────────────────────────────────────────── */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.45 }}
            className="flex flex-col sm:flex-row gap-4"
          >
            <button
              onClick={handleLogout}
              className="flex items-center justify-center gap-2 px-6 py-3 rounded-2xl font-semibold text-sm text-white bg-red-500/15 hover:bg-red-500/25 border border-red-500/30 hover:border-red-500/50 transition-all duration-200 active:scale-95"
            >
              <LogOut className="w-4 h-4" />
              Logout
            </button>

            <button
              onClick={() => navigate("/login")}
              className="flex items-center justify-center gap-2 px-6 py-3 rounded-2xl font-semibold text-sm text-white bg-white/8 hover:bg-white/12 border border-white/15 hover:border-white/25 transition-all duration-200 active:scale-95"
            >
              <RefreshCw className="w-4 h-4" />
              Re-verify Face
            </button>
          </motion.div>

        </div>
      </main>
    </div>
  );
}

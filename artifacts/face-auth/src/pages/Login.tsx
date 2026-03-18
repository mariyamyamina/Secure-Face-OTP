import { motion } from "framer-motion";
import { Link } from "wouter";
import { LockKeyhole, ArrowLeft, ShieldCheck } from "lucide-react";
import { Navbar } from "@/components/layout/Navbar";

export default function Login() {
  return (
    <div className="min-h-screen flex flex-col relative overflow-hidden">
      <div className="fixed inset-0 z-[-2] w-full h-full opacity-20 pointer-events-none">
        <div className="absolute inset-0 bg-gradient-to-br from-background via-[#0a0f1d] to-background" />
        <div className="absolute top-[20%] left-[20%] w-[40rem] h-[40rem] bg-indigo-500/10 rounded-full blur-[120px]" />
        <div className="absolute bottom-[10%] right-[10%] w-[30rem] h-[30rem] bg-purple-500/10 rounded-full blur-[100px]" />
      </div>

      <Navbar />

      <main className="flex-1 flex items-center justify-center p-4">
        <motion.div 
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
          className="w-full max-w-md"
        >
          <div className="glass-panel p-10 rounded-3xl relative overflow-hidden border-indigo-500/20 shadow-[0_0_40px_rgba(79,70,229,0.15)] text-center">
            
            <div className="absolute top-0 inset-x-0 h-px bg-gradient-to-r from-transparent via-indigo-500/50 to-transparent" />
            
            <motion.div 
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.2 }}
              className="flex justify-center mb-8"
            >
              <div className="relative">
                <div className="absolute inset-0 bg-indigo-500/20 blur-xl rounded-full" />
                <div className="w-20 h-20 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-2xl flex items-center justify-center relative z-10 shadow-xl border border-white/10">
                  <LockKeyhole className="w-10 h-10 text-white" />
                </div>
              </div>
            </motion.div>

            <motion.h1 
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.3 }}
              className="text-3xl font-display font-bold text-white mb-4"
            >
              Biometric Login
            </motion.h1>
            
            <motion.div 
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.4 }}
              className="mb-8"
            >
              <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/10 text-sm text-indigo-300 font-medium">
                <ShieldCheck className="w-4 h-4" />
                Module Updating
              </span>
            </motion.div>

            <motion.p 
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.5 }}
              className="text-muted-foreground mb-10 leading-relaxed"
            >
              The authentication gateway is currently undergoing scheduled security enhancements. Our advanced liveness detection models are being calibrated.
            </motion.p>

            <motion.div
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.6 }}
            >
              <Link 
                href="/" 
                className="w-full flex items-center justify-center gap-2 px-6 py-3.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-white transition-all duration-200"
              >
                <ArrowLeft className="w-4 h-4" />
                Return to Base
              </Link>
            </motion.div>
          </div>
        </motion.div>
      </main>
    </div>
  );
}

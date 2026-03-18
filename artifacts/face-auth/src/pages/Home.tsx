import { motion } from "framer-motion";
import { Link } from "wouter";
import { Shield, ScanFace, Activity, Mail, LockKeyhole, Cpu, ArrowRight } from "lucide-react";
import { Navbar } from "@/components/layout/Navbar";
import { Footer } from "@/components/layout/Footer";

const features = [
  {
    icon: <ScanFace className="w-6 h-6 text-blue-400" />,
    title: "Face Recognition",
    description: "Highly accurate 128D facial descriptor matching using state-of-the-art neural networks."
  },
  {
    icon: <Activity className="w-6 h-6 text-purple-400" />,
    title: "Liveness Detection",
    description: "Advanced anti-spoofing algorithms detect eye blinks and micro-movements to prevent photo attacks."
  },
  {
    icon: <Mail className="w-6 h-6 text-pink-400" />,
    title: "Dynamic OTP",
    description: "Time-sensitive secondary validation sent directly to verified email addresses."
  },
  {
    icon: <LockKeyhole className="w-6 h-6 text-indigo-400" />,
    title: "Multi-layer Security",
    description: "Combining biometrics, cryptography, and possession factors for unbreakable defense."
  },
  {
    icon: <Cpu className="w-6 h-6 text-cyan-400" />,
    title: "Edge Processing",
    description: "Biometric templates are extracted securely on the client device before transmission."
  }
];

const steps = [
  {
    number: "01",
    title: "Enroll Identity",
    description: "Register with your email and capture a baseline facial scan securely via your webcam."
  },
  {
    number: "02",
    title: "Descriptor Extraction",
    description: "Your unique facial geometry is converted into an irreversible mathematical vector."
  },
  {
    number: "03",
    title: "Biometric Login",
    description: "Attempt login by presenting your face. The system matches the real-time scan to your vector."
  },
  {
    number: "04",
    title: "Liveness Verification",
    description: "The system actively verifies you are a live human, defeating printed photos or screens."
  },
  {
    number: "05",
    title: "OTP Validation",
    description: "A final dynamic passcode is required, completing the ultimate secure auth chain."
  }
];

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col relative overflow-hidden">
      {/* Background Hero Image */}
      <div className="fixed inset-0 z-[-2] w-full h-full opacity-30 mix-blend-screen pointer-events-none">
        <img 
          src={`${import.meta.env.BASE_URL}images/hero-bg.png`}
          alt="Abstract Security Background"
          className="w-full h-full object-cover"
        />
        <div className="absolute inset-0 bg-gradient-to-b from-background/40 via-background/80 to-background" />
      </div>

      <Navbar />

      <main className="flex-1 flex flex-col items-center pt-32 pb-24">
        {/* Hero Section */}
        <section className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center mt-12 md:mt-24 mb-32 relative">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.8, ease: "easeOut" }}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass-panel mb-8 border-blue-500/30 shadow-[0_0_30px_rgba(59,130,246,0.2)]"
          >
            <Shield className="w-4 h-4 text-blue-400" />
            <span className="text-sm font-medium text-blue-200">Military-Grade Security Protocol</span>
          </motion.div>

          <motion.h1 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.1 }}
            className="text-5xl md:text-7xl lg:text-8xl font-display font-extrabold tracking-tight mb-8 leading-[1.1]"
          >
            Intelligent Anti-Spoof <br />
            <span className="text-gradient">Face Authentication</span>
          </motion.h1>

          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.2 }}
            className="text-lg md:text-xl text-muted-foreground max-w-3xl mx-auto mb-12 leading-relaxed"
          >
            Secure your applications with the next generation of identity verification. 
            Combining AI-powered face recognition, active liveness detection, and dynamic OTP for impenetrable access control.
          </motion.p>

          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.3 }}
            className="flex flex-col sm:flex-row items-center justify-center gap-4"
          >
            <Link 
              href="/register" 
              className="px-8 py-4 rounded-xl font-semibold bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-[0_0_40px_rgba(99,102,241,0.4)] hover:shadow-[0_0_60px_rgba(99,102,241,0.6)] hover:-translate-y-1 transition-all duration-300 flex items-center gap-2"
            >
              Get Started Now
              <ArrowRight className="w-4 h-4" />
            </Link>
            <Link 
              href="/login" 
              className="px-8 py-4 rounded-xl font-semibold glass-panel hover:bg-white/10 text-white transition-all duration-300"
            >
              Try Demo Login
            </Link>
          </motion.div>
        </section>

        {/* Features Section */}
        <section className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24 relative">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-display font-bold mb-4">Defend Against Synthetic Identity</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">Our multi-modal approach ensures that the person authenticating is physically present and authorized.</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-100px" }}
                transition={{ duration: 0.5, delay: idx * 0.1 }}
                className="glass-panel p-8 rounded-2xl group hover:border-blue-500/30 hover:bg-white/5 transition-all duration-500"
              >
                <div className="w-14 h-14 rounded-xl bg-white/5 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-500">
                  {feature.icon}
                </div>
                <h3 className="text-xl font-bold mb-3 text-white">{feature.title}</h3>
                <p className="text-muted-foreground leading-relaxed text-sm">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </section>

        {/* How it Works Section */}
        <section className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24">
          <div className="text-center mb-20">
            <h2 className="text-3xl md:text-4xl font-display font-bold mb-4 text-gradient">The Authentication Chain</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">A seamless sequence designed for maximum security without compromising user experience.</p>
          </div>

          <div className="relative">
            {/* Connecting Line */}
            <div className="absolute top-1/2 left-0 w-full h-px bg-gradient-to-r from-transparent via-blue-500/30 to-transparent hidden lg:block -translate-y-1/2" />
            
            <div className="grid grid-cols-1 lg:grid-cols-5 gap-8">
              {steps.map((step, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.6, delay: idx * 0.15 }}
                  className="relative flex flex-col items-center text-center group"
                >
                  <div className="w-16 h-16 rounded-full glass-panel flex items-center justify-center font-display font-bold text-xl text-blue-300 border-blue-500/30 shadow-[0_0_20px_rgba(59,130,246,0.1)] mb-6 relative z-10 group-hover:bg-blue-500/20 group-hover:text-blue-100 transition-all duration-300">
                    {step.number}
                  </div>
                  <h4 className="text-lg font-bold text-white mb-2">{step.title}</h4>
                  <p className="text-sm text-muted-foreground leading-relaxed">{step.description}</p>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* CTA Section */}
        <section className="w-full max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-24">
          <div className="glass-panel p-12 md:p-16 rounded-3xl text-center relative overflow-hidden border-blue-500/20">
            <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full max-w-md h-32 bg-blue-500/20 blur-[100px] pointer-events-none" />
            
            <h2 className="text-3xl md:text-5xl font-display font-bold mb-6">Ready to secure your platform?</h2>
            <p className="text-lg text-muted-foreground mb-10 max-w-2xl mx-auto">
              Implement zero-trust architecture today. Experience the power of combined biometrics and dynamic OTP.
            </p>
            <Link 
              href="/register" 
              className="inline-flex px-10 py-4 rounded-xl font-bold bg-white text-black hover:bg-gray-200 transition-colors shadow-[0_0_30px_rgba(255,255,255,0.2)] hover:scale-105 active:scale-95 duration-200"
            >
              Start Enrollment
            </Link>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
}

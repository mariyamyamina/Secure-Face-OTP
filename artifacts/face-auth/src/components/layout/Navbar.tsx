import { Link, useLocation } from "wouter";
import { motion } from "framer-motion";
import { Shield, Menu, X } from "lucide-react";
import { useState, useEffect } from "react";
import { cn } from "@/lib/utils";

export function Navbar() {
  const [location] = useLocation();
  const [isScrolled, setIsScrolled] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 20);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const navLinks = [
    { name: "Home", path: "/" },
    { name: "Register", path: "/register" },
    { name: "Login", path: "/login" },
  ];

  return (
    <header
      className={cn(
        "fixed top-0 inset-x-0 z-50 transition-all duration-300",
        isScrolled ? "py-4" : "py-6"
      )}
    >
      <div className="container mx-auto px-4 md:px-6">
        <motion.div 
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.5 }}
          className={cn(
            "flex items-center justify-between rounded-2xl px-6 py-3 transition-all duration-300",
            isScrolled 
              ? "glass-panel shadow-[var(--shadow-glow)]" 
              : "bg-transparent border border-transparent"
          )}
        >
          <Link href="/" className="flex items-center gap-3 group">
            <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 shadow-lg group-hover:shadow-[var(--shadow-glow)] transition-all duration-300">
              <Shield className="w-5 h-5 text-white" />
            </div>
            <span className="font-display font-bold text-xl tracking-tight text-white group-hover:text-transparent group-hover:bg-clip-text group-hover:bg-gradient-to-r group-hover:from-blue-400 group-hover:to-purple-400 transition-all duration-300 hidden sm:block">
              AuraAuth
            </span>
          </Link>

          {/* Desktop Nav */}
          <nav className="hidden md:flex items-center gap-1">
            {navLinks.map((link) => {
              const isActive = location === link.path;
              return (
                <Link
                  key={link.path}
                  href={link.path}
                  className={cn(
                    "relative px-4 py-2 text-sm font-medium transition-colors duration-200 rounded-lg overflow-hidden",
                    isActive ? "text-white" : "text-muted-foreground hover:text-white"
                  )}
                >
                  {isActive && (
                    <motion.div
                      layoutId="navbar-indicator"
                      className="absolute inset-0 bg-white/10 rounded-lg"
                      initial={false}
                      transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                    />
                  )}
                  <span className="relative z-10">{link.name}</span>
                </Link>
              );
            })}
          </nav>

          <div className="hidden md:flex items-center">
            <Link 
              href="/register" 
              className="px-5 py-2.5 text-sm font-medium text-white bg-white/10 hover:bg-white/20 border border-white/10 rounded-xl transition-all duration-300 hover:shadow-[0_0_20px_rgba(255,255,255,0.1)] active:scale-95"
            >
              Get Started
            </Link>
          </div>

          {/* Mobile Toggle */}
          <button 
            className="md:hidden p-2 text-muted-foreground hover:text-white transition-colors"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          >
            {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </motion.div>

        {/* Mobile Menu */}
        {mobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="md:hidden mt-4 glass-panel rounded-2xl p-4 flex flex-col gap-2"
          >
            {navLinks.map((link) => (
              <Link
                key={link.path}
                href={link.path}
                onClick={() => setMobileMenuOpen(false)}
                className={cn(
                  "px-4 py-3 rounded-xl text-sm font-medium transition-colors",
                  location === link.path 
                    ? "bg-white/10 text-white" 
                    : "text-muted-foreground hover:bg-white/5 hover:text-white"
                )}
              >
                {link.name}
              </Link>
            ))}
          </motion.div>
        )}
      </div>
    </header>
  );
}

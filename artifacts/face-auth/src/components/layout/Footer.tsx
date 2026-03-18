import { Shield, Github, Twitter, Linkedin } from "lucide-react";
import { Link } from "wouter";

export function Footer() {
  return (
    <footer className="border-t border-white/10 bg-black/20 backdrop-blur-md mt-auto">
      <div className="container mx-auto px-4 md:px-6 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-8">
          <div className="col-span-1 md:col-span-2">
            <Link href="/" className="flex items-center gap-2 mb-4 group inline-flex">
              <Shield className="w-6 h-6 text-primary group-hover:text-blue-400 transition-colors" />
              <span className="font-display font-bold text-xl text-white">AuraAuth</span>
            </Link>
            <p className="text-muted-foreground text-sm max-w-xs leading-relaxed">
              Next-generation intelligent anti-spoof face authentication system protecting your digital identity with advanced AI liveness detection.
            </p>
          </div>
          
          <div>
            <h4 className="font-display font-semibold text-white mb-4">Product</h4>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li><Link href="/" className="hover:text-primary transition-colors">Features</Link></li>
              <li><Link href="/" className="hover:text-primary transition-colors">Security</Link></li>
              <li><Link href="/" className="hover:text-primary transition-colors">Documentation</Link></li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-display font-semibold text-white mb-4">Connect</h4>
            <div className="flex gap-4">
              <a href="#" className="w-10 h-10 rounded-full bg-white/5 flex items-center justify-center text-muted-foreground hover:bg-primary/20 hover:text-primary transition-all border border-white/10 hover:border-primary/50">
                <Github className="w-4 h-4" />
              </a>
              <a href="#" className="w-10 h-10 rounded-full bg-white/5 flex items-center justify-center text-muted-foreground hover:bg-primary/20 hover:text-primary transition-all border border-white/10 hover:border-primary/50">
                <Twitter className="w-4 h-4" />
              </a>
              <a href="#" className="w-10 h-10 rounded-full bg-white/5 flex items-center justify-center text-muted-foreground hover:bg-primary/20 hover:text-primary transition-all border border-white/10 hover:border-primary/50">
                <Linkedin className="w-4 h-4" />
              </a>
            </div>
          </div>
        </div>
        
        <div className="pt-8 border-t border-white/10 flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-xs text-muted-foreground">
            © {new Date().getFullYear()} AuraAuth Systems. All rights reserved.
          </p>
          <div className="flex gap-6 text-xs text-muted-foreground">
            <a href="#" className="hover:text-white transition-colors">Privacy Policy</a>
            <a href="#" className="hover:text-white transition-colors">Terms of Service</a>
          </div>
        </div>
      </div>
    </footer>
  );
}

import { Link } from "wouter";
import { AlertCircle } from "lucide-react";
import { Navbar } from "@/components/layout/Navbar";

export default function NotFound() {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <main className="flex-1 flex items-center justify-center p-4">
        <div className="glass-panel p-12 rounded-3xl text-center max-w-md border-red-500/20">
          <div className="flex justify-center mb-6">
            <div className="w-20 h-20 rounded-full bg-red-500/10 flex items-center justify-center">
              <AlertCircle className="w-10 h-10 text-red-500" />
            </div>
          </div>
          <h1 className="text-4xl font-display font-bold text-white mb-4">404</h1>
          <p className="text-muted-foreground mb-8">
            The requested sector does not exist or access is restricted.
          </p>
          <Link 
            href="/" 
            className="inline-flex px-6 py-3 rounded-xl font-medium bg-white/10 hover:bg-white/20 text-white transition-colors border border-white/10"
          >
            Return to Dashboard
          </Link>
        </div>
      </main>
    </div>
  );
}

import { Crosshair } from "lucide-react";
import Link from "next/link";

export function Topbar({ signedIn = false, mode = "offline" }: { signedIn?: boolean; mode?: "offline" | "live" }) {
  return (
    <header className="topbar">
      <div className="container topbar-inner">
        <Link className="brand" href="/">
          <span className="brand-mark"><Crosshair size={20} strokeWidth={2.6} /></span>
          <span>PitchVision</span>
        </Link>
        <nav className="nav-actions">
          <Link className={`button ${mode === "live" ? "button-primary" : "button-ghost"}`} href="/live">LIVE INPUT</Link>
          <Link className={`button ${mode === "offline" ? "button-primary" : "button-secondary"}`} href={signedIn ? "/projects" : "/"}>{signedIn ? "PROJECTS" : "OFFLINE WORKSPACE"}</Link>
        </nav>
      </div>
    </header>
  );
}

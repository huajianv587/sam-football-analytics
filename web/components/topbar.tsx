import { Crosshair } from "lucide-react";
import Link from "next/link";

export function Topbar({ signedIn = false }: { signedIn?: boolean }) {
  return (
    <header className="topbar">
      <div className="container topbar-inner">
        <Link className="brand" href="/">
          <span className="brand-mark"><Crosshair size={20} strokeWidth={2.6} /></span>
          <span>PitchVision</span>
        </Link>
        <nav className="nav-actions">
          <span className="button button-ghost">LIVE INPUT / ROADMAP</span>
          <Link className="button button-primary" href={signedIn ? "/projects" : "/"}>{signedIn ? "PROJECTS" : "OFFLINE WORKSPACE"}</Link>
        </nav>
      </div>
    </header>
  );
}

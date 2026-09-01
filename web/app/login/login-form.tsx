"use client";

import { LockKeyhole } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { hasSupabaseConfig } from "@/lib/supabase/config";
import { createClient } from "@/lib/supabase/client";

export function LoginForm() {
  const router = useRouter();
  const configured = hasSupabaseConfig();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (!configured) return;
    setLoading(true);
    setError("");
    const { error: signInError } = await createClient().auth.signInWithPassword({ email, password });
    setLoading(false);
    if (signInError) return setError(signInError.message);
    router.push("/projects");
    router.refresh();
  }

  return (
    <section className="auth-card">
      <span className="brand-mark"><LockKeyhole size={19} /></span>
      <h1>Administrator access</h1>
      <p>Manage match footage and A40 analysis jobs.</p>
      <form className="form-stack" onSubmit={submit}>
        <div className="field"><label htmlFor="email">EMAIL</label><input id="email" className="input" type="email" value={email} onChange={(event) => setEmail(event.target.value)} required /></div>
        <div className="field"><label htmlFor="password">PASSWORD</label><input id="password" className="input" type="password" value={password} onChange={(event) => setPassword(event.target.value)} required /></div>
        {error && <p className="form-error">{error}</p>}
        <button className="button button-primary" disabled={!configured || loading}>{loading ? "SIGNING IN…" : "SIGN IN"}</button>
      </form>
      {!configured && <div className="setup-note">Supabase is not connected. Add the project URL and publishable key to <code>web/.env.local</code>.</div>}
    </section>
  );
}

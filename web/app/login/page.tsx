import { Topbar } from "@/components/topbar";
import { LoginForm } from "./login-form";

export default function LoginPage() {
  return <main className="page-shell"><Topbar /><div className="auth-layout"><LoginForm /></div></main>;
}

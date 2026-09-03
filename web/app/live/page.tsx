import { Topbar } from "@/components/topbar";
import { LiveAnalyzer } from "./live-analyzer";

export default function LivePage() {
  return <main className="page-shell"><Topbar mode="live" /><LiveAnalyzer /></main>;
}

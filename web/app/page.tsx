import { Topbar } from "@/components/topbar";
import { OfflineAnalyzer } from "./offline-analyzer";

export default function Home() {
  return <main className="page-shell"><Topbar /><OfflineAnalyzer /></main>;
}

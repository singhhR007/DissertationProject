import { Route, Routes } from "react-router";
import AppShell from "./components/layout/AppShell";
import DashboardPage from "./pages/DashboardPage";
import AnalysisPage from "./pages/AnalysisPage";
import EvaluationPage from "./pages/EvaluationPage";
import SystemStatusPage from "./pages/SystemStatusPage";

export default function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/analysis" element={<AnalysisPage />} />
        <Route path="/evaluation" element={<EvaluationPage />} />
        <Route path="/system-status" element={<SystemStatusPage />} />
      </Routes>
    </AppShell>
  );
}
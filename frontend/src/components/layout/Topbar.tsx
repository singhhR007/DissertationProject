import { useQuery } from "@tanstack/react-query";
import { getHealth } from "../../lib/api";
import { PROJECT_META } from "../../lib/project";
import Badge from "../ui/Badge";

type TopbarProps = {
  onOpenMenu: () => void;
};

function normalizeStatus(status?: string) {
  return status?.trim().toLowerCase();
}

export default function Topbar({ onOpenMenu }: TopbarProps) {
  const healthQuery = useQuery({
    queryKey: ["topbar-health"],
    queryFn: getHealth,
    retry: false,
    refetchInterval: 30000,
  });

  const normalizedStatus = normalizeStatus(healthQuery.data?.status);
  const isHealthy =
    normalizedStatus === "ok" ||
    normalizedStatus === "healthy" ||
    normalizedStatus === "ready";

  return (
    <header className="sticky top-0 z-20 border-b border-slate-200 bg-white/90 backdrop-blur">
      <div className="flex min-h-20 items-center justify-between gap-4 px-4 sm:px-6 lg:px-8">
        <div className="flex min-w-0 items-center gap-3">
          <button
            type="button"
            onClick={onOpenMenu}
            className="inline-flex h-10 w-10 items-center justify-center rounded-xl border border-slate-200 text-slate-700 lg:hidden"
            aria-label="Open navigation"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-5 w-5"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M4 7h16M4 12h16M4 17h16" />
            </svg>
          </button>

          <div className="min-w-0">
            <p className="text-sm font-medium text-slate-500">Bachelor Project</p>
            <h2 className="truncate text-lg font-semibold tracking-tight text-slate-950">
              {PROJECT_META.uiTitle}
            </h2>
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-end gap-2">
          <Badge variant="info" className="hidden sm:inline-flex">
            {PROJECT_META.artefactLabel}
          </Badge>

          <Badge variant="neutral" className="hidden md:inline-flex">
            {PROJECT_META.environmentLabel}
          </Badge>

          {healthQuery.isLoading ? (
            <Badge variant="warning">Checking API...</Badge>
          ) : healthQuery.isError ? (
            <Badge variant="danger">API unreachable</Badge>
          ) : isHealthy ? (
            <Badge variant="success">API connected</Badge>
          ) : (
            <Badge variant="warning">API responded</Badge>
          )}
        </div>
      </div>
    </header>
  );
}
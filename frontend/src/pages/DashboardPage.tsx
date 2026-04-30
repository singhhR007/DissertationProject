import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getHealth, getModelInfo } from "../lib/api";
import { loadLastPrediction } from "../lib/lastPrediction";
import { PROJECT_META } from "../lib/project";
import Badge from "../components/ui/Badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../components/ui/Card";
import PageHeader from "../components/ui/PageHeader";
import StatCard from "../components/ui/StatCard";

function normalizeStatus(status?: string) {
  return status?.trim().toLowerCase();
}

function formatThreshold(value?: number) {
  if (value === undefined || Number.isNaN(value)) return "Not reported";
  return value.toFixed(4);
}

function formatRisk(value?: number) {
  if (value === undefined || Number.isNaN(value)) return "—";
  return value.toFixed(6);
}

function formatTimestampToLocal(value?: string) {
  if (!value) return "—";

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return new Intl.DateTimeFormat("de-DE", {
    dateStyle: "medium",
    timeStyle: "medium",
  }).format(date);
}

export default function DashboardPage() {
  const [lastPrediction] = useState(() => loadLastPrediction());

  const healthQuery = useQuery({
    queryKey: ["dashboard-health"],
    queryFn: getHealth,
    retry: false,
  });

  const modelInfoQuery = useQuery({
    queryKey: ["dashboard-model-info"],
    queryFn: getModelInfo,
    retry: false,
  });

  const normalizedStatus = normalizeStatus(healthQuery.data?.status);
  const isHealthy =
    normalizedStatus === "ok" ||
    normalizedStatus === "healthy" ||
    normalizedStatus === "ready";

  const dashboardStatusBadge = useMemo(() => {
    if (healthQuery.isLoading) {
      return <Badge variant="warning">Checking system</Badge>;
    }

    if (healthQuery.isError) {
      return <Badge variant="danger">Connection issue</Badge>;
    }

    if (isHealthy) {
      return <Badge variant="success">System healthy</Badge>;
    }

    return <Badge variant="warning">API responded</Badge>;
  }, [healthQuery.isLoading, healthQuery.isError, isHealthy]);

  return (
    <div className="space-y-8">
      <PageHeader
        eyebrow="Overview"
        title="Dashboard"
        description="This interface presents the dissertation artefact through a clean monitoring and inference oriented workflow. It is designed for local testing, demonstration and evaluation."
        actions={dashboardStatusBadge}
      />

      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
        <StatCard
          label="System status"
          value={
            healthQuery.isLoading
              ? "Checking..."
              : healthQuery.isError
              ? "Unavailable"
              : isHealthy
              ? "Healthy"
              : String(healthQuery.data?.status ?? "Unknown")
          }
          helper="Live health status retrieved from the local API."
          trend="Live status"
          trendVariant={healthQuery.isError ? "danger" : "success"}
        />

        <StatCard
          label="Model version"
          value={
            modelInfoQuery.isLoading
              ? "Loading..."
              : String(modelInfoQuery.data?.model_version ?? "Not reported")
          }
          helper="Active model metadata returned by the backend."
          trend="Model info"
          trendVariant={modelInfoQuery.isError ? "warning" : "info"}
        />

        <StatCard
          label="Threshold"
          value={
            modelInfoQuery.isLoading
              ? "Loading..."
              : formatThreshold(modelInfoQuery.data?.threshold)
          }
          helper="Current decision threshold exposed by the model info endpoint."
          trend="Runtime config"
          trendVariant={modelInfoQuery.isError ? "warning" : "neutral"}
        />

        <StatCard
          label="Last prediction"
          value={lastPrediction?.prediction ?? "None"}
          helper="Most recent prediction stored locally from the analysis page."
          trend={lastPrediction ? "Stored locally" : "No recent result"}
          trendVariant={
            !lastPrediction
              ? "neutral"
              : lastPrediction.prediction === "anomalous"
              ? "danger"
              : "success"
          }
        />

        <StatCard
          label="Last risk score"
          value={formatRisk(lastPrediction?.riskScore)}
          helper="Most recent risk score returned by the prediction endpoint."
          trend={lastPrediction ? "Latest response" : "No recent result"}
          trendVariant={lastPrediction ? "info" : "neutral"}
        />

        <StatCard
          label="Last input mode"
          value={lastPrediction?.inputMode ?? "None"}
          helper="Shows whether the latest request used JSON or raw text."
          trend={PROJECT_META.environmentLabel}
          trendVariant="neutral"
        />
      </section>

      <section className="grid gap-6 xl:grid-cols-[1.3fr_0.7fr]">
        <Card>
          <CardHeader>
            <CardTitle>Most recent prediction</CardTitle>
            <CardDescription>
              Latest analysis result stored in the browser for dashboard display.
            </CardDescription>
          </CardHeader>

          <CardContent className="space-y-4">
            {lastPrediction ? (
              <>
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="rounded-2xl border border-slate-200 p-4">
                    <p className="text-sm text-slate-500">Prediction</p>
                    <p className="mt-2 text-xl font-semibold text-slate-950">
                      {lastPrediction.prediction}
                    </p>
                  </div>

                  <div className="rounded-2xl border border-slate-200 p-4">
                    <p className="text-sm text-slate-500">Risk score</p>
                    <p className="mt-2 text-xl font-semibold text-slate-950">
                      {formatRisk(lastPrediction.riskScore)}
                    </p>
                  </div>

                  <div className="rounded-2xl border border-slate-200 p-4">
                    <p className="text-sm text-slate-500">Source</p>
                    <p className="mt-2 break-words text-base font-semibold text-slate-900">
                      {lastPrediction.source}
                    </p>
                  </div>

                  <div className="rounded-2xl border border-slate-200 p-4">
                    <p className="text-sm text-slate-500">Sequence ID</p>
                    <p className="mt-2 break-words text-base font-semibold text-slate-900">
                      {lastPrediction.sequenceId}
                    </p>
                  </div>
                </div>

                <div className="rounded-2xl border border-slate-200 p-4">
                  <p className="text-sm text-slate-500">Processed at</p>
                  <p className="mt-2 text-base font-semibold text-slate-900">
                    {formatTimestampToLocal(lastPrediction.processedAt)}
                  </p>
                  <p className="mt-2 text-xs text-slate-500">
                    UTC: {lastPrediction.processedAt}
                  </p>
                </div>
              </>
            ) : (
              <div className="rounded-2xl border border-dashed border-slate-300 bg-slate-50 p-6 text-sm text-slate-600">
                No prediction has been stored yet. Run a request from the Analysis page
                to populate the dashboard with the latest result.
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Project objective</CardTitle>
            <CardDescription>
              The frontend is intended to make the artefact easier to operate,
              interpret and demonstrate.
            </CardDescription>
          </CardHeader>

          <CardContent className="space-y-4 text-sm leading-7 text-slate-600">
            <p>{PROJECT_META.dissertationTitle}</p>
            <p>
              The dashboard combines live backend status with the latest locally stored
              prediction result so that demonstrations can quickly move between
              analysis, result inspection and overview.
            </p>
          </CardContent>
        </Card>
      </section>
    </div>
  );
}
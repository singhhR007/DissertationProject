import { useQuery } from "@tanstack/react-query";
import PageHeader from "../components/ui/PageHeader";
import Badge from "../components/ui/Badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/Card";
import { API_BASE, HAS_DEV_TOKEN, getHealth, getModelInfo } from "../lib/api";

function normalizeStatus(status?: string) {
  return status?.trim().toLowerCase();
}

function formatThreshold(value?: number) {
  if (value === undefined || Number.isNaN(value)) return "Not reported";
  return value.toFixed(6);
}

export default function SystemStatusPage() {
  const healthQuery = useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    retry: false,
  });

  const modelInfoQuery = useQuery({
    queryKey: ["model-info"],
    queryFn: getModelInfo,
    retry: false,
  });

  const normalizedStatus = normalizeStatus(healthQuery.data?.status);
  const isHealthy =
    normalizedStatus === "ok" ||
    normalizedStatus === "healthy" ||
    normalizedStatus === "ready";

  return (
    <div className="space-y-8">
      <PageHeader
        eyebrow="Operations"
        title="System status"
        description="This page shows API availability, authentication readiness and model metadata from the backend."
        actions={
          healthQuery.isLoading ? (
            <Badge variant="warning">Checking API...</Badge>
          ) : healthQuery.isError ? (
            <Badge variant="danger">API unreachable</Badge>
          ) : isHealthy ? (
            <Badge variant="success">API healthy</Badge>
          ) : (
            <Badge variant="warning">API responded</Badge>
          )
        }
      />

      <section className="grid gap-6 md:grid-cols-2 xl:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle>API availability</CardTitle>
            <CardDescription>Connectivity to the local backend service.</CardDescription>
          </CardHeader>

          <CardContent className="space-y-3">
            {healthQuery.isLoading && <Badge variant="warning">Checking...</Badge>}
            {healthQuery.isError && <Badge variant="danger">Unavailable</Badge>}
            {!healthQuery.isLoading && !healthQuery.isError && (
              <Badge variant={isHealthy ? "success" : "warning"}>
                {healthQuery.data?.status ?? "Unknown"}
              </Badge>
            )}

            <p className="text-sm leading-6 text-slate-600">
              {healthQuery.isError
                ? "The frontend could not reach the local API. Check whether the Python service is running and whether the proxy configuration is correct."
                : "The frontend successfully contacted the backend service."}
            </p>

            <div>
              <p className="text-sm text-slate-500">API base</p>
              <p className="mt-1 break-all text-base font-semibold text-slate-900">
                {API_BASE}
              </p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Authentication</CardTitle>
            <CardDescription>Bearer token usage in the local environment.</CardDescription>
          </CardHeader>

          <CardContent className="space-y-3">
            <Badge variant={HAS_DEV_TOKEN ? "neutral" : "danger"}>
              {HAS_DEV_TOKEN ? "Local dev token configured" : "No token configured"}
            </Badge>

            <p className="text-sm leading-6 text-slate-600">
              Protected endpoints such as model information and predictions use the
              bearer token defined in the local development environment.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Model service</CardTitle>
            <CardDescription>Readiness and metadata from the backend model endpoint.</CardDescription>
          </CardHeader>

          <CardContent className="space-y-3">
            {modelInfoQuery.isLoading && (
              <Badge variant="warning">Loading model info...</Badge>
            )}

            {modelInfoQuery.isError && (
              <Badge variant="danger">Model info unavailable</Badge>
            )}

            {!modelInfoQuery.isLoading && !modelInfoQuery.isError && (
              <Badge variant="success">Model info loaded</Badge>
            )}

            <div>
              <p className="text-sm text-slate-500">Model version</p>
              <p className="mt-1 text-base font-semibold text-slate-900">
                {String(modelInfoQuery.data?.model_version ?? "Not reported")}
              </p>
            </div>

            <div>
              <p className="text-sm text-slate-500">Threshold</p>
              <p className="mt-1 text-base font-semibold text-slate-900">
                {formatThreshold(modelInfoQuery.data?.threshold)}
              </p>
            </div>
          </CardContent>
        </Card>
      </section>
    </div>
  );
}
import { useMemo, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import PageHeader from "../components/ui/PageHeader";
import Badge from "../components/ui/Badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../components/ui/Card";
import { runPrediction, runRawTextPrediction } from "../lib/api";
import { saveLastPrediction } from "../lib/lastPrediction";
import type { PredictionRequest, PredictionResponse } from "../types/api";

type InputMode = "json" | "raw";

const DEFAULT_REQUEST = `{
  "context": {
    "environment": "lab"
  },
  "log_sequence": {
    "events": [
      {
        "component": "DataNode",
        "host": "10.250.19.102",
        "message": "Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010",
        "service": "hdfs",
        "severity": "info"
      },
      {
        "component": "DataNode",
        "host": "10.250.19.102",
        "message": "PacketResponder 1 for block blk_-1608999687919862906 terminating",
        "service": "hdfs",
        "severity": "info"
      }
    ],
    "sequence_id": "blk_-1608999687919862906"
  },
  "source": "hdfs-node-01",
  "telemetry_schema_version": "log_sequence_v1",
  "telemetry_type": "log_sequence",
  "timestamp": "2026-04-03T12:30:00Z"
}`;

const DEFAULT_RAW_TEXT = `INFO dfs.DataNode$DataXceiver: Receiving block blk_750348333 src: /10.250.19.102:54106 dest: /10.250.19.102:50010
INFO dfs.DataNode: PacketResponder 0 for block blk_750348333 terminating
INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated
INFO dfs.DataNode: Received block blk_750348333 of size 67108864 from /10.250.19.102
INFO dfs.FSDataset: Finalized block blk_750348333`;

function formatJson(value: unknown) {
  return JSON.stringify(value, null, 2);
}

function formatNumber(value?: number) {
  return value !== undefined ? value.toFixed(6) : "—";
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

function getSummaryFromResponse(
  response: PredictionResponse | undefined,
  sequenceIdFallback: string
) {
  return {
    prediction: response?.prediction ?? "—",
    riskScore: formatNumber(response?.risk_score),
    threshold: formatNumber(response?.threshold),
    sequenceId: sequenceIdFallback || "—",
    processedAtLocal: formatTimestampToLocal(response?.processed_at),
    processedAtRaw: response?.processed_at ?? "—",
  };
}

export default function AnalysisPage() {
  const [mode, setMode] = useState<InputMode>("json");
  const [requestBody, setRequestBody] = useState(DEFAULT_REQUEST);
  const [rawTextBody, setRawTextBody] = useState(DEFAULT_RAW_TEXT);
  const [inputError, setInputError] = useState<string | null>(null);

  const parsedRequest = useMemo(() => {
    try {
      return JSON.parse(requestBody) as PredictionRequest;
    } catch {
      return null;
    }
  }, [requestBody]);

  const jsonPredictionMutation = useMutation({
    mutationFn: (payload: PredictionRequest) => runPrediction(payload),
    onSuccess: (data, variables) => {
      saveLastPrediction({
        inputMode: "JSON",
        prediction: data.prediction,
        riskScore: data.risk_score,
        threshold: data.threshold,
        processedAt: data.processed_at,
        modelVersion: data.model_version,
        sequenceId: variables.log_sequence.sequence_id,
        source: variables.source,
        savedAt: new Date().toISOString(),
      });
    },
  });

  const rawPredictionMutation = useMutation({
    mutationFn: (payload: string) => runRawTextPrediction(payload),
    onSuccess: (data) => {
      saveLastPrediction({
        inputMode: "Raw text",
        prediction: data.prediction,
        riskScore: data.risk_score,
        threshold: data.threshold,
        processedAt: data.processed_at,
        modelVersion: data.model_version,
        sequenceId: "Parsed internally from raw text",
        source: "frontend-raw",
        savedAt: new Date().toISOString(),
      });
    },
  });

  const activeMutation =
    mode === "json" ? jsonPredictionMutation : rawPredictionMutation;

  const handleSubmit = () => {
    setInputError(null);

    if (mode === "json") {
      if (!parsedRequest) {
        setInputError("The request body is not valid JSON.");
        return;
      }

      jsonPredictionMutation.mutate(parsedRequest);
      return;
    }

    if (!rawTextBody.trim()) {
      setInputError("The raw text input is empty.");
      return;
    }

    rawPredictionMutation.mutate(rawTextBody);
  };

  const handleReset = () => {
    setInputError(null);
    jsonPredictionMutation.reset();
    rawPredictionMutation.reset();
    setRequestBody(DEFAULT_REQUEST);
    setRawTextBody(DEFAULT_RAW_TEXT);
  };

  const sequenceIdFallback =
    mode === "json"
      ? parsedRequest?.log_sequence?.sequence_id ?? ""
      : "Parsed internally from raw text";

  const summary = useMemo(() => {
    return getSummaryFromResponse(activeMutation.data, sequenceIdFallback);
  }, [activeMutation.data, sequenceIdFallback]);

  return (
    <div className="space-y-8">
      <PageHeader
        eyebrow="Core workflow"
        title="Log analysis"
        description="This page demonstrates the local REST API by sending either a structured JSON request or a raw-log-text request and displaying the returned prediction response."
        actions={
          activeMutation.isPending ? (
            <Badge variant="warning">Submitting request...</Badge>
          ) : activeMutation.isError ? (
            <Badge variant="danger">Request failed</Badge>
          ) : activeMutation.isSuccess ? (
            <Badge variant="success">Response received</Badge>
          ) : mode === "json" ? (
            <Badge variant="info">JSON mode</Badge>
          ) : (
            <Badge variant="info">Raw text mode</Badge>
          )
        }
      />

      <div className="grid gap-6 xl:grid-cols-[1.15fr_0.85fr]">
        <Card>
          <CardHeader>
            <CardTitle>Request body</CardTitle>
            <CardDescription>
              Switch between structured JSON and raw text input to test both API
              input modes.
            </CardDescription>
          </CardHeader>

          <CardContent className="space-y-4">
            <div className="flex flex-wrap gap-3">
              <button
                type="button"
                onClick={() => setMode("json")}
                className={
                  mode === "json"
                    ? "inline-flex items-center rounded-xl bg-slate-900 px-4 py-2 text-sm font-medium text-white"
                    : "inline-flex items-center rounded-xl border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
                }
              >
                JSON
              </button>

              <button
                type="button"
                onClick={() => setMode("raw")}
                className={
                  mode === "raw"
                    ? "inline-flex items-center rounded-xl bg-slate-900 px-4 py-2 text-sm font-medium text-white"
                    : "inline-flex items-center rounded-xl border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
                }
              >
                Raw text
              </button>
            </div>

            {mode === "json" ? (
              <div>
                <label className="mb-2 block text-sm font-medium text-slate-700">
                  Prediction request JSON
                </label>

                <textarea
                  value={requestBody}
                  onChange={(e) => setRequestBody(e.target.value)}
                  rows={18}
                  spellCheck={false}
                  className="w-full rounded-2xl border border-slate-300 bg-white px-4 py-3 font-mono text-sm text-slate-800 outline-none transition focus:border-slate-400"
                />
              </div>
            ) : (
              <div className="space-y-3">
                <div className="rounded-2xl border border-amber-200 bg-amber-50 p-4">
                  <p className="text-sm font-medium text-amber-900">
                    Raw text input
                  </p>
                  <p className="mt-2 text-sm leading-6 text-amber-800">
                    Raw log text can be pasted directly for quick testing.
                    Internally, the frontend still submits a JSON request and
                    passes the text in the{" "}
                    <code className="rounded bg-amber-100 px-1 py-0.5">
                      raw_log_text
                    </code>{" "}
                    field. The most reliable behaviour is generally expected when
                    logs resemble formats seen during training, especially HDFS
                    and OpenStack.
                  </p>
                </div>

                <div>
                  <label className="mb-2 block text-sm font-medium text-slate-700">
                    Raw log text
                  </label>

                  <textarea
                    value={rawTextBody}
                    onChange={(e) => setRawTextBody(e.target.value)}
                    rows={18}
                    spellCheck={false}
                    className="w-full rounded-2xl border border-slate-300 bg-white px-4 py-3 font-mono text-sm text-slate-800 outline-none transition focus:border-slate-400"
                  />
                </div>
              </div>
            )}

            {inputError ? (
              <div className="rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
                {inputError}
              </div>
            ) : null}

            {activeMutation.isError ? (
              <div className="rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
                {(activeMutation.error as Error).message}
              </div>
            ) : null}

            <div className="flex flex-wrap gap-3">
              <button
                type="button"
                onClick={handleSubmit}
                disabled={activeMutation.isPending}
                className="inline-flex items-center rounded-xl bg-slate-900 px-5 py-3 text-sm font-medium text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {activeMutation.isPending ? "Running prediction..." : "Run prediction"}
              </button>

              <button
                type="button"
                onClick={handleReset}
                className="inline-flex items-center rounded-xl border border-slate-300 bg-white px-5 py-3 text-sm font-medium text-slate-700 transition hover:bg-slate-50"
              >
                Reset example
              </button>
            </div>
          </CardContent>
        </Card>

        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Prediction summary</CardTitle>
              <CardDescription>
                Key values extracted from the response.
              </CardDescription>
            </CardHeader>

            <CardContent className="space-y-4">
              <div className="rounded-2xl border border-slate-200 p-4">
                <p className="text-sm text-slate-500">Prediction</p>
                <p className="mt-2 text-2xl font-semibold text-slate-950">
                  {summary.prediction}
                </p>
              </div>

              <div className="grid gap-4 sm:grid-cols-2">
                <div className="rounded-2xl border border-slate-200 p-4">
                  <p className="text-sm text-slate-500">Risk score</p>
                  <p className="mt-2 text-lg font-semibold text-slate-900">
                    {summary.riskScore}
                  </p>
                </div>

                <div className="rounded-2xl border border-slate-200 p-4">
                  <p className="text-sm text-slate-500">Threshold</p>
                  <p className="mt-2 text-lg font-semibold text-slate-900">
                    {summary.threshold}
                  </p>
                </div>

                <div className="rounded-2xl border border-slate-200 p-4">
                  <p className="text-sm text-slate-500">Sequence ID</p>
                  <p className="mt-2 break-words text-lg font-semibold text-slate-900">
                    {summary.sequenceId}
                  </p>
                </div>

                <div className="rounded-2xl border border-slate-200 p-4">
                  <p className="text-sm text-slate-500">Processed at</p>
                  <p className="mt-2 break-words text-lg font-semibold text-slate-900">
                    {summary.processedAtLocal}
                  </p>
                  <p className="mt-2 break-words text-xs text-slate-500">
                    UTC: {summary.processedAtRaw}
                  </p>
                </div>
              </div>

              {activeMutation.data?.advisory ? (
                <div className="rounded-2xl border border-blue-200 bg-blue-50 p-4">
                  <p className="text-sm font-medium text-blue-900">Advisory</p>
                  <p className="mt-2 text-sm leading-6 text-blue-800">
                    {activeMutation.data.advisory}
                  </p>
                </div>
              ) : null}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Raw API response</CardTitle>
              <CardDescription>
                Full response payload returned by the backend.
              </CardDescription>
            </CardHeader>

            <CardContent>
              <div className="overflow-auto rounded-2xl border border-slate-200 bg-slate-950 p-4">
                <pre className="whitespace-pre-wrap break-words font-mono text-sm text-slate-100">
                  {activeMutation.data
                    ? formatJson(activeMutation.data)
                    : "Run a prediction to see the live response here."}
                </pre>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
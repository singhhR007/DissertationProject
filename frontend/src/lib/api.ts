import type {
  HealthResponse,
  ModelInfoResponse,
  PredictionRequest,
  PredictionResponse,
} from "../types/api";

export const API_BASE = import.meta.env.VITE_API_BASE ?? "/api";
export const HAS_DEV_TOKEN = Boolean(import.meta.env.VITE_DEV_BEARER_TOKEN);
const DEV_TOKEN = import.meta.env.VITE_DEV_BEARER_TOKEN ?? "";

async function apiRequest<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      ...(DEV_TOKEN ? { Authorization: `Bearer ${DEV_TOKEN}` } : {}),
      ...(options.headers ?? {}),
    },
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`API request failed (${response.status}): ${text}`);
  }

  return response.json() as Promise<T>;
}

export async function getHealth(): Promise<HealthResponse> {
  return apiRequest<HealthResponse>("/v1/health", {
    method: "GET",
  });
}

export async function getModelInfo(): Promise<ModelInfoResponse> {
  return apiRequest<ModelInfoResponse>("/v1/model/info", {
    method: "GET",
  });
}

export async function runPrediction(
  payload: PredictionRequest
): Promise<PredictionResponse> {
  return apiRequest<PredictionResponse>("/v1/predictions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

export async function runRawTextPrediction(
  rawText: string
): Promise<PredictionResponse> {
  return apiRequest<PredictionResponse>("/v1/predictions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      raw_log_text: rawText,
      timestamp: new Date().toISOString(),
      source: "frontend-raw",
      telemetry_type: "log_sequence",
      telemetry_schema_version: "log_sequence_v1",
    }),
  });
}
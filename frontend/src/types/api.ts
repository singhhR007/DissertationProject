export type PredictionLabel = "normal" | "anomalous";

export type LogEvent = {
  component: string;
  host: string;
  message: string;
  service: string;
  severity: string;
};

export type PredictionRequest = {
  context?: {
    environment?: string;
    [key: string]: unknown;
  };
  log_sequence: {
    events: LogEvent[];
    sequence_id: string;
  };
  source: string;
  telemetry_schema_version: string;
  telemetry_type: string;
  timestamp: string;
};

export type PredictionResponse = {
  request_id: string;
  prediction: PredictionLabel;
  risk_score: number;
  threshold: number;
  score_type: string;
  model_version: string;
  processed_at: string;
  advisory: string;
};

export type HealthResponse = {
  status: string;
  message?: string;
};

export type ModelInfoResponse = {
  model_version?: string;
  threshold?: number;
  [key: string]: unknown;
};

export type LastPredictionRecord = {
  inputMode: "JSON" | "Raw text";
  prediction: PredictionLabel;
  riskScore: number;
  threshold: number;
  processedAt: string;
  modelVersion: string;
  sequenceId: string;
  source: string;
  savedAt: string;
};
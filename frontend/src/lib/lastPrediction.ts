import type { LastPredictionRecord } from "../types/api";

const LAST_PREDICTION_KEY = "lad:last-prediction";

export function saveLastPrediction(record: LastPredictionRecord) {
  localStorage.setItem(LAST_PREDICTION_KEY, JSON.stringify(record));
}

export function loadLastPrediction(): LastPredictionRecord | null {
  const raw = localStorage.getItem(LAST_PREDICTION_KEY);

  if (!raw) {
    return null;
  }

  try {
    return JSON.parse(raw) as LastPredictionRecord;
  } catch {
    localStorage.removeItem(LAST_PREDICTION_KEY);
    return null;
  }
}

export function clearLastPrediction() {
  localStorage.removeItem(LAST_PREDICTION_KEY);
}

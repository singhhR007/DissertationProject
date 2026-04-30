from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

from app.services.preprocessing import (
    LabeledSequenceRecord,
    NormalizedLogSequence,
    build_hdfs_sequences_from_files,
)


# Shared helpers

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
EVENT_TOKEN = "<E>"
PAD_INDEX = 0
UNK_INDEX = 1
EVENT_INDEX = 2


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_sequence_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def label_to_binary(label: str) -> int:
    return 1 if label == "anomalous" else 0


def binary_to_label(value: int) -> str:
    return "anomalous" if int(value) == 1 else "normal"


def sanitize_model_key(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("-", "_")
    )


def build_sequence_lookup(
    records: list[LabeledSequenceRecord],
) -> dict[str, LabeledSequenceRecord]:
    return {record.sequence.sequence_id: record for record in records}


def select_records_by_ids(
    sequence_lookup: dict[str, LabeledSequenceRecord],
    sequence_ids: list[str],
) -> list[LabeledSequenceRecord]:
    missing = [sequence_id for sequence_id in sequence_ids if sequence_id not in sequence_lookup]
    if missing:
        raise KeyError(
            f"Missing {len(missing)} sequence ids when reconstructing split. "
            f"First missing id: {missing[0]}"
        )
    return [sequence_lookup[sequence_id] for sequence_id in sequence_ids]


# Text rendering for classical baselines

def render_baseline_sequence_text(
    sequence: NormalizedLogSequence,
    *,
    mode: str,
) -> str:
    parts: list[str] = []

    for event in sequence.events:
        if mode == "messages":
            if event.message:
                parts.append(event.message)
            continue

        event_parts: list[str] = []
        if event.component:
            event_parts.append(f"component={event.component}")
        if event.severity:
            event_parts.append(f"severity={event.severity}")
        if event.service:
            event_parts.append(f"service={event.service}")
        if event.message:
            event_parts.append(f"message={event.message}")

        if event_parts:
            parts.append(" ".join(event_parts))

    return "\n".join(parts)


# Text rendering + tokenization for BiLSTM

def render_bilstm_sequence_text(
    sequence: NormalizedLogSequence,
    *,
    event_token: str,
) -> str:
    event_texts = [event.message for event in sequence.events if event.message]
    return f" {event_token} ".join(event_texts)


def tokenize_with_pattern(text: str, token_pattern: re.Pattern[str]) -> list[str]:
    return token_pattern.findall(text.lower())


def encode_tokens(
    tokens: list[str],
    *,
    vocab: dict[str, int],
    max_tokens: int,
) -> list[int]:
    trimmed = tokens[:max_tokens]
    return [vocab.get(token, UNK_INDEX) for token in trimmed]


class SequenceDataset(Dataset[tuple[list[int], int]]):
    def __init__(self, sequences: list[list[int]], labels: list[int]) -> None:
        if len(sequences) != len(labels):
            raise ValueError("sequences and labels must have identical length.")
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        return self.sequences[index], self.labels[index]


def collate_batch(
    batch: list[tuple[list[int], int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences, labels = zip(*batch)

    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    max_len = int(lengths.max().item()) if len(lengths) > 0 else 1

    padded = torch.full(
        (len(sequences), max_len),
        fill_value=PAD_INDEX,
        dtype=torch.long,
    )

    for row, seq in enumerate(sequences):
        if seq:
            padded[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)

    label_tensor = torch.tensor(labels, dtype=torch.float32)
    return padded, lengths, label_tensor


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        pad_index: int,
    ) -> None:
        super().__init__()

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_index,
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        embedded = self.embedding(input_ids)

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
        )

        max_len = output.size(1)
        mask = (
            torch.arange(max_len, device=lengths.device)
            .unsqueeze(0)
            .expand(lengths.size(0), max_len)
            < lengths.unsqueeze(1)
        )
        mask = mask.unsqueeze(-1).float()

        summed = (output * mask).sum(dim=1)
        denom = lengths.unsqueeze(1).clamp(min=1).float()
        pooled = summed / denom

        logits = self.classifier(self.dropout(pooled)).squeeze(1)
        return logits


@torch.no_grad()
def predict_probabilities_bilstm(
    model: nn.Module,
    dataloader: DataLoader,
) -> np.ndarray:
    model.eval()
    all_probs: list[np.ndarray] = []

    for input_ids, lengths, _labels in dataloader:
        logits = model(input_ids, lengths)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)

    return np.concatenate(all_probs, axis=0) if all_probs else np.array([], dtype=float)


# Metrics and exports

@dataclass(slots=True)
class ExportedModelResult:
    model_key: str
    scope: str
    dataset: str
    model_type: str
    text_mode: str
    threshold: float
    default_metrics: dict[str, Any]
    tuned_metrics: dict[str, Any]


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float,
) -> tuple[dict[str, Any], np.ndarray]:
    y_pred = (y_score >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_score)

    try:
        roc_auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        roc_auc = float("nan")

    brier = brier_score_loss(y_true, y_score)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    metrics = {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "average_precision": float(pr_auc),
        "roc_auc": roc_auc,
        "brier_score": float(brier),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }
    return metrics, y_pred


def build_threshold_sweep_rows(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for threshold in np.linspace(0.0, 1.0, 501):
        y_pred = (y_score >= threshold).astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        rows.append(
            {
                "threshold": round(float(threshold), 6),
                "precision": round(float(precision), 6),
                "recall": round(float(recall), 6),
                "f1": round(float(f1), 6),
            }
        )

    return rows


def write_predictions_csv(
    path: Path,
    *,
    sequence_ids: list[str],
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
) -> None:
    rows: list[dict[str, Any]] = []

    for sequence_id, truth, score, pred in zip(sequence_ids, y_true, y_score, y_pred, strict=True):
        rows.append(
            {
                "sequence_id": sequence_id,
                "y_true": binary_to_label(int(truth)),
                "y_score": round(float(score), 8),
                "y_pred": binary_to_label(int(pred)),
                "threshold": round(float(threshold), 8),
            }
        )

    write_csv(
        path,
        rows,
        ["sequence_id", "y_true", "y_score", "y_pred", "threshold"],
    )


def write_model_metrics_csv(
    path: Path,
    *,
    scope: str,
    dataset: str,
    model_key: str,
    model_type: str,
    text_mode: str,
    default_metrics: dict[str, Any],
    tuned_metrics: dict[str, Any],
) -> None:
    row = {
        "scope": scope,
        "dataset": dataset,
        "model_key": model_key,
        "model_type": model_type,
        "text_mode": text_mode,
        "default_threshold": default_metrics["threshold"],
        "default_precision": default_metrics["precision"],
        "default_recall": default_metrics["recall"],
        "default_f1": default_metrics["f1"],
        "default_accuracy": default_metrics["accuracy"],
        "default_pr_auc": default_metrics["average_precision"],
        "default_roc_auc": default_metrics["roc_auc"],
        "default_brier_score": default_metrics["brier_score"],
        "default_tn": default_metrics["tn"],
        "default_fp": default_metrics["fp"],
        "default_fn": default_metrics["fn"],
        "default_tp": default_metrics["tp"],
        "tuned_threshold": tuned_metrics["threshold"],
        "tuned_precision": tuned_metrics["precision"],
        "tuned_recall": tuned_metrics["recall"],
        "tuned_f1": tuned_metrics["f1"],
        "tuned_accuracy": tuned_metrics["accuracy"],
        "tuned_pr_auc": tuned_metrics["average_precision"],
        "tuned_roc_auc": tuned_metrics["roc_auc"],
        "tuned_brier_score": tuned_metrics["brier_score"],
        "tuned_tn": tuned_metrics["tn"],
        "tuned_fp": tuned_metrics["fp"],
        "tuned_fn": tuned_metrics["fn"],
        "tuned_tp": tuned_metrics["tp"],
    }
    write_csv(path, [row], list(row.keys()))


# Plotting

def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_path: Path,
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_roc_curve_figure(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_path: Path,
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_calibration_curve_figure(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_path: Path,
) -> None:
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="uniform")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed anomalous frequency")
    ax.set_title("Calibration Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_threshold_curve_figure(
    sweep_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    thresholds = [float(row["threshold"]) for row in sweep_rows]
    precision_values = [float(row["precision"]) for row in sweep_rows]
    recall_values = [float(row["recall"]) for row in sweep_rows]
    f1_values = [float(row["f1"]) for row in sweep_rows]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(thresholds, precision_values, label="Precision")
    ax.plot(thresholds, recall_values, label="Recall")
    ax.plot(thresholds, f1_values, label="F1")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric value")
    ax.set_title("Threshold vs Precision / Recall / F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_confusion_matrix_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    *,
    title: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["normal", "anomalous"])
    disp.plot(ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_bar_metric(
    rows: list[dict[str, Any]],
    *,
    metric_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    model_names = [row["model_key"] for row in rows]
    values = [float(row[metric_key]) for row in rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(model_names, values)
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_default_vs_tuned(
    rows: list[dict[str, Any]],
    *,
    default_key: str,
    tuned_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    model_names = [row["model_key"] for row in rows]
    default_values = [float(row[default_key]) for row in rows]
    tuned_values = [float(row[tuned_key]) for row in rows]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, default_values, width, label="Default threshold")
    ax.bar(x + width / 2, tuned_values, width, label="Tuned threshold")
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_bilstm_history(
    history: list[dict[str, Any]],
    *,
    prefix: str,
    output_dir: Path,
) -> None:
    if not history:
        return

    epochs = [float(item["epoch"]) for item in history]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, [float(item["train_loss"]) for item in history], marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train loss")
    ax.set_title("BiLSTM Training Loss")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}__history_train_loss.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, [float(item["val_f1"]) for item in history], marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation F1")
    ax.set_title("BiLSTM Validation F1")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}__history_val_f1.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, [float(item["val_average_precision"]) for item in history], marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation average precision")
    ax.set_title("BiLSTM Validation Average Precision")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}__history_val_average_precision.png", dpi=300)
    plt.close(fig)


# Exporters

def export_classical_model(
    *,
    model_dir: Path,
    model_key: str,
    scope: str,
    sequence_lookup: dict[str, LabeledSequenceRecord],
    metrics_dir: Path,
    figures_dir: Path,
) -> ExportedModelResult:
    report_path = model_dir / "report.json"
    report = read_json(report_path)

    joblib_files = list(model_dir.glob("*.joblib"))
    if not joblib_files:
        raise FileNotFoundError(f"No .joblib artefact found in {model_dir}")
    artefact = joblib.load(joblib_files[0])

    test_ids = read_sequence_ids(model_dir / "test_sequence_ids.txt")
    test_records = select_records_by_ids(sequence_lookup, test_ids)

    text_mode = artefact.get("text_mode") or report.get("text_mode", "messages")
    vectorizer = artefact["vectorizer"]
    classifier = artefact["classifier"]
    threshold = float(
        artefact.get(
            "threshold",
            report.get("threshold_selection", {}).get("selected_threshold", 0.5),
        )
    )

    texts = [
        render_baseline_sequence_text(record.sequence, mode=text_mode)
        for record in test_records
    ]
    y_true = np.array([label_to_binary(record.label) for record in test_records], dtype=int)

    x_test = vectorizer.transform(texts)
    y_score = classifier.predict_proba(x_test)[:, 1]

    default_metrics, y_pred_default = compute_metrics(y_true, y_score, threshold=0.5)
    tuned_metrics, y_pred_tuned = compute_metrics(y_true, y_score, threshold=threshold)
    sweep_rows = build_threshold_sweep_rows(y_true, y_score)

    prefix = f"{scope}__hdfs__{model_key}"

    write_predictions_csv(
        metrics_dir / f"{prefix}__predictions.csv",
        sequence_ids=test_ids,
        y_true=y_true,
        y_score=y_score,
        y_pred=y_pred_tuned,
        threshold=threshold,
    )
    write_csv(
        metrics_dir / f"{prefix}__threshold_sweep.csv",
        sweep_rows,
        ["threshold", "precision", "recall", "f1"],
    )
    write_model_metrics_csv(
        metrics_dir / f"{prefix}__metrics.csv",
        scope=scope,
        dataset="HDFS",
        model_key=model_key,
        model_type=report.get("model_type", ""),
        text_mode=text_mode,
        default_metrics=default_metrics,
        tuned_metrics=tuned_metrics,
    )

    plot_precision_recall_curve(y_true, y_score, figures_dir / f"{prefix}__pr_curve.png")
    plot_roc_curve_figure(y_true, y_score, figures_dir / f"{prefix}__roc_curve.png")
    plot_calibration_curve_figure(y_true, y_score, figures_dir / f"{prefix}__calibration_curve.png")
    plot_threshold_curve_figure(sweep_rows, figures_dir / f"{prefix}__threshold_curve.png")
    plot_confusion_matrix_figure(
        y_true,
        y_pred_tuned,
        figures_dir / f"{prefix}__confusion_matrix_tuned.png",
        title=f"{model_key} - Confusion Matrix (Tuned Threshold)",
    )
    plot_confusion_matrix_figure(
        y_true,
        y_pred_default,
        figures_dir / f"{prefix}__confusion_matrix_default.png",
        title=f"{model_key} - Confusion Matrix (Default Threshold)",
    )

    return ExportedModelResult(
        model_key=model_key,
        scope=scope,
        dataset="HDFS",
        model_type=report.get("model_type", ""),
        text_mode=text_mode,
        threshold=threshold,
        default_metrics=default_metrics,
        tuned_metrics=tuned_metrics,
    )


def export_bilstm_model(
    *,
    model_dir: Path,
    model_key: str,
    scope: str,
    sequence_lookup: dict[str, LabeledSequenceRecord],
    metrics_dir: Path,
    figures_dir: Path,
) -> ExportedModelResult:
    report_path = model_dir / "report.json"
    report = read_json(report_path)

    pt_files = list(model_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt artefact found in {model_dir}")

    artefact = torch.load(pt_files[0], map_location="cpu", weights_only=False)

    test_ids = read_sequence_ids(model_dir / "test_sequence_ids.txt")
    test_records = select_records_by_ids(sequence_lookup, test_ids)

    vocab: dict[str, int] = artefact["vocab"]
    token_pattern = re.compile(artefact["token_pattern"])
    max_tokens = int(artefact["max_tokens"])
    event_token = artefact.get("event_token", EVENT_TOKEN)
    threshold = float(
        artefact.get(
            "threshold",
            report.get("best_checkpoint", {}).get("validation_threshold", 0.5),
        )
    )

    texts = [
        render_bilstm_sequence_text(record.sequence, event_token=event_token)
        for record in test_records
    ]
    labels = [label_to_binary(record.label) for record in test_records]

    encoded_sequences = [
        encode_tokens(
            tokenize_with_pattern(text, token_pattern),
            vocab=vocab,
            max_tokens=max_tokens,
        )
        for text in texts
    ]

    dataset = SequenceDataset(encoded_sequences, labels)
    batch_size = int(report.get("training_config", {}).get("batch_size", 64))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    model_config = artefact["model_config"]
    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=int(model_config["embedding_dim"]),
        hidden_size=int(model_config["hidden_size"]),
        num_layers=int(model_config["num_layers"]),
        dropout=float(model_config["dropout"]),
        pad_index=int(artefact.get("pad_index", PAD_INDEX)),
    )
    model.load_state_dict(artefact["state_dict"])
    model.eval()

    y_true = np.array(labels, dtype=int)
    y_score = predict_probabilities_bilstm(model, dataloader)

    default_metrics, y_pred_default = compute_metrics(y_true, y_score, threshold=0.5)
    tuned_metrics, y_pred_tuned = compute_metrics(y_true, y_score, threshold=threshold)
    sweep_rows = build_threshold_sweep_rows(y_true, y_score)

    prefix = f"{scope}__hdfs__{model_key}"

    write_predictions_csv(
        metrics_dir / f"{prefix}__predictions.csv",
        sequence_ids=test_ids,
        y_true=y_true,
        y_score=y_score,
        y_pred=y_pred_tuned,
        threshold=threshold,
    )
    write_csv(
        metrics_dir / f"{prefix}__threshold_sweep.csv",
        sweep_rows,
        ["threshold", "precision", "recall", "f1"],
    )
    write_model_metrics_csv(
        metrics_dir / f"{prefix}__metrics.csv",
        scope=scope,
        dataset="HDFS",
        model_key=model_key,
        model_type=report.get("model_type", ""),
        text_mode="messages",
        default_metrics=default_metrics,
        tuned_metrics=tuned_metrics,
    )

    plot_precision_recall_curve(y_true, y_score, figures_dir / f"{prefix}__pr_curve.png")
    plot_roc_curve_figure(y_true, y_score, figures_dir / f"{prefix}__roc_curve.png")
    plot_calibration_curve_figure(y_true, y_score, figures_dir / f"{prefix}__calibration_curve.png")
    plot_threshold_curve_figure(sweep_rows, figures_dir / f"{prefix}__threshold_curve.png")
    plot_confusion_matrix_figure(
        y_true,
        y_pred_tuned,
        figures_dir / f"{prefix}__confusion_matrix_tuned.png",
        title=f"{model_key} - Confusion Matrix (Tuned Threshold)",
    )
    plot_confusion_matrix_figure(
        y_true,
        y_pred_default,
        figures_dir / f"{prefix}__confusion_matrix_default.png",
        title=f"{model_key} - Confusion Matrix (Default Threshold)",
    )

    history = report.get("history", [])
    if history:
        plot_bilstm_history(history, prefix=prefix, output_dir=figures_dir)

    return ExportedModelResult(
        model_key=model_key,
        scope=scope,
        dataset="HDFS",
        model_type=report.get("model_type", ""),
        text_mode="messages",
        threshold=threshold,
        default_metrics=default_metrics,
        tuned_metrics=tuned_metrics,
    )


# Comparison exports

def export_scope_comparison(
    *,
    scope: str,
    results: list[ExportedModelResult],
    metrics_dir: Path,
    figures_dir: Path,
) -> None:
    rows: list[dict[str, Any]] = []

    for result in results:
        rows.append(
            {
                "scope": scope,
                "dataset": result.dataset,
                "model_key": result.model_key,
                "model_type": result.model_type,
                "text_mode": result.text_mode,
                "threshold": result.threshold,
                "precision": result.tuned_metrics["precision"],
                "recall": result.tuned_metrics["recall"],
                "f1": result.tuned_metrics["f1"],
                "accuracy": result.tuned_metrics["accuracy"],
                "pr_auc": result.tuned_metrics["average_precision"],
                "roc_auc": result.tuned_metrics["roc_auc"],
                "brier_score": result.tuned_metrics["brier_score"],
                "tn": result.tuned_metrics["tn"],
                "fp": result.tuned_metrics["fp"],
                "fn": result.tuned_metrics["fn"],
                "tp": result.tuned_metrics["tp"],
                "default_precision": result.default_metrics["precision"],
                "default_recall": result.default_metrics["recall"],
                "default_f1": result.default_metrics["f1"],
                "tuned_precision": result.tuned_metrics["precision"],
                "tuned_recall": result.tuned_metrics["recall"],
                "tuned_f1": result.tuned_metrics["f1"],
            }
        )

    write_csv(
        metrics_dir / f"{scope}__hdfs__model_comparison.csv",
        rows,
        list(rows[0].keys()),
    )

    plot_bar_metric(
        rows,
        metric_key="f1",
        title="Model Comparison: F1",
        ylabel="F1",
        output_path=figures_dir / f"{scope}__hdfs__model_comparison__bar_f1.png",
    )
    plot_bar_metric(
        rows,
        metric_key="precision",
        title="Model Comparison: Precision",
        ylabel="Precision",
        output_path=figures_dir / f"{scope}__hdfs__model_comparison__bar_precision.png",
    )
    plot_bar_metric(
        rows,
        metric_key="recall",
        title="Model Comparison: Recall",
        ylabel="Recall",
        output_path=figures_dir / f"{scope}__hdfs__model_comparison__bar_recall.png",
    )
    plot_bar_metric(
        rows,
        metric_key="pr_auc",
        title="Model Comparison: PR-AUC",
        ylabel="PR-AUC",
        output_path=figures_dir / f"{scope}__hdfs__model_comparison__bar_pr_auc.png",
    )
    plot_bar_metric(
        rows,
        metric_key="roc_auc",
        title="Model Comparison: ROC-AUC",
        ylabel="ROC-AUC",
        output_path=figures_dir / f"{scope}__hdfs__model_comparison__bar_roc_auc.png",
    )
    plot_bar_metric(
        rows,
        metric_key="brier_score",
        title="Model Comparison: Brier Score",
        ylabel="Brier Score",
        output_path=figures_dir / f"{scope}__hdfs__model_comparison__bar_brier_score.png",
    )

    plot_default_vs_tuned(
        rows,
        default_key="default_f1",
        tuned_key="tuned_f1",
        title="Default vs Tuned Threshold: F1",
        ylabel="F1",
        output_path=figures_dir / f"{scope}__hdfs__threshold_effect__f1.png",
    )
    plot_default_vs_tuned(
        rows,
        default_key="default_precision",
        tuned_key="tuned_precision",
        title="Default vs Tuned Threshold: Precision",
        ylabel="Precision",
        output_path=figures_dir / f"{scope}__hdfs__threshold_effect__precision.png",
    )
    plot_default_vs_tuned(
        rows,
        default_key="default_recall",
        tuned_key="tuned_recall",
        title="Default vs Tuned Threshold: Recall",
        ylabel="Recall",
        output_path=figures_dir / f"{scope}__hdfs__threshold_effect__recall.png",
    )


# Main

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate all result artefacts from existing trained HDFS model artefacts."
    )
    parser.add_argument("--hdfs-log", type=Path, required=True)
    parser.add_argument("--hdfs-labels", type=Path, required=True)

    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--baseline-calibrated-dir", type=Path, required=True)
    parser.add_argument("--baseline-enriched-dir", type=Path, required=True)
    parser.add_argument("--baseline-50k-dir", type=Path, required=True)
    parser.add_argument("--bilstm-50k-dir", type=Path, required=True)

    parser.add_argument("--output-root", type=Path, default=Path("outputs"))

    args = parser.parse_args()

    for path in [
        args.hdfs_log,
        args.hdfs_labels,
        args.baseline_dir,
        args.baseline_calibrated_dir,
        args.baseline_enriched_dir,
        args.baseline_50k_dir,
        args.bilstm_50k_dir,
    ]:
        if not path.exists():
            raise SystemExit(f"Required path not found: {path}")

    metrics_dir = args.output_root / "metrics"
    figures_dir = args.output_root / "figures"

    ensure_dir(metrics_dir)
    ensure_dir(figures_dir)

    print("=" * 80)
    print("LOADING HDFS SEQUENCES ONCE")
    print("=" * 80)
    records = build_hdfs_sequences_from_files(
        log_path=args.hdfs_log,
        label_csv_path=args.hdfs_labels,
    )
    sequence_lookup = build_sequence_lookup(records)
    print(f"Loaded sequences: {len(records)}")

    print("\n" + "=" * 80)
    print("EXPORTING FULL CLASSICAL RESULTS")
    print("=" * 80)
    classical_results = [
        export_classical_model(
            model_dir=args.baseline_dir,
            model_key="baseline",
            scope="classical_full",
            sequence_lookup=sequence_lookup,
            metrics_dir=metrics_dir,
            figures_dir=figures_dir,
        ),
        export_classical_model(
            model_dir=args.baseline_calibrated_dir,
            model_key="baseline_calibrated",
            scope="classical_full",
            sequence_lookup=sequence_lookup,
            metrics_dir=metrics_dir,
            figures_dir=figures_dir,
        ),
        export_classical_model(
            model_dir=args.baseline_enriched_dir,
            model_key="baseline_enriched",
            scope="classical_full",
            sequence_lookup=sequence_lookup,
            metrics_dir=metrics_dir,
            figures_dir=figures_dir,
        ),
    ]
    export_scope_comparison(
        scope="classical_full",
        results=classical_results,
        metrics_dir=metrics_dir,
        figures_dir=figures_dir,
    )

    print("\n" + "=" * 80)
    print("EXPORTING 50K SMOKE-TEST RESULTS")
    print("=" * 80)
    smoke_results = [
        export_classical_model(
            model_dir=args.baseline_50k_dir,
            model_key="baseline_50k",
            scope="smoketest_50k",
            sequence_lookup=sequence_lookup,
            metrics_dir=metrics_dir,
            figures_dir=figures_dir,
        ),
        export_bilstm_model(
            model_dir=args.bilstm_50k_dir,
            model_key="bilstm_50k",
            scope="smoketest_50k",
            sequence_lookup=sequence_lookup,
            metrics_dir=metrics_dir,
            figures_dir=figures_dir,
        ),
    ]
    export_scope_comparison(
        scope="smoketest_50k",
        results=smoke_results,
        metrics_dir=metrics_dir,
        figures_dir=figures_dir,
    )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Metrics written to: {metrics_dir.resolve()}")
    print(f"Figures written to: {figures_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
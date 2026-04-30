from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, TypedDict

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from app.services.preprocessing import (
    LabeledSequenceRecord,
    NormalizedLogSequence,
    build_hdfs_sequences_from_files,
)


class ConfusionMatrixDict(TypedDict):
    tn: int
    fp: int
    fn: int
    tp: int


class MetricsDict(TypedDict):
    threshold: float
    precision: float
    recall: float
    f1: float
    average_precision: float
    roc_auc: float | None
    confusion_matrix: ConfusionMatrixDict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a supervised HDFS baseline with TF-IDF + Logistic Regression."
    )
    parser.add_argument("--hdfs-log", required=True, help="Path to HDFS.log")
    parser.add_argument(
        "--hdfs-labels",
        required=True,
        help="Path to anomaly_label.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="artefacts/models/hdfs_baseline",
        help="Directory where model artefacts and reports are stored.",
    )
    parser.add_argument(
        "--text-mode",
        choices=["messages", "enriched"],
        default="messages",
        help=(
            "'messages' uses only message text. "
            "'enriched' adds component, severity, and service tokens."
        ),
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fraction of the dataset reserved for test split.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.10,
        help="Fraction of the dataset reserved for validation split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible splits and model training.",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=2,
        help="Minimum document frequency for TF-IDF vocabulary.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=50000,
        help="Maximum TF-IDF vocabulary size.",
    )
    parser.add_argument(
        "--ngram-min",
        type=int,
        default=1,
        help="Lower bound of ngram range.",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=2,
        help="Upper bound of ngram range.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum number of LogisticRegression iterations.",
    )
    parser.add_argument(
        "--solver",
        choices=["lbfgs", "liblinear", "saga"],
        default="lbfgs",
        help="Solver for LogisticRegression.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help=(
            "Optional maximum number of sequences to use. "
            "A stratified subset is sampled before the train/validation/test split."
        ),
    )
    return parser.parse_args()


def render_sequence_text(
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


def label_to_binary(label: str) -> int:
    return 1 if label == "anomalous" else 0


def subset_records_stratified(
    records: list[LabeledSequenceRecord],
    *,
    max_records: int | None,
    random_state: int,
) -> list[LabeledSequenceRecord]:
    """
    Select an optional stratified subset before vectorization and splitting.
    """
    if max_records is None or max_records >= len(records):
        return records

    if max_records < 2:
        raise ValueError("max-records must be at least 2 when provided.")

    labels = np.array(
        [label_to_binary(record.label) for record in records],
        dtype=int,
    )
    all_indices = np.arange(len(records))

    subset_idx_raw, _ = train_test_split(
        all_indices,
        train_size=max_records,
        random_state=random_state,
        stratify=labels,
    )

    subset_idx: list[int] = [int(i) for i in subset_idx_raw]
    return [records[i] for i in subset_idx]


def choose_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    if thresholds.size == 0:
        return 0.5

    f1_scores = (
        2.0 * precision[:-1] * recall[:-1]
        / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    )

    best_idx = int(np.nanargmax(f1_scores))
    return float(thresholds[best_idx])


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float,
) -> MetricsDict:
    y_pred = (y_prob >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    roc_auc: float | None
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = None

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "roc_auc": roc_auc,
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
    }


def save_split_manifest(
    output_dir: Path,
    split_name: str,
    sequence_ids: list[str],
) -> None:
    path = output_dir / f"{split_name}_sequence_ids.txt"
    path.write_text("\n".join(sequence_ids) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    if args.test_size <= 0 or args.val_size <= 0:
        raise ValueError("test_size and val_size must both be > 0.")

    if args.test_size + args.val_size >= 1.0:
        raise ValueError("test_size + val_size must be < 1.0.")

    if args.ngram_min > args.ngram_max:
        raise ValueError("ngram-min must be <= ngram-max.")

    if args.max_records is not None and args.max_records < 2:
        raise ValueError("max-records must be at least 2 when provided.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading HDFS sequences...")
    records: list[LabeledSequenceRecord] = build_hdfs_sequences_from_files(
        log_path=args.hdfs_log,
        label_csv_path=args.hdfs_labels,
    )

    if not records:
        raise RuntimeError("No HDFS sequences were loaded.")

    full_count = len(records)
    full_labels = np.array(
        [label_to_binary(record.label) for record in records],
        dtype=int,
    )
    full_anomaly_count = int(full_labels.sum())
    full_normal_count = int(full_count - full_anomaly_count)

    print(f"Loaded sequences from disk: {full_count}")
    print(f"Normal: {full_normal_count}")
    print(f"Anomalous: {full_anomaly_count}")

    records = subset_records_stratified(
        records,
        max_records=args.max_records,
        random_state=args.random_state,
    )

    subset_applied = len(records) != full_count
    if subset_applied:
        subset_labels_preview = np.array(
            [label_to_binary(record.label) for record in records],
            dtype=int,
        )
        subset_anomaly_count = int(subset_labels_preview.sum())
        subset_normal_count = int(len(records) - subset_anomaly_count)

        print()
        print("Using stratified subset for training/evaluation:")
        print(f"Subset sequences: {len(records)}")
        print(f"Subset normal:    {subset_normal_count}")
        print(f"Subset anomalous: {subset_anomaly_count}")

    sequence_ids: list[str] = [record.sequence.sequence_id for record in records]
    texts: list[str] = [
        render_sequence_text(record.sequence, mode=args.text_mode)
        for record in records
    ]
    labels = np.array(
        [label_to_binary(record.label) for record in records],
        dtype=int,
    )

    total_count = len(records)
    anomaly_count = int(labels.sum())
    normal_count = int(total_count - anomaly_count)

    print()
    print(f"Sequences used for this run: {total_count}")
    print(f"Normal: {normal_count}")
    print(f"Anomalous: {anomaly_count}")

    all_indices = np.arange(total_count)

    train_val_idx, test_idx = train_test_split(
        all_indices,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=labels,
    )

    val_relative_size = args.val_size / (1.0 - args.test_size)

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_relative_size,
        random_state=args.random_state,
        stratify=labels[train_val_idx],
    )

    train_idx = [int(i) for i in train_idx]
    val_idx = [int(i) for i in val_idx]
    test_idx = [int(i) for i in test_idx]

    train_texts: list[str] = [texts[i] for i in train_idx]
    val_texts: list[str] = [texts[i] for i in val_idx]
    test_texts: list[str] = [texts[i] for i in test_idx]

    y_train = labels[train_idx]
    y_val = labels[val_idx]
    y_test = labels[test_idx]

    save_split_manifest(output_dir, "train", [sequence_ids[i] for i in train_idx])
    save_split_manifest(output_dir, "validation", [sequence_ids[i] for i in val_idx])
    save_split_manifest(output_dir, "test", [sequence_ids[i] for i in test_idx])

    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(args.ngram_min, args.ngram_max),
        min_df=args.min_df,
        max_features=args.max_features,
        sublinear_tf=True,
    )

    x_train = vectorizer.fit_transform(train_texts)
    x_val = vectorizer.transform(val_texts)
    x_test = vectorizer.transform(test_texts)

    print("Training Logistic Regression...")
    classifier = LogisticRegression(
        max_iter=args.max_iter,
        class_weight="balanced",
        random_state=args.random_state,
        solver=args.solver,
    )
    classifier.fit(x_train, y_train)

    val_prob = classifier.predict_proba(x_val)[:, 1]
    test_prob = classifier.predict_proba(x_test)[:, 1]

    tuned_threshold = choose_threshold_by_f1(y_val, val_prob)

    validation_default = compute_metrics(y_val, val_prob, threshold=0.5)
    validation_tuned = compute_metrics(y_val, val_prob, threshold=tuned_threshold)
    test_default = compute_metrics(y_test, test_prob, threshold=0.5)
    test_tuned = compute_metrics(y_test, test_prob, threshold=tuned_threshold)

    artefact: dict[str, Any] = {
        "model_type": "tfidf_logistic_regression",
        "positive_label": "anomalous",
        "negative_label": "normal",
        "text_mode": args.text_mode,
        "threshold": tuned_threshold,
        "vectorizer": vectorizer,
        "classifier": classifier,
        "feature_config": {
            "ngram_range": [args.ngram_min, args.ngram_max],
            "min_df": args.min_df,
            "max_features": args.max_features,
            "sublinear_tf": True,
        },
    }

    model_path = output_dir / "hdfs_baseline.joblib"
    joblib.dump(artefact, model_path)

    report: dict[str, Any] = {
        "dataset": "HDFS",
        "model_type": "TF-IDF + Logistic Regression",
        "text_mode": args.text_mode,
        "paths": {
            "hdfs_log": str(args.hdfs_log),
            "hdfs_labels": str(args.hdfs_labels),
            "model_artifact": str(model_path),
        },
        "full_dataset_summary": {
            "total_sequences": full_count,
            "normal_sequences": full_normal_count,
            "anomalous_sequences": full_anomaly_count,
        },
        "dataset_summary": {
            "total_sequences": total_count,
            "normal_sequences": normal_count,
            "anomalous_sequences": anomaly_count,
            "subset_applied": subset_applied,
        },
        "split_summary": {
            "train_size": int(len(train_idx)),
            "validation_size": int(len(val_idx)),
            "test_size": int(len(test_idx)),
            "train_anomalous": int(y_train.sum()),
            "validation_anomalous": int(y_val.sum()),
            "test_anomalous": int(y_test.sum()),
        },
        "training_config": {
            "random_state": args.random_state,
            "test_size": args.test_size,
            "val_size": args.val_size,
            "solver": args.solver,
            "max_iter": args.max_iter,
            "class_weight": "balanced",
            "ngram_range": [args.ngram_min, args.ngram_max],
            "min_df": args.min_df,
            "max_features": args.max_features,
            "max_records": args.max_records,
        },
        "threshold_selection": {
            "strategy": "maximize_f1_on_validation",
            "selected_threshold": tuned_threshold,
        },
        "metrics": {
            "validation_default_threshold_0_5": validation_default,
            "validation_tuned_threshold": validation_tuned,
            "test_default_threshold_0_5": test_default,
            "test_tuned_threshold": test_tuned,
        },
    }

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print("Training finished.")
    print(f"Saved model artifact: {model_path}")
    print(f"Saved report: {report_path}")
    print()
    print("Selected threshold from validation:")
    print(f"  {tuned_threshold:.6f}")
    print()
    print("Test metrics at tuned threshold:")
    print(f"  Precision: {test_tuned['precision']:.4f}")
    print(f"  Recall:    {test_tuned['recall']:.4f}")
    print(f"  F1:        {test_tuned['f1']:.4f}")
    print(f"  PR-AUC:    {test_tuned['average_precision']:.4f}")
    if test_tuned["roc_auc"] is not None:
        print(f"  ROC-AUC:   {test_tuned['roc_auc']:.4f}")
    else:
        print("  ROC-AUC:   n/a")
    print("  Confusion matrix:")
    print(f"    TN={test_tuned['confusion_matrix']['tn']}")
    print(f"    FP={test_tuned['confusion_matrix']['fp']}")
    print(f"    FN={test_tuned['confusion_matrix']['fn']}")
    print(f"    TP={test_tuned['confusion_matrix']['tp']}")


if __name__ == "__main__":
    main()
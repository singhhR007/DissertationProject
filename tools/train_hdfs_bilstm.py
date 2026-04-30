from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

from app.services.preprocessing import (
    LabeledSequenceRecord,
    NormalizedLogSequence,
    build_hdfs_sequences_from_files,
)


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
EVENT_TOKEN = "<E>"
PAD_INDEX = 0
UNK_INDEX = 1
EVENT_INDEX = 2


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
        description="Train a supervised HDFS BiLSTM baseline for comparison."
    )
    parser.add_argument("--hdfs-log", required=True, help="Path to HDFS.log")
    parser.add_argument(
        "--hdfs-labels",
        required=True,
        help="Path to anomaly_label.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="artefacts/models/hdfs_bilstm",
        help="Directory where model artefacts and reports are stored.",
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
        help="Random seed for reproducible splits and training.",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=2,
        help="Minimum token frequency for inclusion in the vocabulary.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens per sequence after truncation.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
        help="Embedding dimension.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
        help="LSTM hidden size.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=1,
        help="Number of LSTM layers.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.30,
        help="Dropout applied after pooled sequence representation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early stopping patience measured in epochs without validation F1 improvement.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Adam weight decay.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Training device selection.",
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
    parser.add_argument(
        "--log-every",
        type=int,
        default=200,
        help=(
            "Print training progress every N batches. "
            "Use 0 to disable batch-level progress logging."
        ),
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def label_to_binary(label: str) -> int:
    return 1 if label == "anomalous" else 0


def subset_records_stratified(
    records: list[LabeledSequenceRecord],
    *,
    max_records: int | None,
    random_state: int,
) -> list[LabeledSequenceRecord]:
    """
    Select an optional stratified subset before tokenization and splitting.

    This keeps the anomaly ratio approximately stable while making local
    comparison runs feasible on limited hardware.
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


def render_sequence_text(sequence: NormalizedLogSequence) -> str:
    event_texts = [event.message for event in sequence.events if event.message]
    return f" {EVENT_TOKEN} ".join(event_texts)


_TOKEN_RE = re.compile(rf"{re.escape(EVENT_TOKEN)}|[A-Za-z0-9_./:\-$]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def build_vocabulary(
    tokenized_texts: list[list[str]],
    *,
    min_freq: int,
) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)

    vocab: dict[str, int] = {
        PAD_TOKEN: PAD_INDEX,
        UNK_TOKEN: UNK_INDEX,
        EVENT_TOKEN: EVENT_INDEX,
    }

    next_index = len(vocab)
    for token, freq in counter.items():
        if token == EVENT_TOKEN:
            continue
        if freq >= min_freq and token not in vocab:
            vocab[token] = next_index
            next_index += 1

    return vocab


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


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    all_probs: list[np.ndarray] = []

    for input_ids, lengths, _labels in dataloader:
        input_ids = input_ids.to(device)
        lengths = lengths.to(device)

        logits = model(input_ids, lengths)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)

    return np.concatenate(all_probs, axis=0) if all_probs else np.array([], dtype=float)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    log_every: int = 0,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    total_batches = len(dataloader)

    for batch_idx, (input_ids, lengths, labels) in enumerate(dataloader, start=1):
        input_ids = input_ids.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids, lengths)
        loss = loss_fn(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        batch_size = int(labels.size(0))
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

        if log_every > 0 and (batch_idx % log_every == 0 or batch_idx == total_batches):
            avg_loss = total_loss / max(total_examples, 1)
            print(
                f"    batch {batch_idx}/{total_batches} | "
                f"avg_train_loss={avg_loss:.4f}"
            )

    if total_examples == 0:
        return 0.0
    return total_loss / total_examples


def main() -> None:
    args = parse_args()
    set_seed(args.random_state)

    if args.test_size <= 0 or args.val_size <= 0:
        raise ValueError("test_size and val_size must both be > 0.")
    if args.test_size + args.val_size >= 1.0:
        raise ValueError("test_size + val_size must be < 1.0.")
    if args.min_freq < 1:
        raise ValueError("min-freq must be >= 1.")
    if args.max_tokens < 8:
        raise ValueError("max-tokens must be >= 8.")
    if args.max_records is not None and args.max_records < 2:
        raise ValueError("max-records must be at least 2.")
    if args.log_every < 0:
        raise ValueError("log-every must be >= 0.")

    device = select_device(args.device)
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
    texts: list[str] = [render_sequence_text(record.sequence) for record in records]
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
    print(f"Device: {device.type}")

    all_indices = np.arange(total_count)

    train_val_idx_raw, test_idx_raw = train_test_split(
        all_indices,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=labels,
    )

    val_relative_size = args.val_size / (1.0 - args.test_size)

    train_idx_raw, val_idx_raw = train_test_split(
        train_val_idx_raw,
        test_size=val_relative_size,
        random_state=args.random_state,
        stratify=labels[train_val_idx_raw],
    )

    train_idx: list[int] = [int(i) for i in train_idx_raw]
    val_idx: list[int] = [int(i) for i in val_idx_raw]
    test_idx: list[int] = [int(i) for i in test_idx_raw]

    train_ids: list[str] = [sequence_ids[i] for i in train_idx]
    val_ids: list[str] = [sequence_ids[i] for i in val_idx]
    test_ids: list[str] = [sequence_ids[i] for i in test_idx]

    save_split_manifest(output_dir, "train", train_ids)
    save_split_manifest(output_dir, "validation", val_ids)
    save_split_manifest(output_dir, "test", test_ids)

    train_texts = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    test_texts = [texts[i] for i in test_idx]

    y_train = np.array([labels[i] for i in train_idx], dtype=int)
    y_val = np.array([labels[i] for i in val_idx], dtype=int)
    y_test = np.array([labels[i] for i in test_idx], dtype=int)

    print()
    print("Tokenizing training split...")
    train_tokenized = [tokenize(text) for text in train_texts]
    print("Tokenizing validation split...")
    val_tokenized = [tokenize(text) for text in val_texts]
    print("Tokenizing test split...")
    test_tokenized = [tokenize(text) for text in test_texts]

    print("Building vocabulary...")
    vocab = build_vocabulary(
        train_tokenized,
        min_freq=args.min_freq,
    )

    print("Encoding training split...")
    x_train = [
        encode_tokens(tokens, vocab=vocab, max_tokens=args.max_tokens)
        for tokens in train_tokenized
    ]
    print("Encoding validation split...")
    x_val = [
        encode_tokens(tokens, vocab=vocab, max_tokens=args.max_tokens)
        for tokens in val_tokenized
    ]
    print("Encoding test split...")
    x_test = [
        encode_tokens(tokens, vocab=vocab, max_tokens=args.max_tokens)
        for tokens in test_tokenized
    ]

    train_dataset = SequenceDataset(x_train, y_train.tolist())
    val_dataset = SequenceDataset(x_val, y_val.tolist())
    test_dataset = SequenceDataset(x_test, y_test.tolist())

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pad_index=PAD_INDEX,
    ).to(device)

    pos_count = int(y_train.sum())
    neg_count = int(len(y_train) - pos_count)
    if pos_count == 0:
        raise RuntimeError("Training split contains no anomalous sequences.")
    pos_weight_value = neg_count / pos_count

    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_state_dict: dict[str, torch.Tensor] | None = None
    best_val_f1 = -1.0
    best_threshold = 0.5
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    print()
    print("Training BiLSTM...")
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch:02d} started...")
        train_loss = train_one_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            loss_fn=loss_fn,
            log_every=args.log_every,
        )

        val_prob = predict_probabilities(model, val_loader, device=device)
        val_threshold = choose_threshold_by_f1(y_val, val_prob)
        val_metrics = compute_metrics(y_val, val_prob, threshold=val_threshold)

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "val_average_precision": val_metrics["average_precision"],
                "val_threshold": val_threshold,
            }
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_pr_auc={val_metrics['average_precision']:.4f} | "
            f"val_threshold={val_threshold:.6f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_threshold = val_threshold
            best_epoch = epoch
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print("Early stopping triggered.")
            break

    if best_state_dict is None:
        raise RuntimeError("No model checkpoint was captured during training.")

    model.load_state_dict(best_state_dict)

    val_prob = predict_probabilities(model, val_loader, device=device)
    test_prob = predict_probabilities(model, test_loader, device=device)

    validation_default = compute_metrics(y_val, val_prob, threshold=0.5)
    validation_tuned = compute_metrics(y_val, val_prob, threshold=best_threshold)
    test_default = compute_metrics(y_test, test_prob, threshold=0.5)
    test_tuned = compute_metrics(y_test, test_prob, threshold=best_threshold)

    model_artefact = {
        "model_type": "bilstm_sequence_classifier",
        "positive_label": "anomalous",
        "negative_label": "normal",
        "score_type": "anomalous_class_probability",
        "threshold": float(best_threshold),
        "vocab": vocab,
        "pad_index": PAD_INDEX,
        "unk_index": UNK_INDEX,
        "event_token": EVENT_TOKEN,
        "max_tokens": args.max_tokens,
        "token_pattern": _TOKEN_RE.pattern,
        "state_dict": best_state_dict,
        "model_config": {
            "embedding_dim": args.embedding_dim,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "bidirectional": True,
        },
    }

    model_artefact_path = output_dir / "hdfs_bilstm.pt"
    torch.save(model_artefact, model_artefact_path)

    report = {
        "dataset": "HDFS",
        "model_type": "BiLSTM sequence classifier",
        "paths": {
            "hdfs_log": str(args.hdfs_log),
            "hdfs_labels": str(args.hdfs_labels),
            "model_artefact": str(model_artefact_path),
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
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "embedding_dim": args.embedding_dim,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "min_freq": args.min_freq,
            "max_tokens": args.max_tokens,
            "max_records": args.max_records,
            "log_every": args.log_every,
            "pos_weight": pos_weight_value,
            "device": device.type,
        },
        "vocabulary": {
            "size": len(vocab),
        },
        "best_checkpoint": {
            "epoch": best_epoch,
            "validation_threshold": float(best_threshold),
        },
        "metrics": {
            "validation_default_threshold_0_5": validation_default,
            "validation_tuned_threshold": validation_tuned,
            "test_default_threshold_0_5": test_default,
            "test_tuned_threshold": test_tuned,
        },
        "history": history,
    }

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print("Training finished.")
    print(f"Saved model artefact: {model_artefact_path}")
    print(f"Saved report: {report_path}")
    print()
    print("Best validation checkpoint:")
    print(f"  Epoch:      {best_epoch}")
    print(f"  Threshold:  {best_threshold:.6f}")
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
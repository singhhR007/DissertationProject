from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

from app.services.preprocessing import (
    LabeledSequenceRecord,
    build_hdfs_sequences_from_files,
    build_openstack_sequences_from_files,
)


# Small helpers

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_mean(values: list[int]) -> float:
    return statistics.mean(values) if values else 0.0


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# Summary builders

def _build_summary_row(dataset_name: str, records: list[LabeledSequenceRecord]) -> dict:
    event_lengths = [len(record.sequence.events) for record in records]
    normal_count = sum(1 for record in records if record.label == "normal")
    anomalous_count = sum(1 for record in records if record.label == "anomalous")

    return {
        "dataset": dataset_name,
        "total_sequences": len(records),
        "normal_sequences": normal_count,
        "anomalous_sequences": anomalous_count,
        "min_events_per_sequence": min(event_lengths) if event_lengths else 0,
        "avg_events_per_sequence": round(_safe_mean(event_lengths), 4),
        "max_events_per_sequence": max(event_lengths) if event_lengths else 0,
    }


def _build_manifest_rows(dataset_name: str, records: list[LabeledSequenceRecord]) -> list[dict]:
    rows: list[dict] = []

    for record in records:
        rows.append(
            {
                "dataset": dataset_name,
                "sequence_id": record.sequence.sequence_id,
                "label": record.label,
                "event_count": len(record.sequence.events),
                "source": record.sequence.source,
            }
        )

    return rows


# Export functions

def export_hdfs_statistics(
    *,
    hdfs_log_path: Path,
    hdfs_label_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    records = build_hdfs_sequences_from_files(
        log_path=hdfs_log_path,
        label_csv_path=hdfs_label_path,
    )

    summary_path = output_dir / "hdfs__sequence_summary.csv"
    manifest_path = output_dir / "hdfs__sequence_manifest.csv"

    summary_row = _build_summary_row("HDFS", records)
    manifest_rows = _build_manifest_rows("HDFS", records)

    _write_csv(
        summary_path,
        rows=[summary_row],
        fieldnames=list(summary_row.keys()),
    )
    _write_csv(
        manifest_path,
        rows=manifest_rows,
        fieldnames=["dataset", "sequence_id", "label", "event_count", "source"],
    )

    return summary_path, manifest_path


def export_openstack_statistics(
    *,
    openstack_log_paths: list[Path],
    openstack_label_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    records = build_openstack_sequences_from_files(
        log_paths=openstack_log_paths,
        anomaly_label_path=openstack_label_path,
    )

    summary_path = output_dir / "openstack__sequence_summary.csv"
    manifest_path = output_dir / "openstack__sequence_manifest.csv"

    summary_row = _build_summary_row("OpenStack", records)
    manifest_rows = _build_manifest_rows("OpenStack", records)

    _write_csv(
        summary_path,
        rows=[summary_row],
        fieldnames=list(summary_row.keys()),
    )
    _write_csv(
        manifest_path,
        rows=manifest_rows,
        fieldnames=["dataset", "sequence_id", "label", "event_count", "source"],
    )

    return summary_path, manifest_path


# Main

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export reproducible sequence statistics for HDFS and OpenStack."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/datasets"),
        help="Directory where summary/manifests should be written.",
    )
    parser.add_argument(
        "--mode",
        choices=["hdfs", "openstack", "both"],
        default="both",
        help="Which dataset(s) to process.",
    )

    parser.add_argument("--hdfs-log", type=Path, help="Path to HDFS.log")
    parser.add_argument("--hdfs-labels", type=Path, help="Path to HDFS anomaly_label.csv")

    parser.add_argument(
        "--openstack-log",
        type=Path,
        nargs="*",
        help="Paths to OpenStack log files.",
    )
    parser.add_argument(
        "--openstack-labels",
        type=Path,
        help="Path to OpenStack anomaly_labels.txt",
    )

    args = parser.parse_args()
    _ensure_dir(args.output_dir)

    if args.mode in {"hdfs", "both"}:
        if args.hdfs_log is None or args.hdfs_labels is None:
            raise SystemExit("For HDFS export you must provide --hdfs-log and --hdfs-labels.")
        if not args.hdfs_log.exists():
            raise SystemExit(f"HDFS log file not found: {args.hdfs_log}")
        if not args.hdfs_labels.exists():
            raise SystemExit(f"HDFS label file not found: {args.hdfs_labels}")

        summary_path, manifest_path = export_hdfs_statistics(
            hdfs_log_path=args.hdfs_log,
            hdfs_label_path=args.hdfs_labels,
            output_dir=args.output_dir,
        )
        print(f"[HDFS] Wrote summary:  {summary_path}")
        print(f"[HDFS] Wrote manifest: {manifest_path}")

    if args.mode in {"openstack", "both"}:
        if not args.openstack_log or args.openstack_labels is None:
            raise SystemExit(
                "For OpenStack export you must provide --openstack-log and --openstack-labels."
            )

        missing_logs = [path for path in args.openstack_log if not path.exists()]
        if missing_logs:
            raise SystemExit(
                "OpenStack log file(s) not found:\n" + "\n".join(str(path) for path in missing_logs)
            )

        if not args.openstack_labels.exists():
            raise SystemExit(f"OpenStack label file not found: {args.openstack_labels}")

        summary_path, manifest_path = export_openstack_statistics(
            openstack_log_paths=args.openstack_log,
            openstack_label_path=args.openstack_labels,
            output_dir=args.output_dir,
        )
        print(f"[OpenStack] Wrote summary:  {summary_path}")
        print(f"[OpenStack] Wrote manifest: {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
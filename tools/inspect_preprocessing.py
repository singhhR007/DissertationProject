from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from app.services.preprocessing import (
    LabeledSequenceRecord,
    build_hdfs_sequences_from_files,
    build_openstack_sequences_from_files,
)


# Formatting helpers

def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def safe_mean(values: list[int]) -> float:
    if not values:
        return 0.0
    return statistics.mean(values)


# Inspection helpers

def summarize_records(
    records: list[LabeledSequenceRecord],
    *,
    dataset_name: str,
    max_examples: int = 3,
) -> None:
    """
    Print summary statistics for one dataset's generated sequence records.
    """
    print_section(f"{dataset_name} sequence summary")

    total_sequences = len(records)
    normal_count = sum(1 for record in records if record.label == "normal")
    anomalous_count = sum(1 for record in records if record.label == "anomalous")

    event_lengths = [len(record.sequence.events) for record in records]

    print(f"Total sequences:      {total_sequences}")
    print(f"Normal sequences:     {normal_count}")
    print(f"Anomalous sequences:  {anomalous_count}")

    if event_lengths:
        print(f"Min events/sequence:  {min(event_lengths)}")
        print(f"Avg events/sequence:  {safe_mean(event_lengths):.2f}")
        print(f"Max events/sequence:  {max(event_lengths)}")
    else:
        print("No sequence lengths available.")

    print_section(f"{dataset_name} example sequences")

    for index, record in enumerate(records[:max_examples], start=1):
        sequence = record.sequence
        print(f"[Example {index}]")
        print(f"sequence_id: {sequence.sequence_id}")
        print(f"label:       {record.label}")
        print(f"source:      {sequence.source}")
        print(f"event_count: {len(sequence.events)}")

        if sequence.events:
            first_event = sequence.events[0]
            print("first_event:")
            print(f"  component: {first_event.component}")
            print(f"  severity:  {first_event.severity}")
            print(f"  service:   {first_event.service}")
            print(f"  host:      {first_event.host}")
            preview = first_event.message[:200]
            print(f"  message:   {preview}")
        print("-" * 80)


# Dataset-specific runners

def inspect_hdfs(log_path: Path, label_csv_path: Path) -> None:
    """
    Build and inspect HDFS sequences from the downloaded dataset files.
    """
    records = build_hdfs_sequences_from_files(
        log_path=log_path,
        label_csv_path=label_csv_path,
    )
    summarize_records(records, dataset_name="HDFS")


def inspect_openstack(
    log_paths: list[Path],
    anomaly_label_path: Path,
) -> None:
    """
    Build and inspect OpenStack sequences from the downloaded dataset files.
    """
    records = build_openstack_sequences_from_files(
        log_paths=log_paths,
        anomaly_label_path=anomaly_label_path,
    )
    summarize_records(records, dataset_name="OpenStack")


# Main entry point

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect sequence construction for HDFS and OpenStack datasets."
    )

    parser.add_argument("--hdfs-log", type=Path, help="Path to HDFS.log")
    parser.add_argument(
        "--hdfs-labels",
        type=Path,
        help="Path to HDFS preprocessed/anomaly_label.csv",
    )

    parser.add_argument(
        "--openstack-log",
        type=Path,
        nargs="*",
        help="Paths to OpenStack log files (e.g. abnormal + normal1 + normal2)",
    )
    parser.add_argument(
        "--openstack-labels",
        type=Path,
        help="Path to OpenStack anomaly_labels.txt",
    )

    parser.add_argument(
        "--mode",
        choices=["hdfs", "openstack", "both"],
        default="both",
        help="Which dataset(s) to inspect.",
    )

    args = parser.parse_args()

    if args.mode in {"hdfs", "both"}:
        if args.hdfs_log is None or args.hdfs_labels is None:
            raise SystemExit(
                "For HDFS inspection you must provide --hdfs-log and --hdfs-labels."
            )

        if not args.hdfs_log.exists():
            raise SystemExit(f"HDFS log file not found: {args.hdfs_log}")
        if not args.hdfs_labels.exists():
            raise SystemExit(f"HDFS label file not found: {args.hdfs_labels}")

        inspect_hdfs(args.hdfs_log, args.hdfs_labels)

    if args.mode in {"openstack", "both"}:
        if not args.openstack_log or args.openstack_labels is None:
            raise SystemExit(
                "For OpenStack inspection you must provide --openstack-log and --openstack-labels."
            )

        missing_logs = [path for path in args.openstack_log if not path.exists()]
        if missing_logs:
            raise SystemExit(
                "OpenStack log file(s) not found:\n" + "\n".join(str(path) for path in missing_logs)
            )

        if not args.openstack_labels.exists():
            raise SystemExit(f"OpenStack label file not found: {args.openstack_labels}")

        inspect_openstack(args.openstack_log, args.openstack_labels)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
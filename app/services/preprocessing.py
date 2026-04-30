from __future__ import annotations

import csv
import hashlib
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from app.schemas.prediction import PredictionRequest, RawTextFormat
from app.schemas.telemetry import LogEventV1, LogSequenceV1, LogSeverity


# Shared internal types

SequenceLabel = Literal["normal", "anomalous"]
DetectedRawTextFormat = Literal["hdfs", "openstack", "generic"]


@dataclass(slots=True)
class NormalizedLogEvent:
    """
    Internal normalized representation of a single log event.

    This object is intentionally simpler than the public API schema and is used
    after validation. It keeps only the fields that are relevant for later
    preprocessing and inference steps.
    """

    message: str
    component: str | None = None
    severity: str | None = None
    host: str | None = None
    service: str | None = None
    raw_timestamp: str | None = None


@dataclass(slots=True)
class NormalizedLogSequence:
    """
    Internal normalized representation of one logical log sequence.

    This is the core analytical unit for version 1 of the artefact.
    """

    sequence_id: str
    events: list[NormalizedLogEvent]
    source: str
    telemetry_type: str = "log_sequence"
    telemetry_schema_version: str = "log_sequence_v1"
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LabeledSequenceRecord:
    """
    Internal training/evaluation record consisting of one sequence plus a label.
    """

    sequence: NormalizedLogSequence
    label: SequenceLabel


# Dataset-specific regex patterns
# Example HDFS line:
# 081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_-1608999687919862906 terminating
HDFS_LOG_LINE_RE = re.compile(
    r"^(?P<date>\d{6})\s+"
    r"(?P<time>\d{6})\s+"
    r"(?P<pid>\d+)\s+"
    r"(?P<level>[A-Z]+)\s+"
    r"(?P<component>[^:]+):\s+"
    r"(?P<message>.*)$"
)

HDFS_BLOCK_ID_RE = re.compile(r"(blk_-?\d+)")

OPENSTACK_INSTANCE_ID_RE = re.compile(
    r"\[instance:\s*([0-9a-fA-F-]{36})]"
)

OPENSTACK_REQUEST_ID_RE = re.compile(
    r"(req-[0-9a-fA-F-]+)"
)

# Example OpenStack line:
# nova-compute.log.2017-05-14_21:27:09 2017-05-14 19:39:02.007 2931 INFO nova.virt.libvirt.driver [req-...][instance: ...] message
# The logfile prefix is optional because some lines may start directly with the timestamp.
OPENSTACK_LOG_LINE_RE = re.compile(
    r"^(?:(?P<logfile>\S+)\s+)?"
    r"(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+"
    r"(?P<pid>\d+)\s+"
    r"(?P<level>[A-Z]+)\s+"
    r"(?P<component>\S+)"
    r"(?:\s+(?P<message>.*))?$"
)


# Small normalization helpers

def _clean_optional_string(value: str | None) -> str | None:
    """
    Normalize optional strings.

    Empty strings and whitespace-only values become None.
    """
    if value is None:
        return None

    cleaned = value.strip()
    return cleaned if cleaned else None


def _normalize_message(message: str) -> str:
    """
    Normalize a log message for internal use.

    This keeps the original content semantics but removes leading/trailing
    whitespace and collapses repeated inner whitespace.
    """
    return " ".join(message.strip().split())


def _normalize_severity(value: str | LogSeverity | None) -> str | None:
    """
    Normalize severity values to the contract's lower-case vocabulary.
    """
    if value is None:
        return None

    if isinstance(value, LogSeverity):
        return value.value

    cleaned = value.strip().lower()
    if not cleaned:
        return None

    mapping = {
        "warn": "warning",
        "warning": "warning",
        "info": "info",
        "debug": "debug",
        "error": "error",
        "critical": "critical",
        "fatal": "critical",
    }
    return mapping.get(cleaned, cleaned)


def _deduplicate_preserve_order(items: Iterable[str]) -> list[str]:
    """
    Deduplicate strings while preserving first-seen order.
    """
    seen: OrderedDict[str, None] = OrderedDict()
    for item in items:
        if item not in seen:
            seen[item] = None
    return list(seen.keys())


def _derive_openstack_service(
    logfile: str | None,
    component: str | None,
) -> str | None:
    """
    Derive a lightweight service identifier for OpenStack log events.

    Preference:
    1. derive from logfile prefix, e.g. nova-compute.log... -> nova-compute
    2. fall back to the first dotted component segment, e.g.
       nova.virt.libvirt.driver -> nova
    """
    if logfile:
        cleaned = logfile.strip()
        if cleaned:
            return cleaned.split(".log", 1)[0]

    if component:
        cleaned = component.strip()
        if cleaned:
            return cleaned.split(".", 1)[0]

    return None


def _split_non_empty_raw_log_lines(raw_log_text: str) -> list[str]:
    """
    Split raw log text into non-empty lines.
    """
    normalized = raw_log_text.replace("\r\n", "\n").replace("\r", "\n")
    return [line.strip() for line in normalized.split("\n") if line.strip()]


def _build_raw_text_suffix(raw_log_text: str) -> str:
    """
    Build a short stable identifier suffix from raw text.
    """
    return hashlib.sha1(raw_log_text.encode("utf-8")).hexdigest()[:12]


def _detect_raw_text_format(
    lines: list[str],
    *,
    raw_text_format: RawTextFormat | None,
    source: str,
) -> DetectedRawTextFormat:
    """
    Detect which parser should be used for raw log text.

    Explicit format hints take precedence. Otherwise a lightweight heuristic is
    used, followed by the request source as a tie-breaker, and finally a
    generic fallback.
    """
    if raw_text_format in {"hdfs", "openstack"}:
        return raw_text_format

    hdfs_score = 0
    openstack_score = 0

    for line in lines:
        if HDFS_LOG_LINE_RE.match(line):
            hdfs_score += 2
        if extract_hdfs_block_id(line) is not None:
            hdfs_score += 1

        if OPENSTACK_LOG_LINE_RE.match(line):
            openstack_score += 2
        if extract_openstack_instance_id(line) is not None:
            openstack_score += 2
        if extract_openstack_request_id(line) is not None:
            openstack_score += 1

    if hdfs_score > openstack_score and hdfs_score > 0:
        return "hdfs"

    if openstack_score > hdfs_score and openstack_score > 0:
        return "openstack"

    source_lower = source.lower()
    if "hdfs" in source_lower:
        return "hdfs"
    if "openstack" in source_lower or "nova" in source_lower:
        return "openstack"

    return "generic"


def _derive_hdfs_raw_sequence_id(lines: list[str], raw_log_text: str) -> str:
    """
    Derive one sequence identifier for raw HDFS log text.

    A single request is expected to represent one logical sequence. If multiple
    distinct BlockIds appear, the request is rejected as ambiguous.
    """
    block_ids = _deduplicate_preserve_order(
        block_id
        for block_id in (extract_hdfs_block_id(line) for line in lines)
        if block_id is not None
    )

    if len(block_ids) > 1:
        raise ValueError(
            "raw_log_text must contain log lines for a single HDFS block sequence."
        )

    if len(block_ids) == 1:
        return block_ids[0]

    return f"raw-hdfs-{_build_raw_text_suffix(raw_log_text)}"


def _derive_openstack_raw_sequence_id(lines: list[str], raw_log_text: str) -> str:
    """
    Derive one sequence identifier for raw OpenStack log text.

    A single request is expected to represent one logical sequence. If multiple
    distinct instance IDs appear, the request is rejected as ambiguous.
    """
    instance_ids = _deduplicate_preserve_order(
        instance_id
        for instance_id in (extract_openstack_instance_id(line) for line in lines)
        if instance_id is not None
    )

    if len(instance_ids) > 1:
        raise ValueError(
            "raw_log_text must contain log lines for a single OpenStack instance sequence."
        )

    if len(instance_ids) == 1:
        return instance_ids[0]

    return f"raw-openstack-{_build_raw_text_suffix(raw_log_text)}"


def _parse_generic_log_line(line: str) -> NormalizedLogEvent | None:
    """
    Fallback parser for raw text that does not clearly match HDFS or OpenStack.
    """
    stripped = line.strip()
    if not stripped:
        return None

    return NormalizedLogEvent(message=_normalize_message(stripped))


# Public API request normalization

def normalize_log_event(event: LogEventV1) -> NormalizedLogEvent:
    """
    Convert one validated API log event into the internal normalized form.
    """
    return NormalizedLogEvent(
        message=_normalize_message(event.message),
        component=_clean_optional_string(event.component),
        severity=_normalize_severity(event.severity),
        host=_clean_optional_string(event.host),
        service=_clean_optional_string(event.service),
    )


def normalize_log_sequence(
    sequence: LogSequenceV1,
    *,
    source: str,
    context: dict[str, Any] | None = None,
) -> NormalizedLogSequence:
    """
    Convert one validated API log sequence into the internal normalized form.
    """
    return NormalizedLogSequence(
        sequence_id=sequence.sequence_id,
        events=[normalize_log_event(event) for event in sequence.events],
        source=source,
        context=context or {},
    )


def normalize_raw_log_text(
    raw_log_text: str,
    *,
    source: str,
    context: dict[str, Any] | None = None,
    raw_text_format: RawTextFormat | None = None,
) -> NormalizedLogSequence:
    """
    Convert raw log text into one normalized internal sequence for inference.

    The raw text is treated as one logical request-level sequence. Format
    detection is attempted automatically unless an explicit format hint is
    provided.
    """
    lines = _split_non_empty_raw_log_lines(raw_log_text)
    if not lines:
        raise ValueError("raw_log_text must contain at least one non-empty log line.")

    detected_format = _detect_raw_text_format(
        lines,
        raw_text_format=raw_text_format,
        source=source,
    )

    events: list[NormalizedLogEvent] = []

    if detected_format == "hdfs":
        sequence_id = _derive_hdfs_raw_sequence_id(lines, raw_log_text)
        for line in lines:
            event = parse_hdfs_log_line(line)
            if event is not None:
                events.append(event)

    elif detected_format == "openstack":
        sequence_id = _derive_openstack_raw_sequence_id(lines, raw_log_text)
        for line in lines:
            event = parse_openstack_log_line(line)
            if event is not None:
                events.append(event)

    else:
        sequence_id = f"raw-generic-{_build_raw_text_suffix(raw_log_text)}"
        for line in lines:
            event = _parse_generic_log_line(line)
            if event is not None:
                events.append(event)

    if not events:
        raise ValueError("raw_log_text did not contain any usable log lines.")

    normalized_context = dict(context or {})
    normalized_context.setdefault("input_mode", "raw_log_text")
    normalized_context.setdefault("raw_text_format", detected_format)

    return NormalizedLogSequence(
        sequence_id=sequence_id,
        events=events,
        source=source,
        context=normalized_context,
    )


def normalize_prediction_request(request: PredictionRequest) -> NormalizedLogSequence:
    """
    Convert a validated prediction request into the internal normalized form.

    This is the runtime bridge from the API layer into preprocessing/inference.
    """
    if request.log_sequence is not None:
        return normalize_log_sequence(
            request.log_sequence,
            source=request.source,
            context=request.context,
        )

    return normalize_raw_log_text(
        request.raw_log_text or "",
        source=request.source,
        context=request.context,
        raw_text_format=request.raw_text_format,
    )


# Model-facing helper representations

def sequence_to_joined_text(sequence: NormalizedLogSequence) -> str:
    """
    Convert a normalized sequence to one concatenated text representation.

    This is a useful baseline representation for simple text-based models.
    """
    return "\n".join(event.message for event in sequence.events)


def sequence_to_feature_dict(sequence: NormalizedLogSequence) -> dict[str, Any]:
    """
    Build a lightweight feature summary from a normalized sequence.

    This is not the final ML feature pipeline, but it provides a stable,
    explicit intermediate representation that can later feed classical ML
    baselines or debugging/inspection tools.
    """
    components = _deduplicate_preserve_order(
        component
        for component in (event.component for event in sequence.events)
        if component is not None
    )
    services = _deduplicate_preserve_order(
        service
        for service in (event.service for event in sequence.events)
        if service is not None
    )
    severities = [event.severity for event in sequence.events if event.severity is not None]

    return {
        "sequence_id": sequence.sequence_id,
        "event_count": len(sequence.events),
        "joined_text": sequence_to_joined_text(sequence),
        "components": components,
        "services": services,
        "severity_counts": {
            level: severities.count(level)
            for level in sorted(set(severities))
        },
    }


# HDFS preprocessing

def extract_hdfs_block_id(text: str) -> str | None:
    """
    Extract the HDFS BlockId from one raw log line or message.
    """
    match = HDFS_BLOCK_ID_RE.search(text)
    if match is None:
        return None
    return match.group(1)


def parse_hdfs_log_line(line: str) -> NormalizedLogEvent | None:
    """
    Parse one raw HDFS log line into a normalized event.

    If the line does not match the expected structure, it is still accepted
    in a fallback form as long as it is non-empty.
    """
    stripped = line.strip()
    if not stripped:
        return None

    match = HDFS_LOG_LINE_RE.match(stripped)
    if match is None:
        return NormalizedLogEvent(message=_normalize_message(stripped))

    raw_timestamp = f"{match.group('date')} {match.group('time')}"
    severity = _normalize_severity(match.group("level"))
    component = _clean_optional_string(match.group("component"))
    message = _normalize_message(match.group("message"))

    return NormalizedLogEvent(
        message=message,
        component=component,
        severity=severity,
        service="hdfs",
        raw_timestamp=raw_timestamp,
    )


def load_hdfs_label_mapping(csv_path: str | Path) -> dict[str, SequenceLabel]:
    """
    Load HDFS anomaly labels from anomaly_label.csv.

    Expected columns:
    - BlockId
    - Label
    """
    mapping: dict[str, SequenceLabel] = {}

    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            block_id = (row.get("BlockId") or "").strip()
            label_value = (row.get("Label") or "").strip().lower()

            if not block_id:
                continue

            if label_value == "anomaly":
                mapping[block_id] = "anomalous"
            else:
                mapping[block_id] = "normal"

    return mapping


def build_hdfs_sequences_from_log_lines(
    log_lines: Iterable[str],
    *,
    label_mapping: dict[str, SequenceLabel] | None = None,
    source: str = "hdfs",
) -> list[LabeledSequenceRecord]:
    """
    Build labeled HDFS sequences from raw HDFS log lines.

    Version 1 grouping rule:
    - all log lines with the same BlockId belong to one sequence
    - sequence_id = BlockId
    - event order = original log order
    """
    grouped_events: dict[str, list[NormalizedLogEvent]] = OrderedDict()

    for line in log_lines:
        block_id = extract_hdfs_block_id(line)
        if block_id is None:
            # Lines without BlockId do not participate in the HDFS v1 sequence rule.
            continue

        event = parse_hdfs_log_line(line)
        if event is None:
            continue

        grouped_events.setdefault(block_id, []).append(event)

    records: list[LabeledSequenceRecord] = []

    for block_id, events in grouped_events.items():
        sequence = NormalizedLogSequence(
            sequence_id=block_id,
            events=events,
            source=source,
            context={"dataset_origin": "HDFS"},
        )

        label: SequenceLabel = "normal"
        if label_mapping is not None:
            label = label_mapping.get(block_id, "normal")

        records.append(
            LabeledSequenceRecord(
                sequence=sequence,
                label=label,
            )
        )

    return records


# OpenStack preprocessing

def extract_openstack_instance_id(text: str) -> str | None:
    """
    Extract the OpenStack instance_id from one raw log line.
    """
    match = OPENSTACK_INSTANCE_ID_RE.search(text)
    if match is None:
        return None
    return match.group(1)


def extract_openstack_request_id(text: str) -> str | None:
    """
    Extract the OpenStack request id from one raw log line.

    This is not the primary grouping key in version 1, but it may later be
    useful for richer correlation.
    """
    match = OPENSTACK_REQUEST_ID_RE.search(text)
    if match is None:
        return None
    return match.group(1)


def parse_openstack_log_line(line: str) -> NormalizedLogEvent | None:
    """
    Parse one raw OpenStack log line into a normalized event.

    This parser is intentionally permissive:
    - If the line matches the common OpenStack format, extract timestamp,
      severity, component, message, and a lightweight service identifier.
    - If not, keep the raw line as a message-only event so that information
      is not lost during preprocessing.
    """
    stripped = line.strip()
    if not stripped:
        return None

    match = OPENSTACK_LOG_LINE_RE.match(stripped)
    if match is None:
        return NormalizedLogEvent(message=_normalize_message(stripped))

    logfile = _clean_optional_string(match.group("logfile"))
    raw_timestamp = _clean_optional_string(match.group("timestamp"))
    severity = _normalize_severity(match.group("level"))
    component = _clean_optional_string(match.group("component"))
    message = _normalize_message(match.group("message") or "")
    service = _derive_openstack_service(logfile, component)

    return NormalizedLogEvent(
        message=message,
        component=component,
        severity=severity,
        service=service,
        raw_timestamp=raw_timestamp,
    )


def load_openstack_anomalous_instance_ids(path: str | Path) -> set[str]:
    """
    Load anomalous instance IDs from anomaly_labels.txt.
    """
    instance_ids: set[str] = set()

    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            cleaned = line.strip()
            if cleaned:
                instance_ids.add(cleaned)

    return instance_ids


def build_openstack_sequences_from_log_lines(
    log_lines: Iterable[str],
    *,
    anomalous_instance_ids: set[str] | None = None,
    source: str = "openstack",
) -> list[LabeledSequenceRecord]:
    """
    Build labeled OpenStack sequences from raw OpenStack log lines.

    Version 1 grouping rule:
    - group by explicit instance_id
    - sequence_id = instance_id
    - event order = original log order

    This deliberately keeps the first implementation simple and fully
    deterministic. Lines without an explicit [instance: ...] identifier are not
    grouped into a sequence in v1.
    """
    grouped_events: dict[str, list[NormalizedLogEvent]] = OrderedDict()

    for line in log_lines:
        instance_id = extract_openstack_instance_id(line)
        if instance_id is None:
            # OpenStack v1 groups only lines that explicitly expose instance_id.
            continue

        event = parse_openstack_log_line(line)
        if event is None:
            continue

        grouped_events.setdefault(instance_id, []).append(event)

    records: list[LabeledSequenceRecord] = []

    for instance_id, events in grouped_events.items():
        sequence = NormalizedLogSequence(
            sequence_id=instance_id,
            events=events,
            source=source,
            context={"dataset_origin": "OpenStack"},
        )

        label: SequenceLabel = "normal"
        if anomalous_instance_ids is not None and instance_id in anomalous_instance_ids:
            label = "anomalous"

        records.append(
            LabeledSequenceRecord(
                sequence=sequence,
                label=label,
            )
        )

    return records


# Convenience file loaders for local experimentation

def read_text_lines(path: str | Path) -> list[str]:
    """
    Read one UTF-8 text file into a list of lines.
    """
    return Path(path).read_text(encoding="utf-8", errors="replace").splitlines()


def build_hdfs_sequences_from_files(
    *,
    log_path: str | Path,
    label_csv_path: str | Path,
) -> list[LabeledSequenceRecord]:
    """
    Convenience helper for local experiments using the downloaded HDFS files.
    """
    lines = read_text_lines(log_path)
    label_mapping = load_hdfs_label_mapping(label_csv_path)
    return build_hdfs_sequences_from_log_lines(
        lines,
        label_mapping=label_mapping,
    )


def build_openstack_sequences_from_files(
    *,
    log_paths: list[str | Path],
    anomaly_label_path: str | Path,
) -> list[LabeledSequenceRecord]:
    """
    Convenience helper for local experiments using the downloaded OpenStack files.

    Multiple log files can be combined into one iterable stream while preserving
    per-file line order.
    """
    all_lines: list[str] = []
    for path in log_paths:
        all_lines.extend(read_text_lines(path))

    anomalous_instance_ids = load_openstack_anomalous_instance_ids(anomaly_label_path)
    return build_openstack_sequences_from_log_lines(
        all_lines,
        anomalous_instance_ids=anomalous_instance_ids,
    )
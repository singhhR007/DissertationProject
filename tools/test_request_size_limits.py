from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request


# Runtime configuration
BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
TOKEN = os.getenv("API_BEARER_TOKEN", "dev-secret-token")

SINGLE_LIMIT_BYTES = 256 * 1024         # 256 KiB
BATCH_LIMIT_BYTES = 2 * 1024 * 1024     # 2 MiB

# Keep each message valid under the schema limit of 4096 characters.
TARGET_MESSAGE_LENGTH = 3000


# Payload builders
def make_valid_message(target_length: int = TARGET_MESSAGE_LENGTH) -> str:
    """
    Build a valid log message that stays below the per-field schema limit.
    """
    base = (
        "Receiving block blk_-1608999687919862906 "
        "src: /10.250.19.102:54106 dest: /10.250.19.102:50010 "
    )
    repeats = (target_length // len(base)) + 1
    message = (base * repeats)[:target_length]
    return message


def make_event(index: int) -> dict:
    """
    Build one valid log event.
    """
    return {
        "component": "DataNode",
        "host": "10.250.19.102",
        "message": f"{make_valid_message()} event_index={index}",
        "service": "hdfs",
        "severity": "info",
    }


def build_single_payload() -> dict:
    """
    Build a valid single-prediction payload that exceeds 256 KiB.

    Strategy:
    - Keep each message valid (< 4096 chars)
    - Add more valid events until the total request size exceeds the limit
    """
    payload = {
        "timestamp": "2026-04-03T12:30:00Z",
        "source": "hdfs-node-01",
        "telemetry_type": "log_sequence",
        "telemetry_schema_version": "log_sequence_v1",
        "log_sequence": {
            "sequence_id": "blk_-1608999687919862906",
            "events": [],
        },
        "context": {
            "environment": "lab"
        },
    }

    events = payload["log_sequence"]["events"]

    # Add events until the serialized payload is larger than the configured limit.
    index = 0
    while True:
        events.append(make_event(index))
        index += 1

        encoded = json.dumps(payload).encode("utf-8")
        if len(encoded) > SINGLE_LIMIT_BYTES:
            break

        if len(events) >= 500:
            raise RuntimeError(
                "Could not exceed single-request size limit without violating "
                "the max event count of 500."
            )

    return payload


def make_record(record_index: int, events_per_record: int = 10) -> dict:
    """
    Build one valid batch record with multiple valid events.
    """
    return {
        "timestamp": "2026-04-03T12:30:00Z",
        "source": f"hdfs-node-{record_index:02d}",
        "telemetry_type": "log_sequence",
        "telemetry_schema_version": "log_sequence_v1",
        "log_sequence": {
            "sequence_id": f"blk_-1608999687919862906_{record_index}",
            "events": [make_event(i) for i in range(events_per_record)],
        },
        "context": {
            "environment": "lab"
        },
    }


def build_batch_payload() -> dict:
    """
    Build a valid batch-prediction payload that exceeds 2 MiB.

    Strategy:
    - Keep every record valid
    - Keep batch size <= 100
    - Add more valid records until total payload exceeds the batch limit
    """
    payload = {"records": []}
    records = payload["records"]

    record_index = 0
    while True:
        records.append(make_record(record_index, events_per_record=10))
        record_index += 1

        encoded = json.dumps(payload).encode("utf-8")
        if len(encoded) > BATCH_LIMIT_BYTES:
            break

        if len(records) >= 100:
            raise RuntimeError(
                "Could not exceed batch-request size limit without violating "
                "the max record count of 100."
            )

    return payload


# HTTP helpers
def send_post(path: str, payload: dict) -> None:
    """
    Send a JSON POST request and print the result.
    """
    url = f"{BASE_URL}{path}"
    data = json.dumps(payload).encode("utf-8")
    print("=" * 80)
    print(f"POST {url}")
    print(f"Payload size: {len(data)} bytes")

    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {TOKEN}",
        },
    )

    try:
        with urllib.request.urlopen(req) as response:
            body = response.read().decode("utf-8", errors="replace")
            print(f"HTTP status: {response.status}")
            print("Response headers:")
            for key, value in response.headers.items():
                print(f"  {key}: {value}")
            print("Response body:")
            print(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP status: {exc.code}")
        print("Response headers:")
        for key, value in exc.headers.items():
            print(f"  {key}: {value}")
        print("Response body:")
        print(body)


# Main entry point
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test request-size limits for the local anomaly-detection API."
    )
    parser.add_argument(
        "--mode",
        choices=["single", "batch", "both"],
        default="both",
        help="Which endpoint to test.",
    )
    args = parser.parse_args()

    if args.mode in {"single", "both"}:
        single_payload = build_single_payload()
        send_post("/api/v1/predictions", single_payload)

    if args.mode in {"batch", "both"}:
        batch_payload = build_batch_payload()
        send_post("/api/v1/predictions/batch", batch_payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
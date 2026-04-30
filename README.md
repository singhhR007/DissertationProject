# Secure RESTful API for ML-Based Log Anomaly Detection

Bachelor dissertation artefact, BSc (Hons) Software Engineering at
Southampton Solent University, module COM629 (AE2), May 2026.

This repository contains the working software produced for the
dissertation *Design and Implementation of a RESTful API for
Anomaly Detection in System Log Data* by Randip Singh. The
artefact is a locally deployed FastAPI service that exposes a
calibrated TF-IDF + Logistic Regression model behind a strict,
versioned and authenticated contract, together with a React
frontend used for demonstration and evaluation.

## Quick start

The five commands below get the system running on a clean
machine. The setup was tested on Windows 11 with Python 3.13 and
Node.js 20.

```bash
# 1. Clone the repository
git clone <this-repo-url>
cd <repo-name>

# 2. Install backend dependencies
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
python -m pip install -r requirements.txt

# 3. Set the bearer token and start the backend
$env:API_BEARER_TOKEN="dev-secret-token"
uvicorn app.main:app --reload

# 4. In a second terminal, install and start the frontend
cd frontend
npm install
npm run dev

# 5. Open the application
# Backend OpenAPI UI:  http://127.0.0.1:8000/docs
# Frontend:            http://127.0.0.1:5173
```

## Project scope

The artefact brings together a supervised machine learning model,
a secure REST API and a web frontend. Version 1 of the contract
supports one telemetry type, `log_sequence`, and one schema
version, `log_sequence_v1`. Both structured JSON input and raw
multi-line log text are accepted, and any raw text is parsed
internally into the same sequence representation used for
structured input before inference is run.

The deployed model has the following properties.

| Property | Value |
|---|---|
| Model type | TF-IDF + Logistic Regression with Platt scaling |
| Training approach | Supervised binary classification |
| Classification unit | Log sequence |
| Primary dataset | HDFS |
| Secondary offline benchmark | OpenStack |
| Score type | Calibrated anomalous-class probability |
| Decision threshold | 0.4162 (validation-tuned) |

The deployed model file is expected at:

```
artefacts/models/hdfs_baseline_calibrated/hdfs_baseline.joblib
```

## API endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/api/v1/health` | Public | Liveness and readiness check |
| GET | `/api/v1/model/info` | Bearer | Deployment metadata |
| POST | `/api/v1/predictions` | Bearer | Single sequence inference |
| POST | `/api/v1/predictions/batch` | Bearer | Batch inference, up to 100 records |

## Authentication

Protected endpoints expect an `Authorization: Bearer <token>`
header. The default development token is `dev-secret-token`. You
can override it with the `API_BEARER_TOKEN` environment variable.

```powershell
# Windows PowerShell
$env:API_BEARER_TOKEN="your-token"
uvicorn app.main:app --reload
```

```bash
# macOS / Linux
export API_BEARER_TOKEN="your-token"
uvicorn app.main:app --reload
```

## Example prediction request

```json
{
  "timestamp": "2026-04-03T12:30:00Z",
  "source": "hdfs-node-01",
  "telemetry_type": "log_sequence",
  "telemetry_schema_version": "log_sequence_v1",
  "log_sequence": {
    "sequence_id": "blk_-1608999687919862906",
    "events": [
      {
        "message": "Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010",
        "component": "DataNode",
        "severity": "info",
        "host": "10.250.19.102",
        "service": "hdfs"
      },
      {
        "message": "PacketResponder 1 for block blk_-1608999687919862906 terminating",
        "component": "DataNode",
        "severity": "info",
        "host": "10.250.19.102",
        "service": "hdfs"
      }
    ]
  },
  "context": {
    "environment": "lab"
  }
}
```

The full API contract, including all schemas, validation rules,
error codes and edge cases, is in Appendix C of the dissertation
report.

## Tests

The pytest suite has 34 tests across six files and covers schema
validation, bearer-token authentication, raw-log-text
preprocessing, inference logic and endpoint behaviour. The full
suite runs in about two seconds on a development machine.

```bash
$env:API_BEARER_TOKEN="dev-secret-token"
python -m pytest tests
```

The complete test output is in Appendix D of the dissertation
report.

## Training scripts

### Inspect the preprocessing pipeline

```bash
python -m tools.inspect_preprocessing --mode both \
  --hdfs-log "..\HDFS_v1\HDFS.log" \
  --hdfs-labels "..\HDFS_v1\preprocessed\anomaly_label.csv" \
  --openstack-log "..\OpenStack\openstack_abnormal.log" \
                  "..\OpenStack\openstack_normal1.log" \
                  "..\OpenStack\openstack_normal2.log" \
  --openstack-labels "..\OpenStack\anomaly_labels.txt"
```

### Train the classical HDFS configurations

```bash
python -m tools.train_hdfs_baseline \
  --hdfs-log "..\HDFS_v1\HDFS.log" \
  --hdfs-labels "..\HDFS_v1\preprocessed\anomaly_label.csv" \
  --output-dir "artefacts\models\hdfs_baseline_calibrated"
```

### Optional BiLSTM smoke test

```bash
python -m tools.train_hdfs_bilstm \
  --hdfs-log "..\HDFS_v1\HDFS.log" \
  --hdfs-labels "..\HDFS_v1\preprocessed\anomaly_label.csv" \
  --output-dir "artefacts\models\hdfs_bilstm"
```

## Notes on datasets

HDFS is the primary supervised training dataset. OpenStack is
included as a secondary offline benchmark for preprocessing
validation and for discussion of cross-dataset behaviour. OpenStack
is not used as the training basis of the deployed model because it
contains a very small number of labelled anomalous sequences.

Both datasets come from the Loghub collection. The expected
directory layout matches the paths shown in the training commands
above.

## Versions and tested environment

- Python 3.13
- Node.js 20
- FastAPI 0.135.3
- Pydantic 2.12.5
- Uvicorn 0.44.0
- React 18 with TypeScript and Vite
- Tested on Windows 11

The full pinned dependency list is in `requirements.txt` for the
backend and `frontend/package.json` for the frontend.

## Documentation

The dissertation report contains the full documentation for the
artefact, including:

- The literature review and research gap (Chapter 2)
- The methodology and requirements analysis (Chapter 3)
- The implementation and design decisions (Chapter 4)
- The empirical evaluation results (Chapter 5)
- The full API contract specification (Appendix C)
- The full test suite output (Appendix D)
- The model hyperparameters (Appendix H)
- The calibration curves (Appendix I)

## Author

Randip Singh  
BSc (Hons) Software Engineering, Southampton Solent University  
Module COM629 (AE2), submitted May 2026  
Supervisor: Daniel Olabanji

## License

This repository is part of a BSc dissertation submission. The code
is shared for educational and academic review purposes.

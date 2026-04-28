"""프로젝트 루트 기준 경로 (데이터 / 모델 / 출력 / 문서 분리).

소스는 py/ 아래에 두고, data·models·output·docs는 프로젝트 루트에 둡니다.
"""

from __future__ import annotations

from pathlib import Path

# py/project_paths.py → 프로젝트 루트는 한 단계 위
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
DOCS_DIR = PROJECT_ROOT / "docs"

# 기본 파일명
DATA_CSV = DATA_DIR / "sensor_data.csv"
DATA_CSV_TRAIN = DATA_DIR / "sensor_train.csv"
DATA_CSV_TEST = DATA_DIR / "sensor_test.csv"
MODEL_PKL = MODELS_DIR / "anomaly_model.pkl"
MODEL_KERAS = MODELS_DIR / "anomaly_model.keras"
MODEL_PKL_WINDOW = MODELS_DIR / "anomaly_model_window.pkl"
MODEL_PKL_ROW = MODELS_DIR / "anomaly_model_row.pkl"
# LSTM 예지보전(교사·학생 증류 후 학생 가중치): Keras + joblib 메타
MODEL_KERAS_PM_LSTM = MODELS_DIR / "anomaly_pm_lstm.keras"
MODEL_PKL_PM_LSTM = MODELS_DIR / "anomaly_pm_lstm.pkl"
OUTPUT_PLOT_DEFAULT = OUTPUT_DIR / "anomaly_result.png"
OUTPUT_DETECTION_DEFAULT = OUTPUT_DIR / "detection_results.csv"
OUTPUT_PM_WINDOW_DEFAULT = OUTPUT_DIR / "pm_window_results.csv"
GPU_SETUP_DOC = DOCS_DIR / "GPU_SETUP_WSL.txt"


def resolve(path_like: str | Path) -> Path:
    """상대 경로는 프로젝트 루트 기준."""
    p = Path(path_like)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def ensure_dirs() -> None:
    for d in (DATA_DIR, MODELS_DIR, OUTPUT_DIR, DOCS_DIR):
        d.mkdir(parents=True, exist_ok=True)

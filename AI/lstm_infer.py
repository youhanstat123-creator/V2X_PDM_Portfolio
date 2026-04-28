"""학습된 LSTM + 스케일러(joblib)로 윈도우 단위 이상 점수 추론.

모델이 없으면 None을 반환하고 호출부에서 규칙 기반만 사용.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

import project_paths as pp
from sensor_data_config import FEATURE_COLS

WINDOW_SIZE = 12


class LstmAnomalyEngine:
    def __init__(self) -> None:
        self._bundle: dict[str, Any] | None = None
        self._model = None
        self._threshold = 0.5
        self._ok = False

    def load(self) -> bool:
        pp.ensure_dirs()
        pkl = str(pp.MODEL_PKL)
        keras_path = str(pp.MODEL_KERAS)
        if not os.path.isfile(pkl) or not os.path.isfile(keras_path):
            return False
        try:
            import joblib
            from tensorflow.keras.models import load_model

            self._bundle = joblib.load(pkl)
            self._model = load_model(keras_path)
            self._threshold = float(self._bundle.get("threshold", 0.5))
            self._ok = True
            return True
        except Exception:
            self._ok = False
            return False

    @property
    def available(self) -> bool:
        return self._ok

    def score_window(self, features_2d: np.ndarray) -> float | None:
        """features_2d: shape (WINDOW_SIZE, len(FEATURE_COLS)) 원시 스케일(학습과 동일 단위)."""
        if not self._ok or self._model is None or self._bundle is None:
            return None
        if features_2d.shape != (WINDOW_SIZE, len(FEATURE_COLS)):
            return None
        try:
            import joblib

            scaler = self._bundle["raw_scaler"]
            X = scaler.transform(features_2d.astype(float))
            Xw = X.reshape(1, WINDOW_SIZE, len(FEATURE_COLS)).astype(np.float32)
            pred = self._model.predict(Xw, verbose=0).reshape(-1)[0]
            return float(pred)
        except Exception:
            return None

    def is_anomaly(self, score: float | None) -> bool:
        if score is None:
            return False
        return score >= self._threshold


def features_matrix_from_rows(rows: list[dict]) -> np.ndarray | None:
    """rows: FEATURE_COLS 키를 가진 dict 목록, 길이 WINDOW_SIZE."""
    if len(rows) != WINDOW_SIZE:
        return None
    mat = np.zeros((WINDOW_SIZE, len(FEATURE_COLS)), dtype=np.float64)
    for i, r in enumerate(rows):
        for j, c in enumerate(FEATURE_COLS):
            mat[i, j] = float(r.get(c, 0.0))
    return mat

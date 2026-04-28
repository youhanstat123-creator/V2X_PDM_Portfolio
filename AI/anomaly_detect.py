"""지도학습 LSTM 이상탐지 스크립트.

- LSTM 입력은 항상 FEATURE_COLS(9개)만 사용. severity는 스케일러·신경망에 넣지 않음.
- 테스트 CSV에 severity가 있으면: 추론 후 윈도우 단위로 y_true와 비교해 임계값·정밀도·재현율·오탐률 산출.
- 학습 CSV 검증 구간의 오탐률은 데이터·모델에 따라 0이 아닐 수 있음(임계값이 검증 점수로 튜닝되므로).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

import project_paths as pp
from sensor_data_config import (
    COL_CONTROLLER_DEVICE_ID,
    COL_CONTROLLER_LOG_TIME,
    COL_SEVERITY,
    CSV_COLUMNS,
    CSV_COLUMNS_WITHOUT_SEVERITY,
    FEATURE_COLS,
)

MODEL_PKL = str(pp.MODEL_PKL)
MODEL_KERAS = str(pp.MODEL_KERAS)
WINDOW_SIZE = 12
MIN_ROWS_PER_DEVICE = WINDOW_SIZE + 5


@dataclass(frozen=True)
class WindowedData:
    X: np.ndarray
    end_time: np.ndarray
    label: np.ndarray


def load_sensor_csv(path: str, *, expect_severity: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    if {COL_CONTROLLER_DEVICE_ID, COL_CONTROLLER_LOG_TIME} - set(df.columns):
        names = list(CSV_COLUMNS) if expect_severity else list(CSV_COLUMNS_WITHOUT_SEVERITY)
        df = pd.read_csv(path, header=None, names=names)
    df[COL_CONTROLLER_LOG_TIME] = pd.to_datetime(df[COL_CONTROLLER_LOG_TIME], errors="coerce")
    df = df.dropna(subset=[COL_CONTROLLER_LOG_TIME]).copy()
    if expect_severity:
        if COL_SEVERITY not in df.columns:
            raise ValueError(f"학습용 CSV에는 {COL_SEVERITY} 컬럼이 필요합니다: {path}")
        df[COL_SEVERITY] = df[COL_SEVERITY].astype(str).str.strip()
    elif COL_SEVERITY in df.columns:
        df = df.drop(columns=[COL_SEVERITY])
    return df


def split_by_time(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sdf = df.sort_values(COL_CONTROLLER_LOG_TIME).reset_index(drop=True)
    n = len(sdf)
    c1 = int(n * 0.6)
    c2 = c1 + int(n * 0.2)
    return sdf.iloc[:c1].copy(), sdf.iloc[c1:c2].copy(), sdf.iloc[c2:].copy()


def fit_scaler(train_df: pd.DataFrame) -> MinMaxScaler:
    fit_df = train_df.copy()
    if COL_SEVERITY in fit_df.columns:
        normal = fit_df[fit_df[COL_SEVERITY].astype(str).str.strip() == "정상"]
        if len(normal) >= MIN_ROWS_PER_DEVICE:
            fit_df = normal
    scaler = MinMaxScaler()
    scaler.fit(fit_df.loc[:, FEATURE_COLS].astype(float).to_numpy())
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: MinMaxScaler) -> pd.DataFrame:
    """FEATURE_COLS만 변환. severity 등 그 외 열은 그대로(없으면 없음)."""
    out = df.copy()
    out = out.astype({col: "float64" for col in FEATURE_COLS})
    scaled = scaler.transform(out.loc[:, FEATURE_COLS].to_numpy())
    out.loc[:, FEATURE_COLS] = scaled
    return out


def apply_scaler_drop_severity_first(df: pd.DataFrame, scaler: MinMaxScaler) -> pd.DataFrame:
    """테스트 평가용: severity를 스케일 입력에서 분리한 뒤, 스케일된 프레임에 라벨만 다시 붙임."""
    if COL_SEVERITY not in df.columns:
        return apply_scaler(df, scaler)
    sev = df[COL_SEVERITY].astype(str).str.strip()
    base = df.drop(columns=[COL_SEVERITY])
    out = apply_scaler(base, scaler)
    out[COL_SEVERITY] = sev.values
    return out


def make_windows(df: pd.DataFrame, window: int) -> WindowedData:
    parts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for _, g in df.groupby(COL_CONTROLLER_DEVICE_ID, sort=False):
        g = g.sort_values(COL_CONTROLLER_LOG_TIME)
        if len(g) < MIN_ROWS_PER_DEVICE:
            continue
        feat = g.loc[:, FEATURE_COLS].astype(float).to_numpy()
        X_dev = sliding_window_view(feat, window, axis=0)
        if X_dev.shape[1] == len(FEATURE_COLS) and X_dev.shape[2] == window:
            X_dev = np.swapaxes(X_dev, 1, 2)
        n_w = X_dev.shape[0]
        parts.append(X_dev)
        ends.append(g[COL_CONTROLLER_LOG_TIME].to_numpy()[window - 1 : window - 1 + n_w])
        if COL_SEVERITY in g.columns:
            sev = g[COL_SEVERITY].to_numpy()
        else:
            sev = np.array([""] * len(g), dtype=object)
        labels.append(sev[window - 1 : window - 1 + n_w])
    if not parts:
        raise ValueError("윈도우를 만들 수 있는 데이터가 부족합니다.")
    return WindowedData(
        X=np.concatenate(parts, axis=0),
        end_time=np.concatenate(ends, axis=0),
        label=np.concatenate(labels, axis=0),
    )


def split_windows_by_cutoff(
    win: WindowedData, train_time_max: pd.Timestamp, val_time_max: pd.Timestamp
) -> tuple[WindowedData, WindowedData, WindowedData]:
    et = pd.to_datetime(win.end_time, errors="coerce")
    m_tr = et <= pd.Timestamp(train_time_max)
    m_va = (~m_tr) & (et <= pd.Timestamp(val_time_max))
    m_te = et > pd.Timestamp(val_time_max)
    return (
        WindowedData(win.X[m_tr], win.end_time[m_tr], win.label[m_tr]),
        WindowedData(win.X[m_va], win.end_time[m_va], win.label[m_va]),
        WindowedData(win.X[m_te], win.end_time[m_te], win.label[m_te]),
    )


def make_y(win: WindowedData) -> np.ndarray:
    return (np.asarray(win.label, dtype=object) == "위험").astype(np.int8)


def build_lstm(window_size: int, n_features: int) -> Sequential:
    model = Sequential(
        [
            LSTM(64, input_shape=(window_size, n_features), return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=3e-4), loss="binary_crossentropy")
    return model


def metrics_from_scores(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[float, float, float, int]:
    y = np.asarray(y_true, dtype=np.int8).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    pred = (s >= float(threshold)).astype(np.int8)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    n_pred = int(np.sum(pred))
    return precision, recall, fpr, n_pred


def tune_threshold(y_val: np.ndarray, val_scores: np.ndarray) -> float:
    grid = np.unique(np.quantile(val_scores, np.linspace(0.01, 0.999, 500)))
    best_thr = float(np.quantile(val_scores, 0.95))
    best_key: tuple[float, float, float] | None = None
    for thr in grid:
        p, r, f, _ = metrics_from_scores(y_val, val_scores, float(thr))
        key = (r, p, -f)
        if best_key is None or key > best_key:
            best_key = key
            best_thr = float(thr)
    return best_thr


def clamp_threshold(threshold: float, low: float, high: float) -> float:
    lo = float(min(low, high))
    hi = float(max(low, high))
    return float(min(max(float(threshold), lo), hi))


def count_danger_rows(df: pd.DataFrame) -> int:
    if COL_SEVERITY not in df.columns:
        return 0
    return int((df[COL_SEVERITY].astype(str).str.strip() == "위험").sum())


def csv_has_severity_column(path: str) -> bool:
    peek = pd.read_csv(path, nrows=0)
    return COL_SEVERITY in peek.columns


def main() -> None:
    pp.ensure_dirs()
    p = argparse.ArgumentParser(
        description="학습 CSV로 학습 후 테스트 CSV로 이상탐지 (테스트에 severity 있으면 오탐률 등 평가)",
    )
    p.add_argument("--train-csv", default=str(pp.DATA_CSV_TRAIN), help="학습용 CSV (severity 포함)")
    p.add_argument(
        "--test-csv",
        default=str(pp.DATA_CSV_TEST),
        help="테스트 CSV (severity 포함 권장: 실제 이상 개수·오탐률 산출)",
    )
    p.add_argument("--model-pkl", default=MODEL_PKL)
    p.add_argument("--model-keras", default=MODEL_KERAS)
    p.add_argument("--mode", choices=["auto", "train", "detect", "train_detect"], default="auto")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--임계값-최소", type=float, default=0.3)
    p.add_argument("--임계값-최대", type=float, default=0.4)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--plot-file", default=str(pp.OUTPUT_PLOT_DEFAULT))
    args = p.parse_args()

    train_csv = str(pp.resolve(args.train_csv))
    test_csv = str(pp.resolve(args.test_csv))
    model_path = str(pp.resolve(args.model_pkl))
    keras_path = str(pp.resolve(args.model_keras))
    plot_path = None if args.no_plot else str(pp.resolve(args.plot_file))

    mode = args.mode
    if mode == "auto":
        has_bundle = os.path.exists(model_path) and os.path.exists(keras_path)
        mode = "detect" if has_bundle else "train_detect"

    if mode in ("train_detect", "detect") and not os.path.isfile(test_csv):
        raise FileNotFoundError(f"테스트 CSV가 없습니다: {test_csv} — generate_sensor_data_test.py 로 생성하세요.")
    if mode in ("train", "train_detect") and not os.path.isfile(train_csv):
        raise FileNotFoundError(f"학습 CSV가 없습니다: {train_csv} — generate_sensor_data_train.py 로 생성하세요.")

    scaler: MinMaxScaler | None = None
    model = None
    threshold = 0.0

    if mode in ("train", "train_detect"):
        raw_train = load_sensor_csv(train_csv, expect_severity=True)
        train_df, val_df, _ = split_by_time(raw_train)
        scaler = fit_scaler(train_df)
        scaled_train = apply_scaler(raw_train, scaler)
        all_win_train = make_windows(scaled_train, WINDOW_SIZE)
        train_win, val_win, _ = split_windows_by_cutoff(
            all_win_train,
            pd.Timestamp(train_df[COL_CONTROLLER_LOG_TIME].max()),
            pd.Timestamp(val_df[COL_CONTROLLER_LOG_TIME].max()),
        )

        y_train = make_y(train_win)
        y_val = make_y(val_win)

        model = build_lstm(WINDOW_SIZE, len(FEATURE_COLS))
        es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=0)
        model.fit(
            train_win.X.astype(np.float32),
            y_train.astype(np.float32),
            validation_data=(val_win.X.astype(np.float32), y_val.astype(np.float32)),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            verbose=1,
            callbacks=[es],
        )
        model.save(keras_path)
        val_scores = model.predict(val_win.X.astype(np.float32), verbose=0).reshape(-1)
        threshold = tune_threshold(y_val, val_scores)
        threshold = clamp_threshold(threshold, args.임계값_최소, args.임계값_최대)
        joblib.dump({"raw_scaler": scaler, "threshold": threshold, "window_size": WINDOW_SIZE}, model_path)
        print(f"모델 저장: {model_path}")
        print(f"Keras 저장: {keras_path}")

        n_tr = len(raw_train)
        n_d_tr = count_danger_rows(raw_train)
        print(f"[학습 CSV] 행 단위: 전체 {n_tr}행, 실제 위험(이상) {n_d_tr}행")

        p_v, r_v, f_v, n_pred_v = metrics_from_scores(y_val, val_scores, threshold)
        n_true_val = int(np.sum(y_val))
        print(
            f"[학습 CSV · 검증 구간 평가] 윈도우 {len(val_scores)}개 | "
            f"임계값={threshold:.6f} | 정밀도={p_v:.3f} | 재현율={r_v:.3f} | 오탐률={f_v:.3f} | "
            f"예측 양성(이상 탐지) {n_pred_v}개 | 실제 이상 윈도우 {n_true_val}개"
        )

        if mode == "train":
            if plot_path:
                is_anom_v = val_scores >= threshold
                plt.figure(figsize=(10, 4))
                plt.plot(val_scores, label="anomaly_score (validation)")
                plt.scatter(np.where(is_anom_v)[0], val_scores[is_anom_v], s=20, label="predicted positive")
                plt.axhline(threshold, color="r", linestyle="--", label="threshold")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.title("Train CSV — validation split")
                plt.tight_layout()
                plt.savefig(plot_path, dpi=150, bbox_inches="tight")
                plt.close()
            return

    if mode == "detect":
        if not os.path.exists(model_path) or not os.path.exists(keras_path):
            raise FileNotFoundError(
                f"detect 모드에 필요한 모델이 없습니다. pkl={os.path.exists(model_path)} keras={os.path.exists(keras_path)}"
            )
        bundle = joblib.load(model_path)
        scaler = bundle["raw_scaler"]
        threshold = float(bundle["threshold"])
        threshold = clamp_threshold(threshold, args.임계값_최소, args.임계값_최대)
        model = load_model(keras_path)
        print(f"모델 로드: {model_path}")

    assert scaler is not None and model is not None

    # 테스트: 9 피처만 스케일·모델 입력. severity는 파일에 있어도 스케일 단계에서 분리 후, 윈도우 라벨용으로만 결합.
    if mode in ("train_detect", "detect"):
        has_labels = csv_has_severity_column(test_csv)
        raw_test = load_sensor_csv(test_csv, expect_severity=has_labels)
        n_rows = len(raw_test)
        n_danger_rows = count_danger_rows(raw_test) if has_labels else 0

        scaled_test = apply_scaler_drop_severity_first(raw_test, scaler)
        test_win = make_windows(scaled_test, WINDOW_SIZE)
        test_scores = model.predict(test_win.X.astype(np.float32), verbose=0).reshape(-1)

        y_test: np.ndarray | None = None
        if has_labels:
            y_test = make_y(test_win)
            n_true_win = int(np.sum(y_test))
            print(
                f"[테스트 CSV] 행 단위: 전체 {n_rows}행, 실제 위험(이상) {n_danger_rows}행 "
                f"(파일에서 확인 가능)"
            )
            print(
                f"[테스트 CSV] 윈도우 단위: 전체 {len(test_scores)}개, "
                f"실제 이상(위험) 윈도우 {n_true_win}개 (윈도우 끝 시각 기준 라벨)"
            )
        else:
            print(
                f"[테스트 CSV] severity 열 없음 — 행/윈도우 단위 실제 이상 개수·오탐률은 산출하지 않습니다. "
                f"(generate_sensor_data_test.py 기본 생성은 severity 포함)"
            )

        n_anom = int(np.sum(test_scores >= threshold))
        thr_used = threshold

        if not has_labels and n_anom == 0 and len(test_scores) > 0:
            thr_used = float(args.임계값_최대)
            n_anom = int(np.sum(test_scores >= thr_used))

        if has_labels and y_test is not None:
            p_t, r_t, f_t, n_pred_t = metrics_from_scores(y_test, test_scores, thr_used)
            n_true_t = int(np.sum(y_test))
            print(
                f"[테스트 CSV · 평가] 윈도우 {len(test_scores)}개 | "
                f"임계값={thr_used:.6f} | 정밀도={p_t:.3f} | 재현율={r_t:.3f} | 오탐률={f_t:.3f} | "
                f"예측 양성(이상 탐지) {n_pred_t}개 | 실제 이상 윈도우 {n_true_t}개"
            )
        else:
            print(
                f"[테스트 CSV] 윈도우 {len(test_scores)}개 중 이상 탐지 {n_anom}개 | 임계값={thr_used:.6f}"
            )

        if plot_path:
            is_anom = test_scores >= thr_used
            plt.figure(figsize=(10, 4))
            plt.plot(test_scores, label="anomaly_score (test)")
            plt.scatter(np.where(is_anom)[0], test_scores[is_anom], s=20, label="predicted positive")
            plt.axhline(thr_used, color="r", linestyle="--", label="threshold")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.title("Test CSV" + (" — with labels" if has_labels else " — no labels"))
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()


if __name__ == "__main__":
    main()

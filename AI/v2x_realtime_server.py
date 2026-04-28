"""V2XCONTROL 웹용 실시간 더미 데이터 인입 + LSTM 이상탐지 연동 서버.

- PostgreSQL(spring.datasource와 동일 DB)에 intersection_info·로그·risk_analysis_result 삽입
- `sensor_collector.generate_sensor_data` 로 센서 행 생성 (교차로 ID는 ICN-01 … 형식)
- `models/anomaly_model.pkl` + `anomaly_model.keras` 가 있으면 윈도우(12샘플) 단위 LSTM 점수 반영

실행 (프로젝트 루트 또는 py 폴더에서):

    cd py
    pip install -r requirements.txt
    python v2x_realtime_server.py

환경변수:
    DATABASE_URL   postgresql://postgres:1234@localhost:5432/postgres
    TICK_SEC       기본 3 — 한 번에 한 교차로·한 쌍의 로그를 넣는 주기(초)

기동 시 기본 동작: risk_analysis_result, signal_controller_log, v2x_communication_log,
intersection_info 를 TRUNCATE 후 교차로 3곳(ICN-01~03)만 다시 넣고 새로 적재합니다.
(admin_user 등은 유지) 이어서 쓰려면: python v2x_realtime_server.py --no-reset
"""

from __future__ import annotations

import argparse
import os
import random
import signal
import sys
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Any

import psycopg2

import project_paths as pp
import sensor_collector as sc
from lstm_infer import WINDOW_SIZE, LstmAnomalyEngine, features_matrix_from_rows
from sensor_data_config import (
    COL_AVG_LATENCY_MS,
    COL_COMM_FAIL_COUNT,
    COL_CONNECTED_VEHICLE_COUNT,
    COL_CONTROLLER_LOG_TIME,
    COL_CPU_TEMP,
    COL_ERROR_COUNT,
    COL_REBOOT_COUNT,
    COL_RESPONSE_TIME_MS,
    COL_SEVERITY,
    COL_SPAT_FAIL_COUNT,
    COL_SPAT_SEND_COUNT,
    FEATURE_COLS,
)

# 웹·추천 로직과 맞춘 교차로 (RecommendService deviceId substring(4) 가정)
INTERSECTION_META: dict[str, tuple[str, float, float]] = {
    "ICN-01": ("인천시청입구 삼거리", 37.4488, 126.7025),
    "ICN-02": ("예술회관역 사거리", 37.4475, 126.7020),
    "ICN-03": ("중앙공원 사거리", 37.4460, 126.6995),
}


def _dsn() -> str:
    return os.environ.get("DATABASE_URL", "postgresql://postgres:1234@localhost:5432/postgres")


def ensure_intersections(cur) -> None:
    now = datetime.now()
    for iid, (name, lat, lng) in INTERSECTION_META.items():
        cur.execute(
            """
            INSERT INTO intersection_info (intersection_id, intersection_name, latitude, longitude, created_at)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (intersection_id) DO NOTHING
            """,
            (iid, name, lat, lng, now),
        )


def reset_simulation_data(cur) -> None:
    """서버 기동 시 이전 시뮬레이션 데이터를 비우고 ID 시퀀스를 맞춘 뒤, 교차로 메타만 다시 넣기 전 단계까지 처리.

    - risk / 로그 / 교차로(시뮬용) truncate — admin_user 등 앱 계정 테이블은 건드리지 않음.
    - FK 순서는 TRUNCATE … 한 번에 나열 + CASCADE 로 PostgreSQL이 처리.
    """
    cur.execute(
        """
        TRUNCATE TABLE
            risk_analysis_result,
            signal_controller_log,
            v2x_communication_log,
            intersection_info
        RESTART IDENTITY CASCADE
        """
    )


def risk_level_from_total(total: float) -> str:
    # 대시보드(교차로별 최신 1건)와 맞추기: 휴리스틱+LSTM+심각도 가산 후 구간
    if total < 28:
        return "정상"
    if total < 45:
        return "주의"
    if total < 68:
        return "경고"
    return "위험"


def heuristic_risk_scores(data: dict[str, Any], lstm_score: float | None) -> tuple[float, float, float]:
    """controller_risk_score, v2x_risk_score, total_risk_score (0~100)."""
    cpu = float(data[COL_CPU_TEMP])
    resp = float(data[COL_RESPONSE_TIME_MS])
    err = float(data[COL_ERROR_COUNT])
    lat = float(data[COL_AVG_LATENCY_MS])
    spat_f = float(data[COL_SPAT_FAIL_COUNT])
    comm_f = float(data[COL_COMM_FAIL_COUNT])

    ctrl = min(
        100.0,
        max(0.0, (cpu - 38.0) * 1.1 + err * 1.8 + max(0.0, resp - 45.0) * 0.12),
    )
    vx = min(100.0, max(0.0, lat * 0.42 + comm_f * 1.5 + spat_f * 0.35))

    base = (ctrl + vx) / 2.0
    if lstm_score is not None:
        base = min(100.0, base * 0.65 + float(lstm_score) * 100.0 * 0.35)
    return round(ctrl, 2), round(vx, 2), round(min(100.0, base), 2)


def build_comment(severity: str, lstm_score: float | None, lstm_flag: bool) -> str:
    extra = ""
    if lstm_flag and lstm_score is not None:
        extra = f" LSTM 이상확률 {lstm_score:.3f}."
    return f"[실시간] 심각도 {severity}.{extra}"


def tick_once(
    conn,
    buffers: dict[str, deque[dict[str, Any]]],
    engine: LstmAnomalyEngine,
) -> None:
    iid = random.choice(list(INTERSECTION_META.keys()))
    # 웹 추천 화면과 동일 패턴: intersection ICN-01 → CTRL-01, V2X-RSE-01
    suffix = iid.split("-", 1)[1]
    dev = f"CTRL-{suffix}"
    v2x_dev = f"V2X-RSE-{suffix}"
    data = sc.generate_sensor_data(iid, dev)

    now = datetime.now()
    log_time = datetime.strptime(data[COL_CONTROLLER_LOG_TIME], "%Y-%m-%d %H:%M:%S")
    uptime = random.randint(800, 20000)

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO signal_controller_log (
                intersection_id, controller_device_id, controller_log_time,
                response_time_ms, error_count, reboot_count, cpu_temp, uptime_min, created_at
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING controller_log_id
            """,
            (
                iid,
                dev,
                log_time,
                data[COL_RESPONSE_TIME_MS],
                int(data[COL_ERROR_COUNT]),
                int(data[COL_REBOOT_COUNT]),
                data[COL_CPU_TEMP],
                uptime,
                now,
            ),
        )
        cid = cur.fetchone()[0]

        cur.execute(
            """
            INSERT INTO v2x_communication_log (
                intersection_id, v2x_device_id, v2x_log_time,
                spat_send_count, spat_fail_count, avg_latency_ms,
                comm_fail_count, connected_vehicle_count, created_at
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING communication_log_id
            """,
            (
                iid,
                v2x_dev,
                log_time,
                int(data[COL_SPAT_SEND_COUNT]),
                int(data[COL_SPAT_FAIL_COUNT]),
                data[COL_AVG_LATENCY_MS],
                int(data[COL_COMM_FAIL_COUNT]),
                int(data[COL_CONNECTED_VEHICLE_COUNT]),
                now,
            ),
        )
        vid = cur.fetchone()[0]

        feat_row = {c: data[c] for c in FEATURE_COLS}
        buffers[dev].append(feat_row)

        lstm_score: float | None = None
        lstm_used = False
        if engine.available and len(buffers[dev]) >= WINDOW_SIZE:
            rows = list(buffers[dev])[-WINDOW_SIZE:]
            mat = features_matrix_from_rows(rows)
            if mat is not None:
                lstm_score = engine.score_window(mat)
                lstm_used = lstm_score is not None

        sev = str(data[COL_SEVERITY]).strip()
        ctrl_s, vx_s, total = heuristic_risk_scores(data, lstm_score)
        # 센서 심각도(health 기반)는 위험 분석 등급과 일치해야 함.
        # LSTM 사용 시에도 가산: 그렇지 않으면 total이 낮아 최신 행이 전부 "정상"으로만 쌓임.
        if sev == "위험":
            total = min(100.0, total + (25.0 if not lstm_used else 18.0))
        elif sev == "주의":
            total = min(100.0, total + (12.0 if not lstm_used else 8.0))

        level = risk_level_from_total(float(total))
        comment = build_comment(sev, lstm_score, lstm_used)

        cur.execute(
            """
            INSERT INTO risk_analysis_result (
                intersection_id, controller_log_id, communication_log_id,
                analysis_time, controller_risk_score, v2x_risk_score, total_risk_score,
                risk_level, analysis_comment, created_at
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                iid,
                cid,
                vid,
                now,
                ctrl_s,
                vx_s,
                total,
                level,
                comment,
                now,
            ),
        )
    conn.commit()
    lstm_txt = f"LSTM={lstm_score:.3f}" if lstm_score is not None else "LSTM=n/a"
    print(
        f"[{now.strftime('%H:%M:%S')}] {iid} {dev} | {level} total={total:.1f} | {lstm_txt} | inserted risk #{cid}/{vid}",
        flush=True,
    )


def main() -> None:
    pp.ensure_dirs()
    parser = argparse.ArgumentParser(description="V2X 실시간 더미 + 이상탐지 → PostgreSQL")
    parser.add_argument("--tick", type=float, default=float(os.environ.get("TICK_SEC", "3")), help="초 단위 주기")
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="기동 시 DB 초기화(truncate) 생략 — 기존 데이터에 이어서 삽입",
    )
    args = parser.parse_args()

    sc.INTERSECTION_IDS = list(INTERSECTION_META.keys())
    sc.DEVICE_IDS = [f"CTRL-{k.split('-')[1]}" for k in INTERSECTION_META.keys()]

    engine = LstmAnomalyEngine()
    if engine.load():
        print("LSTM 모델 로드 성공 — 윈도우 점수를 위험도에 반영합니다.", flush=True)
    else:
        print(
            "LSTM 모델 없음 — models/anomaly_model.pkl + anomaly_model.keras 를 두면 추론 활성화.",
            flush=True,
        )

    buffers: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=WINDOW_SIZE + 5))

    running = True

    def _stop(*_a: Any) -> None:
        nonlocal running
        running = False
        print("\n종료 신호 — 루프를 마칩니다.", flush=True)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        conn = psycopg2.connect(_dsn())
        conn.autocommit = False
    except Exception as e:
        print(f"DB 연결 실패: {e}\nDATABASE_URL={_dsn()}", file=sys.stderr)
        sys.exit(1)

    with conn.cursor() as cur:
        if args.no_reset:
            print("옵션 --no-reset: 기존 DB 데이터 유지 후 교차로만 보정합니다.", flush=True)
            ensure_intersections(cur)
        else:
            print("기존 시뮬레이션 데이터 초기화 중 (risk·로그·교차로) …", flush=True)
            reset_simulation_data(cur)
            ensure_intersections(cur)
            print("초기화 완료 — 교차로 메타 재등록됨.", flush=True)
    conn.commit()
    print(f"DB 연결 OK — 틱 주기 {args.tick}s. Ctrl+C 로 중지.\n", flush=True)

    while running:
        t0 = time.time()
        try:
            tick_once(conn, buffers, engine)
        except Exception as e:
            conn.rollback()
            print(f"틱 오류(롤백): {e}", flush=True)
        dt = time.time() - t0
        sleep_for = max(0.1, args.tick - dt)
        while sleep_for > 0 and running:
            time.sleep(min(0.2, sleep_for))
            sleep_for -= 0.2

    conn.close()
    print("연결 종료.", flush=True)


if __name__ == "__main__":
    main()

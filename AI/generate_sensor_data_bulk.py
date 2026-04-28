"""sensor_data.csv에 대량 행을 빠르게 채우기 (sensor_collector와 동일 스키마·분포).

비율·분포는 sensor_data_config.py 와 동기화됩니다.

예:  python generate_sensor_data_bulk.py
     python generate_sensor_data_bulk.py --rows 10000 --output data/sensor_data.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime, timedelta

import numpy as np

from sensor_data_config import (
    COL_AVG_LATENCY_MS,
    COL_COMM_FAIL_COUNT,
    COL_CONNECTED_VEHICLE_COUNT,
    COL_CONTROLLER_DEVICE_ID,
    COL_CONTROLLER_LOG_TIME,
    COL_CPU_TEMP,
    COL_ERROR_COUNT,
    COL_FAILURE_EVENT,
    COL_INTERSECTION_ID,
    COL_REBOOT_COUNT,
    COL_RESPONSE_TIME_MS,
    COL_SEVERITY,
    COL_SPAT_FAIL_COUNT,
    COL_SPAT_SEND_COUNT,
    CSV_COLUMNS as COLUMNS,
    CSV_COLUMNS_WITHOUT_SEVERITY,
    P_SEVERITY_DANGER_END,
    P_SEVERITY_WARN_END,
)

import project_paths as pp

INTERSECTION_IDS = [f"INT_{i:03d}" for i in range(1, 6)]
DEVICE_IDS = [f"CTRL_{i:03d}" for i in range(1, 11)]
SEVERITY_NAMES = ("정상", "주의", "위험")


def _clip(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(x, lo, hi)


def generate_rows(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """예지보전용 시계열 생성.

    핵심: failure_event(고장/위험 이벤트)가 "현재 피처 악화"와 연결되도록 device별 health를 시간에 따라 변화시킨다.
    """
    devs = np.array(DEVICE_IDS, dtype=object)
    ints = np.array(INTERSECTION_IDS, dtype=object)
    n_dev = len(devs)

    # 각 device의 상태
    health = rng.uniform(0.0, 0.25, size=n_dev)  # 0 좋음 → 1 고장 임박
    cooldown = np.zeros(n_dev, dtype=np.int64)

    int_id = np.empty(n, dtype=object)
    dev_id = np.empty(n, dtype=object)
    sev = np.empty(n, dtype=np.int8)
    failure_event = np.zeros(n, dtype=np.int8)

    # 피처 배열
    rt = np.zeros(n, dtype=np.float64)
    ec = np.zeros(n, dtype=np.int64)
    rb = np.zeros(n, dtype=np.int64)
    cpu = np.zeros(n, dtype=np.float64)
    spat_s = np.zeros(n, dtype=np.int64)
    spat_f = np.zeros(n, dtype=np.int64)
    lat = np.zeros(n, dtype=np.float64)
    comm = np.zeros(n, dtype=np.int64)
    veh = np.zeros(n, dtype=np.int64)

    base = datetime.now().replace(microsecond=0)
    sec = np.arange(n, dtype=np.int64) // 10
    ts = np.array(
        [(base + timedelta(seconds=int(s))).strftime("%Y-%m-%d %H:%M:%S") for s in sec],
        dtype=object,
    )

    # 각 행 생성: device를 뽑고, 그 device의 health를 업데이트한 후 피처 샘플링
    dev_idx = rng.integers(0, n_dev, size=n)
    int_idx = rng.integers(0, len(ints), size=n)

    for i in range(n):
        d = int(dev_idx[i])
        int_id[i] = ints[int(int_idx[i])]
        dev_id[i] = devs[d]

        if cooldown[d] > 0:
            cooldown[d] -= 1
            health[d] = max(0.0, health[d] - rng.uniform(0.08, 0.18))
        else:
            health[d] = float(
                np.clip(health[d] + rng.uniform(0.0, 0.02) + rng.normal(0.0, 0.01), 0.0, 1.0)
            )

        # severity by health
        if health[d] >= 0.85:
            sev[i] = 2
        elif health[d] >= 0.55:
            sev[i] = 1
        else:
            sev[i] = 0

        # failure event: 위험 구간에서 일정 확률
        if sev[i] == 2 and cooldown[d] == 0 and rng.random() < 0.12:
            failure_event[i] = 1
            health[d] = rng.uniform(0.05, 0.25)
            cooldown[d] = int(rng.integers(8, 19))

        # 피처 샘플링(health가 높을수록 악화)
        h = float(health[d])
        if sev[i] == 0:
            rt[i] = np.round(_clip(rng.normal(44.0 + 18.0 * h, 10.0, 1), 15.0, 140.0), 1)[0]
            ec[i] = int(rng.integers(0, 3))
            rb[i] = int(rng.integers(0, 2))
            cpu[i] = np.round(_clip(rng.normal(41.5 + 18.0 * h, 3.5, 1), 30.0, 70.0), 1)[0]
            spat_s[i] = int(rng.integers(82, 121))
            spat_f[i] = int(rng.integers(0, 5))
            lat[i] = np.round(
                _clip(rng.normal(22.0 + 70.0 * h, 6.0, 1) + 0.42 * max(0.0, rt[i] - 44.0), 5.0, 120.0),
                1,
            )[0]
            comm[i] = int(rng.integers(0, 4))
            veh[i] = int(rng.integers(8, 46))
        elif sev[i] == 1:
            rt[i] = np.round(_clip(rng.normal(98.0 + 35.0 * h, 17.0, 1), 60.0, 260.0), 1)[0]
            ec[i] = int(rng.integers(3, 13))
            rb[i] = int(rng.integers(0, 3))
            cpu[i] = np.round(_clip(rng.normal(58.5 + 25.0 * h, 5.5, 1), 48.0, 85.0), 1)[0]
            spat_s[i] = int(rng.integers(50, 96))
            spat_f[i] = int(rng.integers(5, 26))
            lat[i] = np.round(
                _clip(rng.normal(56.0 + 90.0 * h, 11.0, 1) + 0.48 * max(0.0, rt[i] - 72.0), 40.0, 220.0),
                1,
            )[0]
            comm[i] = int(min(18, max(4, spat_f[i] // 2 + int(rng.integers(0, 7)))))
            veh[i] = int(rng.integers(15, 61))
        else:
            rt[i] = np.round(_clip(rng.normal(188.0 + 55.0 * h, 38.0, 1), 120.0, 520.0), 1)[0]
            ec[i] = int(rng.integers(15, 46))
            rb[i] = int(rng.integers(2, 9))
            cpu[i] = np.round(_clip(rng.normal(73.0 + 30.0 * h, 7.0, 1), 62.0, 99.0), 1)[0]
            spat_s[i] = int(rng.integers(20, 71))
            spat_f[i] = int(rng.integers(25, 81))
            lat[i] = np.round(
                _clip(rng.normal(108.0 + 140.0 * h, 26.0, 1) + 0.26 * max(0.0, rt[i] - 130.0), 80.0, 420.0),
                1,
            )[0]
            comm[i] = int(min(70, max(20, spat_f[i] // 2 + int(rng.integers(0, 19)))))
            veh[i] = int(rng.integers(25, 81))

    sev_str = np.array([SEVERITY_NAMES[c] for c in sev], dtype=object)

    return {
        COL_CONTROLLER_LOG_TIME: ts,
        COL_INTERSECTION_ID: int_id,
        COL_CONTROLLER_DEVICE_ID: dev_id,
        COL_RESPONSE_TIME_MS: rt,
        COL_ERROR_COUNT: ec,
        COL_REBOOT_COUNT: rb,
        COL_CPU_TEMP: cpu,
        COL_SPAT_SEND_COUNT: spat_s,
        COL_SPAT_FAIL_COUNT: spat_f,
        COL_AVG_LATENCY_MS: lat,
        COL_COMM_FAIL_COUNT: comm,
        COL_CONNECTED_VEHICLE_COUNT: veh,
        COL_SEVERITY: sev_str,
        COL_FAILURE_EVENT: failure_event,
    }


def write_csv(path: str, data: dict[str, np.ndarray], n: int) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows = [
        [
            data[COL_CONTROLLER_LOG_TIME][i],
            data[COL_INTERSECTION_ID][i],
            data[COL_CONTROLLER_DEVICE_ID][i],
            data[COL_RESPONSE_TIME_MS][i],
            int(data[COL_ERROR_COUNT][i]),
            int(data[COL_REBOOT_COUNT][i]),
            data[COL_CPU_TEMP][i],
            int(data[COL_SPAT_SEND_COUNT][i]),
            int(data[COL_SPAT_FAIL_COUNT][i]),
            data[COL_AVG_LATENCY_MS][i],
            int(data[COL_COMM_FAIL_COUNT][i]),
            int(data[COL_CONNECTED_VEHICLE_COUNT][i]),
            data[COL_SEVERITY][i],
            int(data[COL_FAILURE_EVENT][i]),
        ]
        for i in range(n)
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(rows)


def write_csv_without_severity(path: str, data: dict[str, np.ndarray], n: int) -> None:
    """학습용과 동일 분포로 생성하되 severity 컬럼은 저장하지 않음 (라벨 없는 테스트 데이터)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows = [
        [
            data[COL_CONTROLLER_LOG_TIME][i],
            data[COL_INTERSECTION_ID][i],
            data[COL_CONTROLLER_DEVICE_ID][i],
            data[COL_RESPONSE_TIME_MS][i],
            int(data[COL_ERROR_COUNT][i]),
            int(data[COL_REBOOT_COUNT][i]),
            data[COL_CPU_TEMP][i],
            int(data[COL_SPAT_SEND_COUNT][i]),
            int(data[COL_SPAT_FAIL_COUNT][i]),
            data[COL_AVG_LATENCY_MS][i],
            int(data[COL_COMM_FAIL_COUNT][i]),
            int(data[COL_CONNECTED_VEHICLE_COUNT][i]),
            int(data[COL_FAILURE_EVENT][i]),
        ]
        for i in range(n)
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS_WITHOUT_SEVERITY)
        w.writerows(rows)


def main() -> None:
    pp.ensure_dirs()
    p = argparse.ArgumentParser(description="sensor_data.csv 대량 생성")
    p.add_argument("--rows", type=int, default=10_000, help="행 개수 (기본 10000)")
    p.add_argument(
        "--output",
        "-o",
        default=str(pp.DATA_CSV),
        help="저장 경로 (기본 data/sensor_data.csv)",
    )
    p.add_argument("--seed", type=int, default=None, help="재현용 난수 시드")
    p.add_argument(
        "--stream",
        action="store_true",
        help="한 행씩 실시간에 가깝게 간격을 두고 append 저장",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=0.1,
        help="--stream 일 때 행 간 대기(초)",
    )
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.stream:
        path = args.output
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        new_file = not os.path.isfile(path)
        t0 = time.perf_counter()
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(COLUMNS)
            for i in range(args.rows):
                row_block = generate_rows(1, rng)
                w.writerow(
                    [
                        row_block[COL_CONTROLLER_LOG_TIME][0],
                        row_block[COL_INTERSECTION_ID][0],
                        row_block[COL_CONTROLLER_DEVICE_ID][0],
                        row_block[COL_RESPONSE_TIME_MS][0],
                        int(row_block[COL_ERROR_COUNT][0]),
                        int(row_block[COL_REBOOT_COUNT][0]),
                        row_block[COL_CPU_TEMP][0],
                        int(row_block[COL_SPAT_SEND_COUNT][0]),
                        int(row_block[COL_SPAT_FAIL_COUNT][0]),
                        row_block[COL_AVG_LATENCY_MS][0],
                        int(row_block[COL_COMM_FAIL_COUNT][0]),
                        int(row_block[COL_CONNECTED_VEHICLE_COUNT][0]),
                        row_block[COL_SEVERITY][0],
                        int(row_block[COL_FAILURE_EVENT][0]),
                    ]
                )
                f.flush()
                if (i + 1) % 25 == 0:
                    print(f"[stream] {i+1}/{args.rows} 행 기록 → {path}")
                time.sleep(max(0.0, args.interval))
        elapsed = time.perf_counter() - t0
        print(
            f"스트리밍 저장 완료: {path} ({args.rows}행, {elapsed:.1f}s)"
        )
        return

    t0 = time.perf_counter()
    data = generate_rows(args.rows, rng)
    write_csv(args.output, data, args.rows)
    elapsed = time.perf_counter() - t0
    print(
        f"저장 완료: {args.output} ({args.rows}행, {elapsed*1000:.1f}ms)  "
        f"[severity 비율: sensor_data_config.py 참고]"
    )


if __name__ == "__main__":
    main()

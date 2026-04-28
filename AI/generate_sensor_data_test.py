"""테스트용 센서 CSV 생성.

기본: 학습 CSV와 동일 스키마(severity 포함) → 행/윈도우 단위 실제 이상 개수·detect 시 오탐률 산출 가능.
옵션: --without-labels 로 severity 없는 파일만 필요할 때.

예:  python generate_sensor_data_test.py
     python generate_sensor_data_test.py --rows 5000 --seed 99 -o data/sensor_test.csv
     python generate_sensor_data_test.py --without-labels
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import project_paths as pp
from generate_sensor_data_bulk import generate_rows, write_csv, write_csv_without_severity
from sensor_data_config import COL_SEVERITY


def main() -> None:
    pp.ensure_dirs()
    p = argparse.ArgumentParser(
        description="테스트용 sensor CSV (기본: severity 포함, 평가용)",
    )
    p.add_argument("--rows", type=int, default=5_000, help="행 개수")
    p.add_argument(
        "-o",
        "--output",
        default=str(pp.DATA_CSV_TEST),
        help=f"저장 경로 (기본 {pp.DATA_CSV_TEST})",
    )
    p.add_argument("--seed", type=int, default=None, help="재현용 난수 시드 (학습용과 다르게)")
    p.add_argument(
        "--without-labels",
        action="store_true",
        help="severity 열 없이 저장 (추론만, 오탐률 등 평가 불가)",
    )
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()
    data = generate_rows(args.rows, rng)
    if args.without_labels:
        write_csv_without_severity(args.output, data, args.rows)
        msg = "severity 제외"
    else:
        write_csv(args.output, data, args.rows)
        n_danger = int((data[COL_SEVERITY] == "위험").sum())
        msg = f"severity 포함 (행 기준 실제 위험 {n_danger}행 / 전체 {args.rows}행)"
    elapsed = time.perf_counter() - t0
    print(f"테스트 데이터 저장: {args.output} ({args.rows}행, {elapsed*1000:.1f}ms) — {msg}")


if __name__ == "__main__":
    main()

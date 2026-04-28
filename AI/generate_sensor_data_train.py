"""학습용 센서 CSV 생성 (severity 포함).

기본: data/sensor_train.csv
별도 시드로 테스트용과 분리해 생성하세요 (예: --seed 42).

예:  python generate_sensor_data_train.py
     python generate_sensor_data_train.py --rows 20000 --seed 42 -o data/sensor_train.csv
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import project_paths as pp
from generate_sensor_data_bulk import generate_rows, write_csv


def main() -> None:
    pp.ensure_dirs()
    p = argparse.ArgumentParser(description="학습용 sensor CSV (severity 포함)")
    p.add_argument("--rows", type=int, default=10_000, help="행 개수")
    p.add_argument(
        "-o",
        "--output",
        default=str(pp.DATA_CSV_TRAIN),
        help=f"저장 경로 (기본 {pp.DATA_CSV_TRAIN})",
    )
    p.add_argument("--seed", type=int, default=None, help="재현용 난수 시드")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()
    data = generate_rows(args.rows, rng)
    write_csv(args.output, data, args.rows)
    elapsed = time.perf_counter() - t0
    print(
        f"학습 데이터 저장: {args.output} ({args.rows}행, {elapsed*1000:.1f}ms) — severity 포함"
    )


if __name__ == "__main__":
    main()

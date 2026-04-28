"""sensor_data.csv 에 10만 행 생성 (generate_sensor_data_bulk.py 와 동일 스키마·분포).

기본 100,000행. 출력·시드만 바꿔 쓰기 쉽게 분리한 진입점입니다.

예:
  python py/generate_sensor_data_100k.py
  python py/generate_sensor_data_100k.py -o data/sensor_data_100k.csv --seed 42
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import project_paths as pp
from generate_sensor_data_bulk import generate_rows, write_csv

DEFAULT_ROWS = 100_000


def main() -> None:
    pp.ensure_dirs()
    p = argparse.ArgumentParser(description="sensor_data 대량 생성 (기본 10만 행)")
    p.add_argument(
        "--rows",
        type=int,
        default=DEFAULT_ROWS,
        help=f"행 개수 (기본 {DEFAULT_ROWS})",
    )
    p.add_argument(
        "--output",
        "-o",
        default=str(pp.DATA_CSV),
        help="저장 경로 (기본 data/sensor_data.csv)",
    )
    p.add_argument("--seed", type=int, default=None, help="재현용 난수 시드")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
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

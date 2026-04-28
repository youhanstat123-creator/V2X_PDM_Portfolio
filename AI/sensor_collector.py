"""교차로·신호제어기(V2X/SPaT) 예지보전용 가상 데이터 수집기.

실시간 스트림·서버 데모는 `realtime_server.py`(가상 생성 루프) 또는
`generate_sensor_data_bulk.py --stream` 을 사용할 수 있습니다.

측정시각은 1초 단위 문자열로 기록하고, 실제 저장 주기는 INTERVAL(초)만큼 대기합니다.
(예: INTERVAL=0.1 이면 같은 초에 최대 10행이 쌓일 수 있음)

수집은 백그라운드 스레드에서 동작하므로, start 후에도 같은 터미널에서 stop / quit 입력 가능합니다.

severity 비율·분포는 sensor_data_config.py 와 동기화됩니다.
"""

import csv
import os
import random
import threading
import time
from datetime import datetime, timedelta

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
    severity_from_uniform,
)

import project_paths as pp

INTERSECTION_IDS = [f"INT_{i:03d}" for i in range(1, 6)]
DEVICE_IDS = [f"CTRL_{i:03d}" for i in range(1, 11)]

CSV_FILE = str(pp.DATA_CSV)
INTERVAL = 0.1  # 초 (실제 생성·저장 간격)

# 1초 단위 타임스탬프 유지: 10번(0.1초×10)마다 1초 증가
_logical_time: datetime | None = None
_samples_in_second = 0
SAMPLES_PER_LOGICAL_SECOND = 10

_ts_lock = threading.Lock()


def _next_controller_log_time_str() -> str:
    global _logical_time, _samples_in_second
    with _ts_lock:
        if _logical_time is None:
            _logical_time = datetime.now().replace(microsecond=0)
        if _samples_in_second >= SAMPLES_PER_LOGICAL_SECOND:
            _logical_time += timedelta(seconds=1)
            _samples_in_second = 0
        _samples_in_second += 1
        return _logical_time.strftime("%Y-%m-%d %H:%M:%S")


def generate_sensor_data(intersection_id: str, controller_device_id: str) -> dict:
    # ── 장비 상태(health) 기반 시뮬레이션 ─────────────────────────────
    # 진짜 예지보전을 위해 "미래 고장 이벤트"가 현재 피처의 악화 추세와 연결되도록 생성한다.
    st = _DEVICE_STATE.setdefault(
        controller_device_id,
        {
            "health": random.uniform(0.0, 0.25),  # 0(좋음) → 1(고장 임박)
            "cooldown": 0,  # 고장 직후 몇 샘플은 회복 상태 유지
        },
    )
    health = float(st["health"])
    cooldown = int(st["cooldown"])

    if cooldown > 0:
        cooldown -= 1
        # 고장 후 재기동/정비로 상태 회복
        health = max(0.0, health - random.uniform(0.08, 0.18))
    else:
        # 시간이 지날수록 조금씩 악화 + 잡음
        health = min(1.0, max(0.0, health + random.uniform(0.0, 0.02) + random.gauss(0.0, 0.01)))

    # 상태→severity (분포 기반이 아니라 health 기반)
    if health >= 0.85:
        severity = "위험"
    elif health >= 0.55:
        severity = "주의"
    else:
        severity = "정상"

    # failure_event: 위험 상태에서 일정 확률로 이벤트 발생(발생 후 회복)
    failure_event = 0
    if severity == "위험" and cooldown == 0 and random.random() < 0.12:
        failure_event = 1
        # 이벤트 발생 후 정비/재기동으로 상태 리셋
        health = random.uniform(0.05, 0.25)
        cooldown = random.randint(8, 18)

    st["health"] = health
    st["cooldown"] = cooldown

    if severity == "정상":
        # 분포 약간 타이트 → 클래스 간 분리에 유리
        response_time_ms = round(random.gauss(44.0, 10.0), 1)
        response_time_ms = max(15.0, min(120.0, response_time_ms))
        error_count = random.randint(0, 2)
        reboot_count = random.randint(0, 1)
        cpu_temp = round(random.gauss(41.5, 3.5), 1)
        cpu_temp = max(30.0, min(65.0, cpu_temp))
        spat_send_count = random.randint(82, 120)
        spat_fail_count = random.randint(0, 4)
        # 응답 지연과 평균 지연 상관(부하가 올라가면 같이 상승)
        avg_latency_ms = round(
            random.gauss(22.0, 6.0) + 0.42 * max(0.0, response_time_ms - 44.0),
            1,
        )
        avg_latency_ms = max(5.0, min(82.0, avg_latency_ms))
        comm_fail_count = random.randint(0, 3)
        connected_vehicle_count = random.randint(8, 45)

    elif severity == "주의":
        response_time_ms = round(random.gauss(98.0, 17.0), 1)
        response_time_ms = max(60.0, min(220.0, response_time_ms))
        error_count = random.randint(3, 12)
        reboot_count = random.randint(0, 2)
        cpu_temp = round(random.gauss(58.5, 5.5), 1)
        cpu_temp = max(48.0, min(78.0, cpu_temp))
        spat_send_count = random.randint(50, 95)
        spat_fail_count = random.randint(5, 25)
        avg_latency_ms = round(
            random.gauss(56.0, 11.0) + 0.48 * max(0.0, response_time_ms - 72.0),
            1,
        )
        avg_latency_ms = max(40.0, min(152.0, avg_latency_ms))
        # SPaT 실패와 통신 실패 동반(약한 결합)
        comm_fail_count = min(
            18,
            max(4, spat_fail_count // 2 + random.randint(0, 6)),
        )
        connected_vehicle_count = random.randint(15, 60)

    else:
        response_time_ms = round(random.gauss(188.0, 38.0), 1)
        response_time_ms = max(120.0, min(450.0, response_time_ms))
        error_count = random.randint(15, 45)
        reboot_count = random.randint(2, 8)
        cpu_temp = round(random.gauss(73.0, 7.0), 1)
        cpu_temp = max(62.0, min(95.0, cpu_temp))
        spat_send_count = random.randint(20, 70)
        spat_fail_count = random.randint(25, 80)
        avg_latency_ms = round(
            random.gauss(108.0, 26.0) + 0.26 * max(0.0, response_time_ms - 130.0),
            1,
        )
        avg_latency_ms = max(80.0, min(350.0, avg_latency_ms))
        comm_fail_count = min(
            70,
            max(20, spat_fail_count // 2 + random.randint(0, 18)),
        )
        connected_vehicle_count = random.randint(25, 80)

    return {
        COL_CONTROLLER_LOG_TIME: _next_controller_log_time_str(),
        COL_INTERSECTION_ID: intersection_id,
        COL_CONTROLLER_DEVICE_ID: controller_device_id,
        COL_RESPONSE_TIME_MS: response_time_ms,
        COL_ERROR_COUNT: error_count,
        COL_REBOOT_COUNT: reboot_count,
        COL_CPU_TEMP: cpu_temp,
        COL_SPAT_SEND_COUNT: spat_send_count,
        COL_SPAT_FAIL_COUNT: spat_fail_count,
        COL_AVG_LATENCY_MS: avg_latency_ms,
        COL_COMM_FAIL_COUNT: comm_fail_count,
        COL_CONNECTED_VEHICLE_COUNT: connected_vehicle_count,
        COL_SEVERITY: severity,
        COL_FAILURE_EVENT: failure_event,
    }


# device별 상태 저장(프로세스 실행 중 유지)
_DEVICE_STATE: dict[str, dict[str, float | int]] = {}


def save_to_csv(data: dict) -> None:
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)


def print_data(data: dict, count: int) -> None:
    color = {"정상": "\033[92m", "주의": "\033[93m", "위험": "\033[91m"}
    reset = "\033[0m"
    c = color.get(data[COL_SEVERITY], "")
    print(
        f"  [{count:>4}] {c}[{data[COL_SEVERITY]}]{reset}  "
        f"{data[COL_INTERSECTION_ID]} / {data[COL_CONTROLLER_DEVICE_ID]}  "
        f"응답:{data[COL_RESPONSE_TIME_MS]}ms  "
        f"지연:{data[COL_AVG_LATENCY_MS]}ms  "
        f"오류:{data[COL_ERROR_COUNT]}  "
        f"→ CSV 저장",
        flush=True,
    )


def _collect_loop(stop_event: threading.Event, state: dict[str, int]) -> None:
    while not stop_event.is_set():
        # 짧게 나눠 자면 stop 반응이 조금 빨라짐
        slept = 0.0
        while slept < INTERVAL and not stop_event.is_set():
            chunk = min(0.05, INTERVAL - slept)
            time.sleep(chunk)
            slept += chunk
        if stop_event.is_set():
            break
        intersection_id = random.choice(INTERSECTION_IDS)
        controller_device_id = random.choice(DEVICE_IDS)
        data = generate_sensor_data(intersection_id, controller_device_id)
        save_to_csv(data)
        state["count"] += 1
        print_data(data, state["count"])


def main() -> None:
    pp.ensure_dirs()
    print("\n교차로·제어기 예지보전 데이터 수집기")
    print("=" * 50)
    print("  명령어:  start  → 수집 시작 (백그라운드)")
    print("           stop   → 수집 중지")
    print("           quit   → 프로그램 종료")
    print(f"  저장 파일: {CSV_FILE}")
    print(f"  저장 간격: {INTERVAL}초  |  측정시각(표시): 1초 단위, 초당 최대 {SAMPLES_PER_LOGICAL_SECOND}행")
    print("=" * 50)

    stop_event = threading.Event()
    collector_thread: threading.Thread | None = None
    state: dict[str, int] = {"count": 0}

    while True:
        cmd = input("\n명령어 입력 > ").strip().lower()

        if cmd == "start":
            if collector_thread is not None and collector_thread.is_alive():
                print("  이미 수집 중입니다. 중지하려면 stop")
                continue
            stop_event.clear()
            state["count"] = 0
            collector_thread = threading.Thread(
                target=_collect_loop,
                args=(stop_event, state),
                daemon=True,
            )
            collector_thread.start()
            print(f"\n▶ 수집 시작 ({INTERVAL}초마다 저장). 중지: stop 입력")

        elif cmd == "stop":
            if collector_thread is None or not collector_thread.is_alive():
                print("  수집 중이 아닙니다.")
                continue
            stop_event.set()
            collector_thread.join(timeout=30.0)
            collector_thread = None
            print(f"\n⏹ 수집 중지 — 총 {state['count']}건 저장 ({CSV_FILE})")

        elif cmd == "quit":
            stop_event.set()
            if collector_thread is not None and collector_thread.is_alive():
                collector_thread.join(timeout=10.0)
            print("\n종료.")
            break

        else:
            print("  알 수 없는 명령입니다. start / stop / quit")


if __name__ == "__main__":
    main()

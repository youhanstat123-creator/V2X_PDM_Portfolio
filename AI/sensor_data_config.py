"""가상 센서 데이터 공통 설정.

- sensor_collector.py
- generate_sensor_data_bulk.py
- test1.py 등

비율·분포를 바꿀 때는 이 파일만 수정하면 두 쪽이 같이 맞춰집니다.

스키마(영문 식별자)는 스프링 DTO·API와 동일하게 맞춥니다.
한글 표시명은 HEADER_LABEL_KO 참고.
"""

from __future__ import annotations

# ── API/CSV/파이썬 공통 영문 필드명 ─────────────────────────────────
COL_CONTROLLER_LOG_TIME = "controller_log_time"
COL_INTERSECTION_ID = "intersection_id"
COL_CONTROLLER_DEVICE_ID = "controller_device_id"
COL_RESPONSE_TIME_MS = "response_time_ms"
COL_ERROR_COUNT = "error_count"
COL_REBOOT_COUNT = "reboot_count"
COL_CPU_TEMP = "cpu_temp"
COL_SPAT_SEND_COUNT = "spat_send_count"
COL_SPAT_FAIL_COUNT = "spat_fail_count"
COL_AVG_LATENCY_MS = "avg_latency_ms"
COL_COMM_FAIL_COUNT = "comm_fail_count"
COL_CONNECTED_VEHICLE_COUNT = "connected_vehicle_count"
COL_SEVERITY = "severity"
COL_FAILURE_EVENT = "failure_event"

# CSV 한 행 컬럼 순서
CSV_COLUMNS: tuple[str, ...] = (
    COL_CONTROLLER_LOG_TIME,
    COL_INTERSECTION_ID,
    COL_CONTROLLER_DEVICE_ID,
    COL_RESPONSE_TIME_MS,
    COL_ERROR_COUNT,
    COL_REBOOT_COUNT,
    COL_CPU_TEMP,
    COL_SPAT_SEND_COUNT,
    COL_SPAT_FAIL_COUNT,
    COL_AVG_LATENCY_MS,
    COL_COMM_FAIL_COUNT,
    COL_CONNECTED_VEHICLE_COUNT,
    COL_SEVERITY,
    COL_FAILURE_EVENT,
)

# 테스트·추론용 CSV (severity 없음 — 라벨 없이 9개 피처 + 메타만)
CSV_COLUMNS_WITHOUT_SEVERITY: tuple[str, ...] = tuple(
    c for c in CSV_COLUMNS if c != COL_SEVERITY
)

# 한글 표시명 (문서·UI·스프링 표 헤더)
HEADER_LABEL_KO: dict[str, str] = {
    COL_CONTROLLER_LOG_TIME: "측정시각",
    COL_INTERSECTION_ID: "교차로ID",
    COL_CONTROLLER_DEVICE_ID: "제어기ID",
    COL_RESPONSE_TIME_MS: "제어기응답시간(ms)",
    COL_ERROR_COUNT: "오류발생횟수",
    COL_REBOOT_COUNT: "재부팅횟수",
    COL_CPU_TEMP: "제어기온도(℃)",
    COL_SPAT_SEND_COUNT: "SPaT전송횟수",
    COL_SPAT_FAIL_COUNT: "SPaT전송실패횟수",
    COL_AVG_LATENCY_MS: "평균통신지연(ms)",
    COL_COMM_FAIL_COUNT: "통신실패횟수",
    COL_CONNECTED_VEHICLE_COUNT: "접속차량수",
    COL_SEVERITY: "심각도",
    COL_FAILURE_EVENT: "고장이벤트",
}

# 모델 입력 수치 피처 (순서 고정)
FEATURE_COLS: tuple[str, ...] = (
    COL_RESPONSE_TIME_MS,
    COL_ERROR_COUNT,
    COL_REBOOT_COUNT,
    COL_CPU_TEMP,
    COL_SPAT_SEND_COUNT,
    COL_SPAT_FAIL_COUNT,
    COL_AVG_LATENCY_MS,
    COL_COMM_FAIL_COUNT,
    COL_CONNECTED_VEHICLE_COUNT,
)

# 예지보전 임계값·튜닝 목표 (test1.py·탐지 스크립트와 동기화)
# PM_USE_FIXED_THRESHOLD=False 이면 검증(val) 점수로 임계값을 튜닝 (PM_TARGET_* 목표 우선).
# True 이면 DEFAULT_PM_THRESHOLD만 사용(고정).
DEFAULT_PM_THRESHOLD = 0.9
PM_TARGET_PRECISION = 0.8
PM_TARGET_RECALL = 0.8
PM_TARGET_FPR_MAX = 0.0
PM_USE_FIXED_THRESHOLD = False

# 예지보전 라벨: 앞으로 이 횟수(행/윈도우) 안에 failure_event·위험 시점이 있으면 y=1 (test1·detect 공통)
PREDICT_HORIZON_WINDOWS = 6

# pm_lstm 학습: 양성 윈도우 오버샘플링 목표 비율(학습 집합에서 대략 이 비율까지 양성 비중 확대)
PM_MAX_POS_RATIO = 0.22

# 검증(val) 임계값 튜닝: 목표 미달 시 페널티 가중 (동시에 최소화, 테스트 라벨 미사용)
# FPR 목표가 0이면 FP 개수에 PM_THRESHOLD_FP_WEIGHT 적용
PM_THRESHOLD_W_PRECISION = 1.0
PM_THRESHOLD_W_RECALL = 1.0
PM_THRESHOLD_W_FPR = 120.0
PM_THRESHOLD_FP_WEIGHT = 12.0

# LSTM 분류기: Focal loss (BinaryFocalCrossentropy 사용 시)
PM_FOCAL_GAMMA = 2.0
PM_FOCAL_ALPHA = 0.38

# LSTM 구조 (Conv1D → LSTM×3 → Dense 헤드) — test1.build_pm_lstm_classifier 가 참조
PM_LSTM_CONV_FILTERS_TEACHER = 72
PM_LSTM_CONV_FILTERS_STUDENT = 56
PM_LSTM_TEACHER_LSTM_UNITS: tuple[int, int, int] = (112, 72, 48)
PM_LSTM_STUDENT_LSTM_UNITS: tuple[int, int, int] = (80, 56, 36)
PM_LSTM_TEACHER_DENSE: tuple[int, int] = (128, 56)
PM_LSTM_STUDENT_DENSE: tuple[int, int] = (96, 40)

# ── severity 비율 (uniform u ∈ [0,1) 기준 누적 임계값) ─────────────────
P_SEVERITY_DANGER_END = 0.05
P_SEVERITY_WARN_END = 0.20


def severity_from_uniform(u: float) -> str:
    """단일 난수 u(0~1)로 severity 문자열."""
    if u < P_SEVERITY_DANGER_END:
        return "위험"
    if u < P_SEVERITY_WARN_END:
        return "주의"
    return "정상"


def severity_code_from_uniform(u: float) -> int:
    """벌크 생성용: 0=정상, 1=주의, 2=위험."""
    if u < P_SEVERITY_DANGER_END:
        return 2
    if u < P_SEVERITY_WARN_END:
        return 1
    return 0

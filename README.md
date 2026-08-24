# V17 IMU 기반 선박 운동 예측 및 인터셉터 제어

6축 IMU 시계열로 미래의 수직 specific force trajectory를 예측하고,
예측 결과를 MSI 추정과 인터셉터 제어에 사용하는 프로젝트입니다.

현재 기본 목표는 다음과 같습니다.

| 항목 | 설정 |
|---|---:|
| IMU sampling rate | 100 Hz |
| 입력 history | 과거 30초 |
| 입력 shape | `(3000, 6)` |
| 기본 prediction horizon | 미래 1초 |
| 출력 shape | `(100,)` |
| target | `true_vertical_specific_force_mps2` |
| 기본 모델 | Conv-LSTM |
| 기본 checkpoint | `model/checkpoints/best_1s.pt` |

1초가 코드에 강제로 고정된 것은 아닙니다. 실시간 시스템은 선택한
checkpoint의 `DataConfig`에서 history와 prediction horizon을 읽어
자동으로 buffer와 출력 길이를 구성합니다.

## 처리 흐름

```text
MPU6050 6축 IMU (100 Hz)
          │
          ├── raw IMU 30초 buffer
          │          │
          │          └── Conv-LSTM 비동기 추론
          │                     │
          │                     ├── raw future trajectory
          │                     │        └── 동일 상태의 ISO filter 복제 → MSI
          │                     │
          │                     └── DC 제거 → LightMPC → actuator
          │
          └── 현재 Z축 가속도 → 실시간 ISO filter → MSI
```

- Predictor 입력은 raw 6축 IMU입니다.
- accelerometer 입력 3축만 ±16g(`±156.96 m/s²`)로 clipping한 뒤
  train 통계로 normalization합니다.
- target은 clipping하지 않습니다.
- MSI용 미래 trajectory는 live filter 상태를 복제한 필터로 처리하므로
  실시간 필터 상태가 변경되지 않습니다.
- MPC에는 median 기준으로 중력/DC를 제거한 동적 trajectory를 전달합니다.
- 추론은 단일 background worker에서 실행되어 100 Hz sensor loop를
  직접 block하지 않습니다.

## 개발 환경 설치

GTX 1050 개발 환경은 제공된 Conda 파일로 생성할 수 있습니다.

```bash
cd /home/jeongbyeolsol/ship_design
conda env create -f environment_gtx1050.yaml
conda activate msi-predictor-gtx1050
```

이미 환경이 만들어져 있다면 `conda activate`만 실행하면 됩니다.

## Dataset

기본 데이터 디렉터리는 다음 구조를 기대합니다.

```text
V17_Synthetic_IMU_Dataset/
├── train.csv 또는 train.parquet
├── validation.csv 또는 validation.parquet
└── test.csv 또는 test.parquet
```

필수 컬럼:

```text
scenario_id
time
imu_acc_x_mps2
imu_acc_y_mps2
imu_acc_z_mps2
imu_gyro_x_rad_s
imu_gyro_y_rad_s
imu_gyro_z_rad_s
true_vertical_specific_force_mps2
```

Dataset loader는 필요한 컬럼만 읽으며, scenario별 timestamp를 stable
sort합니다. 중복되거나 증가하지 않는 timestamp는 즉시 거부하고,
sliding window는 scenario 경계를 넘지 않습니다.

## 빠른 검증

### pytest

```bash
./exe/run_tests.sh -q
```

### 학습 smoke test

```bash
./exe/smoke/train_smoke_test.sh V17_Synthetic_IMU_Dataset
```

### 데스크톱에서 checkpoint 추론 확인

실제 IMU가 없는 PC 또는 WSL에서는 `run_system.sh` 대신 다음 명령을
사용합니다.

```bash
DEVICE=cuda ./exe/smoke/inference_smoke_test.sh \
    V17_Synthetic_IMU_Dataset \
    model/checkpoints/best_1s.pt
```

이 smoke test는 실제 I²C, GPS, pigpio 장치를 요구하지 않습니다.

## 모델 학습

기본 설정인 30초 입력 → 1초 출력 Conv-LSTM:

```bash
./exe/train_model.sh V17_Synthetic_IMU_Dataset lstm
```

명시적으로 실행하려면:

```bash
HISTORY_SECONDS=30 \
PREDICTION_SECONDS=1 \
EPOCHS=80 \
BATCH_SIZE=16 \
NUM_WORKERS=4 \
LR=1e-3 \
CHECKPOINT_NAME=best_1s.pt \
RUN_NAME=lstm_30to1 \
./exe/train_model.sh V17_Synthetic_IMU_Dataset lstm
```

학습 로그는 `logs/`, best checkpoint는 기본적으로
`model/checkpoints/`에 저장됩니다. validation loss가 개선되지 않으면
early stopping되며 실제 best epoch의 checkpoint만 유지합니다.

### 다른 horizon 학습

시간 설정은 학습 시 자유롭게 지정할 수 있습니다.

```bash
# 30초 → 3초
HISTORY_SECONDS=30 \
PREDICTION_SECONDS=3 \
CHECKPOINT_NAME=best_3s.pt \
RUN_NAME=lstm_30to3 \
./exe/train_model.sh V17_Synthetic_IMU_Dataset lstm
```

출력 길이는 checkpoint의 모델 구조에 포함됩니다. 따라서 1초 checkpoint를
실행하면서 출력만 3초로 늘릴 수는 없고, 해당 horizon으로 학습된
checkpoint를 선택해야 합니다.

## Horizon 실험 결과

세 모델을 15초 모델까지 target이 존재하는 동일한 validation/test
window에서 비교한 결과입니다.

| 모델 | 평가 구간 | Validation MSE | Test MSE | Test baseline skill |
|---|---|---:|---:|---:|
| 1초 | 0–1초 | 0.5792 | 0.5834 | +38.18% |
| 3초 | 0–1초 | 0.5887 | 0.5791 | +38.63% |
| 3초 | 1–3초 | 0.9028 | 0.9079 | +3.01% |
| 15초 | 0–1초 | 0.6322 | 0.6302 | +33.22% |
| 15초 | 3–5초 | 0.9248 | 0.9474 | +0.03% |
| 15초 | 5–15초 | 약 0.927 | 약 0.982 | 약 0% |

3초 모델의 세부 test 결과에서는 0–0.5초 skill이 약 55.6%,
0.5–1초가 약 20.7%, 1–1.5초가 약 9.8%였으며 1.5초 이후에는
baseline 대비 이점이 거의 사라졌습니다. 이 결과에 따라 프로젝트의
기본 prediction horizon을 1초로 선택했습니다.

다시 비교하려면:

```bash
./exe/compare_horizons.sh validation \
    model/checkpoints/best_1s.pt \
    model/checkpoints/best_3s.pt \
    model/checkpoints/best_15s.pt

./exe/compare_horizons.sh test \
    model/checkpoints/best_1s.pt \
    model/checkpoints/best_3s.pt \
    model/checkpoints/best_15s.pt
```

비교 결과에는 normalized MSE, physical MSE, 평균 예측 baseline 대비
skill이 함께 출력됩니다.

## 파일 기반 단일 추론

`.npy` 또는 IMU 컬럼을 포함한 `.csv` window를 입력으로 사용할 수
있습니다.

```bash
DEVICE=cuda ./exe/run_inference.sh \
    model/checkpoints/best_1s.pt \
    sample_window.npy \
    outputs/prediction.npy
```

입력 shape는 checkpoint가 요구하는 정확한 `(T, 6)`, 출력 shape는
`(H,)`이어야 합니다.

## 실제 하드웨어 실행

> `run_system.sh`는 실제 Raspberry Pi 하드웨어용입니다. 현재 프로젝트에는
> 데스크톱용 실시간 simulation mode가 없습니다.

필요 장치:

- Raspberry Pi의 활성화된 I²C bus `/dev/i2c-1`
- 주소 `0x68`의 MPU6050
- `/dev/ttyUSB0` GPS
- 실행 중인 `pigpiod`
- GPIO 18에 연결된 actuator

Raspberry Pi에서 I²C 연결을 먼저 확인합니다.

```bash
sudo raspi-config
# Interface Options → I2C → Enable

sudo apt install i2c-tools
i2cdetect -y 1
```

`i2cdetect` 결과에 일반적으로 `68` 주소가 보여야 합니다.

기본 1초 모델 실행:

```bash
./exe/run_system.sh model/checkpoints/best_1s.pt
```

다른 checkpoint도 코드 수정 없이 선택할 수 있습니다.

```bash
./exe/run_system.sh model/checkpoints/best_3s.pt
./exe/run_system.sh model/checkpoints/best_15s.pt
```

PC/WSL에서 다음 오류가 발생하는 것은 모델 문제가 아니라 실제 I²C
장치가 없기 때문입니다.

```text
[Error] IMU 초기화 실패:
No such file or directory: '/dev/i2c-1'
```

종료는 `Ctrl+C`를 사용합니다. 종료 또는 초기화 실패 시 actuator
`emergency_stop`을 우선 실행하고 생성된 IMU, GPS, inference worker를
안전하게 정리합니다.

## Checkpoint 계약

Checkpoint에는 다음 정보가 함께 저장됩니다.

- `DataConfig`: sample rate, history, horizon, 입력 컬럼, clipping 범위
- `ModelConfig`: 모델 종류와 layer/output 구조
- train input/target normalization 통계
- model/optimizer state와 best epoch metric

시스템 시작 시 checkpoint와 runtime의 sample rate, 입력 길이, 채널 수,
출력 길이를 확인합니다. 입력 clipping과 normalization도
train/validation/test/inference에서 동일하게 적용됩니다.

## 프로젝트 구조

```text
.
├── main.py                 # 실시간 시스템 orchestration
├── predictor.py            # 공개 Predictor 및 비동기 worker
├── imu.py                  # MPU6050 6축 입력
├── gps.py                  # GPS speed 입력
├── filter.py               # 실시간/미래 ISO 필터
├── msi.py                  # MSI/MSDV 계산
├── mpc.py                  # 경량 제어 계산
├── controller.py           # pigpio actuator 제어
├── model/
│   ├── config.py
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── network.py
│   ├── train.py
│   ├── inference.py
│   └── evaluate_horizons.py
├── exe/                    # 실행 및 smoke-test shell scripts
├── tests/                  # pytest 회귀 테스트
└── environment_gtx1050.yaml
```

각 shell script의 세부 환경변수와 사용법은
[exe 실행 가이드](exe/README.md)를 참고하세요.

## 현재 한계와 안전 주의

이 코드는 연구·프로토타입 단계입니다. 실제 선박 제어에 투입하기 전에
다음 항목을 반드시 검증해야 합니다.

- IMU/GPS loss 및 stale-data watchdog
- hardware mode에서 actuator fail-closed 동작
- IMU 축 방향과 선박 좌표계 mapping
- acceleration–stroke 계수와 actuator 응답 지연 실측
- stroke 변화율 제한과 제어 부호
- MSI/MSDV 근사식의 적용 기준 검증
- 실제 장비에서 100 Hz loop와 inference latency 측정

특히 `LightMPC`는 현재 고정 양력 계수를 사용하는 경량 해석식 제어기이며,
실선 시험 전 계수 식별과 hardware-in-the-loop 검증이 필요합니다.

# 실행 스크립트 사용 가이드

이 디렉터리의 스크립트는 어느 위치에서 호출해도 프로젝트 루트를
자동으로 찾아 실행합니다. 아래 예시는 프로젝트 루트에서 실행하는
기준입니다.

## 실행 전 준비

프로젝트용 Conda 환경을 생성하고 활성화합니다.

```bash
conda env create -f environment_gtx1050.yaml
conda activate msi-predictor-gtx1050
```

다른 Python을 사용하려면 모든 스크립트에 `PYTHON_BIN` 환경 변수를
지정할 수 있습니다.

```bash
PYTHON_BIN=/path/to/python ./exe/run_tests.sh
```

## 스크립트 요약

| 스크립트 | 용도 |
|---|---|
| `train_model.sh` | 전체 train/validation 데이터로 모델 학습 |
| `run_inference.sh` | 체크포인트와 IMU window 파일로 단일 추론 |
| `run_system.sh` | 실제 IMU/GPS/actuator를 사용하는 실시간 시스템 실행 |
| `run_tests.sh` | 빠른 pytest 회귀 테스트 실행 |
| `smoke/train_smoke_test.sh` | 제한된 실제 데이터로 forward/backward 학습 경로 점검 |
| `smoke/inference_smoke_test.sh` | 체크포인트 로딩부터 공개 Predictor까지 추론 경로 점검 |

## 1. 전체 모델 학습

```bash
./exe/train_model.sh [DATA_DIR] [MODEL_TYPE]
```

예시:

```bash
./exe/train_model.sh V17_Synthetic_IMU_Dataset lstm
BATCH_SIZE=8 EPOCHS=20 ./exe/train_model.sh V17_Synthetic_IMU_Dataset lstm
MODEL_TYPE=mamba ./exe/train_model.sh V17_Synthetic_IMU_Dataset mamba
```

사용 가능한 환경 변수:

- `EPOCHS`: 기본값 `80`
- `BATCH_SIZE`: 기본값 `16`
- `NUM_WORKERS`: 기본값 `4`
- `LR`: 기본값 `1e-3`
- `PYTHON_BIN`: 기본값 `python`

학습 로그는 `logs/train_<model>_<timestamp>.log`, 최적 체크포인트는
`model/checkpoints/best.pt`에 저장됩니다. Mamba 모델은 현재 PyTorch/CUDA와
호환되는 `mamba-ssm` 패키지를 별도로 설치해야 합니다.

## 2. 파일 기반 단일 추론

```bash
./exe/run_inference.sh CHECKPOINT INPUT_FILE [OUTPUT_FILE]
```

입력 파일은 다음 중 하나여야 합니다.

- `.npy`: checkpoint가 요구하는 정확한 `(T, 6)` 배열
- `.csv`: 정확히 `T`행이며 checkpoint의 6개 IMU 컬럼을 포함하는 파일

출력 파일은 `.npy` 또는 `.csv`를 지원하며, 생략하면 프로젝트 루트의
`prediction.npy`에 저장합니다.

```bash
DEVICE=cpu ./exe/run_inference.sh \
    model/checkpoints/best.pt \
    sample_window.npy \
    outputs/prediction.csv
```

`DEVICE`는 `auto`(기본값), `cpu`, `cuda` 중 하나를 사용할 수 있습니다.
출력은 미래 `true_vertical_specific_force_mps2` trajectory `(H,)`입니다.

## 3. 실시간 시스템 실행

```bash
./exe/run_system.sh [CHECKPOINT]
```

예시:

```bash
./exe/run_system.sh model/checkpoints/best.pt
```

checkpoint 인자는 `MODEL_PATH` 환경 변수로 `main.py`에 전달됩니다.
시스템 시작 시 checkpoint의 sample rate, 입력 길이, IMU 채널 수,
예측 길이가 runtime 설정과 다르면 센서 버퍼를 채우기 전에 종료합니다.
이 스크립트는 MPU6050, GPS, actuator가 연결된 Linux 장비에서 실행해야
하며 I2C/serial/GPIO 권한이 필요할 수 있습니다. 종료는 `Ctrl+C`를
사용합니다.

## 4. pytest 회귀 테스트

```bash
./exe/run_tests.sh
./exe/run_tests.sh -q
./exe/run_tests.sh tests/test_inference.py
```

synthetic data만 사용하므로 실제 대용량 dataset이나 GPU가 필요하지
않습니다.

## 5. 학습 smoke test

```bash
./exe/smoke/train_smoke_test.sh [DATA_DIR]
```

실제 train split 일부로 dataset 생성, 모델 forward, loss, backward,
gradient 유한값 여부를 확인합니다. 정확도나 전체 epoch 학습을 평가하는
테스트는 아닙니다.

```bash
SMOKE_ROWS=20000 SMOKE_SCENARIOS=2 \
    ./exe/smoke/train_smoke_test.sh V17_Synthetic_IMU_Dataset
```

CSV는 기본적으로 앞의 `50,000`행만 읽습니다. 선택한 scenario가
30초 history와 15초 future보다 짧으면 `SMOKE_ROWS`를 늘려주세요.

## 6. 추론 smoke test

```bash
./exe/smoke/inference_smoke_test.sh [DATA_DIR] [CHECKPOINT]
```

checkpoint가 존재하면 해당 모델을 사용합니다. 파일이 없으면 임시
random-weight checkpoint를 만들어 다음 경로를 검사합니다.

1. 데이터와 전처리 통계 준비
2. checkpoint 저장 및 `ModelInference` 복원
3. `(T, 6)` raw IMU 입력 추론
4. 공개 `Predictor` 인터페이스 연결
5. 출력 shape 및 NaN/Inf 검사

```bash
DEVICE=cpu ./exe/smoke/inference_smoke_test.sh \
    V17_Synthetic_IMU_Dataset \
    model/checkpoints/best.pt
```

임시 random-weight checkpoint 사용은 연결 상태만 검증하며 모델 정확도를
의미하지 않습니다. CSV 읽기 제한은 `SMOKE_ROWS`로 조정할 수 있습니다.

## 권장 점검 순서

```bash
./exe/run_tests.sh
./exe/smoke/train_smoke_test.sh V17_Synthetic_IMU_Dataset
./exe/smoke/inference_smoke_test.sh V17_Synthetic_IMU_Dataset
./exe/train_model.sh V17_Synthetic_IMU_Dataset lstm
```

실제 시스템 실행은 유효한 학습 checkpoint와 하드웨어 연결을 확인한 뒤
마지막에 진행하세요.

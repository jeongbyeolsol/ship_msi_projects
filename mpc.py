import numpy as np


class LightMPC:
    def __init__(self, control_weight=0.5):
        # 제어 입력 변화에 대한 페널티 가중치
        self.lambda_w = control_weight

    @staticmethod
    def remove_dc(predicted_trajectory):
        """
        specific-force 예측에서 중력과 기타 DC offset을 제거한다.

        15초 horizon의 median을 equilibrium으로 사용해 단일 충격값이
        기준선을 왜곡하지 않게 한다. 원본 prediction은 변경하지 않는다.
        """
        trajectory = np.asarray(
            predicted_trajectory,
            dtype=np.float32,
        ).reshape(-1)

        if trajectory.size == 0:
            raise ValueError(
                "predicted_trajectory must not be empty."
            )

        if not np.all(np.isfinite(trajectory)):
            raise ValueError(
                "predicted_trajectory contains NaN or Inf."
            )

        baseline = np.median(trajectory)
        return (
            trajectory - baseline
        ).astype(
            np.float32,
            copy=False,
        )

    def calculate_target_stroke(self, predicted_trajectory):
        """
        AI가 예측한 가속도 궤적을 최소화하기 위한 인터셉터 스트로크 계산
        수식 J = ||a_pred - a_lift||^2 + lambda * ||u||^2 의 해석적 해를 구함
        """
        dynamic_trajectory = self.remove_dc(
            predicted_trajectory
        )

        # 가장 흔들림이 심한(Peak) 미래 시점의 가속도 성분을 타겟으로 설정
        peak_accel = dynamic_trajectory[
            np.argmax(
                np.abs(dynamic_trajectory)
            )
        ]
        
        # 선형화된 양력 계수 (실제 선형 테스트 후 튜닝 필요)
        lift_coefficient = 1.2 
        
        # 최적화 수식의 해석적 해 (Analytical Solution)
        # 제어 목표: peak_accel + lift_coefficient * stroke = 0
        optimal_stroke = - (lift_coefficient * peak_accel) / (lift_coefficient**2 + self.lambda_w)
        
        return optimal_stroke

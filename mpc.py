import numpy as np

class LightMPC:
    def __init__(self, control_weight=0.5):
        # 제어 입력 변화에 대한 페널티 가중치
        self.lambda_w = control_weight

    def calculate_target_stroke(self, predicted_trajectory):
        """
        AI가 예측한 가속도 궤적을 최소화하기 위한 인터셉터 스트로크 계산
        수식 J = ||a_pred - a_lift||^2 + lambda * ||u||^2 의 해석적 해를 구함
        """
        # 가장 흔들림이 심한(Peak) 미래 시점의 가속도 성분을 타겟으로 설정
        peak_accel = predicted_trajectory[np.argmax(np.abs(predicted_trajectory))]
        
        # 선형화된 양력 계수 (실제 선형 테스트 후 튜닝 필요)
        lift_coefficient = 1.2 
        
        # 최적화 수식의 해석적 해 (Analytical Solution)
        # 제어 목표: peak_accel + lift_coefficient * stroke = 0
        optimal_stroke = - (lift_coefficient * peak_accel) / (lift_coefficient**2 + self.lambda_w)
        
        return optimal_stroke
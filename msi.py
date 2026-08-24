import math
import numpy as np

class MSICalculator:
    def __init__(self, fs=10.0, window_minutes=20.0):
        self.N_past = int(fs * 60 * window_minutes)
        self.past_buffer = np.zeros(self.N_past, dtype=np.float32)
        self.head = 0
        self.sum_sq = 0.0
        self.count = 0
        
    def update_and_calculate_future_msi(self, new_val, future_trajectory):
        # 1. 과거 데이터 O(1) 이동 합산 (Sliding Window)
        old_val = self.past_buffer[self.head]
        self.sum_sq = self.sum_sq - (old_val**2) + (new_val**2)
        self.past_buffer[self.head] = new_val
        self.head = (self.head + 1) % self.N_past
        
        if self.count < self.N_past:
            self.count += 1
            
        # 2. 과거 합산 + AI가 예측한 미래 궤적 합산 (Hybrid MSDV)
        future_sum_sq = float(np.sum(future_trajectory**2))
        total_sum_sq = self.sum_sq + future_sum_sq
        total_samples = self.count + len(future_trajectory)
        
        # 3. MSDV 및 MSI 산출
        msdv = math.sqrt(total_sum_sq / total_samples)
        msi_percentage = (1.0 / 3.0) * msdv * 100.0
        
        return msi_percentage, msdv
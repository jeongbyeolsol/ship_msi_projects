import scipy.signal as signal
import numpy as np

class RealTimeFilter:
    def __init__(self, fs=10.0):
        nyq = 0.5 * fs
        
        # 1. 고주파 진동 제거용 Low-pass 필터 (예: 2Hz 컷오프)
        self.b_lp, self.a_lp = signal.butter(2, 2.0 / nyq, btype='low')
        self.zi_lp = signal.lfilter_zi(self.b_lp, self.a_lp) * 0.0
        
        # 2. ISO 2631-1 수직 방향(Wk) 근사 대역통과 필터 (0.1 ~ 0.5Hz)
        self.b_iso, self.a_iso = signal.butter(2, [0.1 / nyq, 0.5 / nyq], btype='band')
        self.zi_iso = signal.lfilter_zi(self.b_iso, self.a_iso) * 0.0

    def process(self, value):
        # 1차 Low-pass 통과
        filtered_lp, self.zi_lp = signal.lfilter(self.b_lp, self.a_lp, [value], zi=self.zi_lp)
        # 2차 ISO 가중치 통과
        filtered_iso, self.zi_iso = signal.lfilter(self.b_iso, self.a_iso, filtered_lp, zi=self.zi_iso)
        
        return filtered_iso[0]
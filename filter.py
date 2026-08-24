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

    def clone(self):
        """필터 계수와 현재 내부 상태를 복제한다.

        반환된 필터를 사용해도 원본 필터의 실시간 상태는
        변경되지 않는다.
        """
        cloned = object.__new__(type(self))

        cloned.b_lp = self.b_lp.copy()
        cloned.a_lp = self.a_lp.copy()
        cloned.zi_lp = self.zi_lp.copy()

        cloned.b_iso = self.b_iso.copy()
        cloned.a_iso = self.a_iso.copy()
        cloned.zi_iso = self.zi_iso.copy()

        return cloned

    def process_array(self, values):
        """일차원 신호를 현재 필터 상태에서 연속 처리한다.

        이 메서드는 ``process()``를 순차적으로 호출한 것과 같은
        결과와 최종 필터 상태를 만든다.
        """
        values = np.asarray(values, dtype=np.float64)

        if values.ndim != 1:
            raise ValueError(
                "values must be 1-dimensional, "
                f"got {values.shape}"
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "values contains NaN or Inf."
            )

        if values.size == 0:
            return np.empty(0, dtype=np.float32)

        filtered_lp, self.zi_lp = signal.lfilter(
            self.b_lp,
            self.a_lp,
            values,
            zi=self.zi_lp,
        )

        filtered_iso, self.zi_iso = signal.lfilter(
            self.b_iso,
            self.a_iso,
            filtered_lp,
            zi=self.zi_iso,
        )

        return np.asarray(
            filtered_iso,
            dtype=np.float32,
        )

    def preview_array(self, values):
        """현재 상태를 복제해 신호를 필터링한다.

        MSI 미래 trajectory와 같은 가상 신호를 처리할 때
        사용하며, live filter의 내부 상태는 변경하지 않는다.
        """
        return self.clone().process_array(
            values
        )

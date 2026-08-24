import time
try:
    import pigpio
except ImportError:
    pass

class InterceptorController:
    def __init__(self, pin=18, min_stroke=0.0, max_stroke=50.0):
        self.pin = pin
        self.min_stroke = min_stroke
        self.max_stroke = max_stroke
        
        try:
            self.pi = pigpio.pi()
            if not self.pi.connected:
                raise RuntimeError("pigpiod 데몬이 실행되지 않았습니다. (sudo pigpiod)")
            self.pi.set_mode(self.pin, pigpio.OUTPUT)
            self.hardware_available = True
        except NameError:
            print("[Warning] pigpio가 없어 PWM 제어를 시뮬레이션합니다.")
            self.hardware_available = False

    def _stroke_to_pwm_duty(self, stroke):
        # 스트로크(0~50mm)를 서보 PWM 펄스폭(500~2500us)으로 선형 맵핑
        pulse_width = 500 + (stroke / self.max_stroke) * 2000
        return int(pulse_width)

    def set_stroke(self, target_stroke):
        # 1. 안전 제한 (Saturation Limit)
        safe_stroke = max(self.min_stroke, min(target_stroke, self.max_stroke))
        
        # 2. 하드웨어 제어
        if self.hardware_available:
            duty_cycle = self._stroke_to_pwm_duty(safe_stroke)
            self.pi.set_servo_pulsewidth(self.pin, duty_cycle)
            
        return safe_stroke

    def emergency_stop(self):
        # Fail-safe: 시스템 종료 시 인터셉터를 원위치(0mm)하여 저항 최소화
        self.set_stroke(0.0)
        if self.hardware_available:
            time.sleep(0.5)
            self.pi.set_servo_pulsewidth(self.pin, 0)
            self.pi.stop()
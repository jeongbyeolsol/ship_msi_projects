import serial
import time

class GPSSpeedSensor:
    def __init__(self, port='/dev/ttyUSB0', baudrate=9600):
        """
        GPS 시리얼 통신 초기화
        - port: 라즈베리 파이에 연결된 GPS 포트 (USB 모듈의 경우 보통 /dev/ttyUSB0)
        - baudrate: GPS 모듈의 통신 속도 (기본 9600 또는 4800)
        """
        self.current_speed = 0.0
        try:
            # 타임아웃을 짧게 설정하여 실시간 제어 루프(10Hz)에 지연이 없도록 함
            self.ser = serial.Serial(port, baudrate, timeout=0.05)
            self.hardware_available = True
            print(f"[GPS] {port} 포트에 성공적으로 연결되었습니다.")
        except serial.SerialException as e:
            print(f"[Warning] GPS 연결 실패. 속도는 0.0 Knot로 고정됩니다: {e}")
            self.hardware_available = False

    def read_speed_knots(self):
        """
        시리얼 버퍼에서 최신 NMEA 문장을 읽어 속도(Knot)를 반환합니다.
        """
        if not self.hardware_available:
            return 0.0

        try:
            # 수신 버퍼에 데이터가 쌓여있을 수 있으므로 가장 최신 줄까지 읽어냅니다.
            while self.ser.in_waiting > 0:
                line = self.ser.readline().decode('ascii', errors='ignore').strip()
                
                # RMC 데이터 구조 확인 (미국 GPS는 GPRMC, 다중 위성은 GNRMC로 들어옵니다)
                if line.startswith('$GPRMC') or line.startswith('$GNRMC'):
                    parts = line.split(',')
                    
                    # parts[2]는 데이터 유효성 상태 (A: Active/Valid, V: Void/Invalid)
                    if len(parts) > 7 and parts[2] == 'A':
                        try:
                            # NMEA RMC 규격상 7번째 인덱스가 대지 속도(SOG, Knots)입니다.
                            self.current_speed = float(parts[7])
                        except ValueError:
                            pass
        except Exception as e:
            # 일시적인 통신 노이즈 발생 시 이전 속도를 유지하여 시스템 안정을 도모합니다.
            pass
            
        return self.current_speed
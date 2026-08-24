import main
import controller
import pytest


class _Closable:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def close(self):
        self.events.append(f"{self.name}.close")


def test_actuator_initialization_failure_closes_prior_resources(
    monkeypatch,
):
    events = []

    class FakeIMU:
        def __init__(self):
            events.append("imu.init")

        def close(self):
            events.append("imu.close")

    class FakeGPS:
        def __init__(self, **kwargs):
            events.append("gps.init")
            self.ser = _Closable("gps", events)

    class FakePredictor:
        def __init__(self, **kwargs):
            events.append("predictor.init")

        def validate_runtime_contract(self, **kwargs):
            events.append("predictor.validate")

        def close(self):
            events.append("predictor.close")

    class FailingActuator:
        def __init__(self, **kwargs):
            events.append("actuator.init")
            raise RuntimeError("pigpiod unavailable")

    monkeypatch.setattr(main, "MPU6050", FakeIMU)
    monkeypatch.setattr(main, "GPSSpeedSensor", FakeGPS)
    monkeypatch.setattr(main, "Predictor", FakePredictor)
    monkeypatch.setattr(
        main,
        "InterceptorController",
        FailingActuator,
    )

    main.main()

    assert events == [
        "imu.init",
        "gps.init",
        "predictor.init",
        "predictor.validate",
        "actuator.init",
        "predictor.close",
        "gps.close",
        "imu.close",
    ]


def test_keyboard_interrupt_runs_emergency_stop_before_cleanup(
    monkeypatch,
):
    events = []

    class FakeActuator:
        def emergency_stop(self):
            events.append("actuator.emergency_stop")

    resources_to_install = {
        "actuator": FakeActuator(),
        "ai_predictor": _Closable("predictor", events),
        "gps_sensor": _Closable("gps", events),
        "imu_sensor": _Closable("imu", events),
    }

    def interrupt(resources):
        for name, resource in resources_to_install.items():
            setattr(resources, name, resource)
        raise KeyboardInterrupt

    monkeypatch.setattr(main, "_run_system", interrupt)

    main.main()

    assert events == [
        "actuator.emergency_stop",
        "predictor.close",
        "gps.close",
        "imu.close",
    ]


def test_cleanup_continues_if_emergency_stop_fails():
    events = []
    resources = main._RuntimeResources()

    class FailingActuator:
        def emergency_stop(self):
            events.append("actuator.emergency_stop")
            raise RuntimeError("stop failed")

    resources.actuator = FailingActuator()
    resources.gps_sensor = _Closable("gps", events)
    resources.imu_sensor = _Closable("imu", events)

    main._cleanup_resources(resources)

    assert events == [
        "actuator.emergency_stop",
        "gps.close",
        "imu.close",
    ]


def test_partially_initialized_actuator_closes_pigpio(
    monkeypatch,
):
    events = []

    class FakePi:
        connected = False

        def stop(self):
            events.append("pi.stop")

    class FakePigpio:
        OUTPUT = 1

        @staticmethod
        def pi():
            return FakePi()

    monkeypatch.setattr(
        controller,
        "pigpio",
        FakePigpio,
        raising=False,
    )

    with pytest.raises(
        RuntimeError,
        match="pigpiod",
    ):
        controller.InterceptorController()

    assert events == ["pi.stop"]

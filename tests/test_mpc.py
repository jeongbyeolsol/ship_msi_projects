import numpy as np
import pytest

from mpc import LightMPC


def test_mpc_removes_gravity_and_dc_without_mutating_prediction():
    mpc = LightMPC()
    dynamic = np.array(
        [-2.0, -1.0, 0.0, 1.0, 3.0],
        dtype=np.float32,
    )
    raw_prediction = dynamic + 9.81
    original = raw_prediction.copy()

    centered = mpc.remove_dc(raw_prediction)

    np.testing.assert_allclose(
        centered,
        dynamic,
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        raw_prediction,
        original,
    )


def test_mpc_stroke_is_invariant_to_constant_dc_offset():
    mpc = LightMPC(control_weight=0.1)
    trajectory = np.array(
        [-3.0, -1.0, 0.0, 2.0, 4.0],
        dtype=np.float32,
    )

    assert mpc.calculate_target_stroke(
        trajectory
    ) == pytest.approx(
        mpc.calculate_target_stroke(
            trajectory + 50.0
        )
    )


@pytest.mark.parametrize(
    "trajectory",
    [
        np.array([], dtype=np.float32),
        np.array([0.0, np.nan], dtype=np.float32),
        np.array([0.0, np.inf], dtype=np.float32),
    ],
)
def test_mpc_rejects_invalid_prediction(trajectory):
    with pytest.raises(ValueError):
        LightMPC.remove_dc(trajectory)

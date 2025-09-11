import os
import numpy as np
import pytest

import pinocchio as pin
from frankapy import FrankaArm


def test_angular_velocity_consistency():
    fa = FrankaArm(offline=True)

    # Acquire joint position & velocity from FrankaArm
    q = np.asarray(fa.get_joints())  # shape (7,)
    dq = np.asarray(fa.get_joint_velocities())  # shape (7,)

    # Load Pinocchio model from URDF
    urdf_path = os.path.join(os.path.dirname(__file__), 'franka_description', 'industreal_franka.urdf')
    print("Hey?..")
    assert os.path.isfile(urdf_path), f"URDF not found at {urdf_path}"
    print("URDF found.")
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()

    # Forward kinematics & Jacobians
    pin.forwardKinematics(model, data, q, dq)
    pin.computeJointJacobians(model, data, q)
    pin.framesForwardKinematics(model, data, q)

    # Select an end-effector frame (try common names, else last frame)
    frame_names = [f.name for f in model.frames]
    candidates = ['panda_hand_tcp', 'panda_hand', 'panda_link8', 'tool0']
    frame_id = None
    for name in candidates:
        if name in frame_names:
            frame_id = model.getFrameId(name)
            break
    if frame_id is None:
        frame_id = model.frames[-1].id
    
    vel = pin.getFrameVelocity(model, data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    twist_pin = vel.vector

    # FrankaPy twist (assumed order [linear, angular] as in common FrankaPy API)
    twist_fa = np.asarray(fa.get_twist())  # shape (6,)
    # Heuristic to detect ordering if uncertain
    # If norm of first 3 closer to pin linear part magnitude vs angular, reorder accordingly.
    # Compute Pin linear part for comparison
    lin_pin = twist_pin[:3]
    ang_pin = twist_pin[3:]
    if np.linalg.norm(twist_fa[:3]) < np.linalg.norm(lin_pin) * 1.5:
        # Likely [linear, angular]
        ang_fa = twist_fa[3:]
    else:
        # Possibly already [angular, linear]
        ang_fa = twist_fa[:3]

    np.testing.assert_allclose(ang_fa, ang_pin, rtol=1e-2, atol=1e-3)


if __name__ == "__main__":
    test_angular_velocity_consistency()
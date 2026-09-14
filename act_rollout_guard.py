"""Runtime action guard for ACT simulation rollouts.


This module sits between the policy and the MuJoCo environment. It does not
change the model or training data; it only clamps risky rollout-time commands
before env.step(action).
"""


from dataclasses import dataclass
from typing import Optional, Tuple


import numpy as np



@dataclass
class GuardInfo:
    """Small debug record for a single guarded action."""
    raw_action: np.ndarray
    guarded_action: np.ndarray
    delta_clamped: bool
    gripper_locked: bool
    release_allowed: bool
    mug_to_plate_xy: Optional[float]
    ee_to_plate_xy: Optional[float]
    ee_speed: Optional[float]



class ActRolloutGuard:
    """Clamp ACT actions and protect against early gripper release.

    The expected action format for this repository is:
        action[:6] -> arm command, usually joint-angle target
        action[6]  -> gripper command in [0, 1], where larger means closing

    The guard has three jobs:
        1. Keep arm command changes within max_delta.
        2. Latch a closed gripper until release is likely safe.
        3. Optionally apply EMA smoothing to arm commands.
    """

    def __init__(
        self,
        max_delta: float = 0.12,
        gripper_close_threshold: float = 0.5,
        gripper_open_threshold: float = 0.35,
        min_closed_steps: int = 20,
        release_plate_xy: float = 0.12,
        release_ee_xy: float = 0.18,
        release_min_ee_z: float = 0.86,
        still_delta_norm: float = 0.04,
        max_ee_speed_for_release: float = 0.025,
        ema_alpha: float = 1.0,
        verbose: bool = False,
    ):
        if not 0.0 < max_delta:
            raise ValueError("max_delta must be positive")
        if not 0.0 <= ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in [0, 1]")
        self.max_delta = float(max_delta)
        self.gripper_close_threshold = float(gripper_close_threshold)
        self.gripper_open_threshold = float(gripper_open_threshold)
        self.min_closed_steps = int(min_closed_steps)
        self.release_plate_xy = float(release_plate_xy)
        self.release_ee_xy = float(release_ee_xy)
        self.release_min_ee_z = float(release_min_ee_z)
        self.still_delta_norm = float(still_delta_norm)
        self.max_ee_speed_for_release = float(max_ee_speed_for_release)
        self.ema_alpha = float(ema_alpha)
        self.verbose = bool(verbose)
        self.reset()

    def reset(self) -> None:
        self.prev_arm: Optional[np.ndarray] = None
        self.prev_ema_arm: Optional[np.ndarray] = None
        self.prev_ee_pos: Optional[np.ndarray] = None
        self.gripper_latched_closed = False
        self.closed_steps = 0
        self.last_info: Optional[GuardInfo] = None

    def step(self, action: np.ndarray, env=None) -> np.ndarray:
        raw = np.asarray(action, dtype=np.float32).reshape(-1)
        if raw.shape[0] != 7:
            raise ValueError(f"Expected action shape (7,), got {raw.shape}")

        guarded = raw.copy()
        delta_clamped = False
        gripper_locked = False

        arm = guarded[:6].astype(np.float32, copy=True)
        if self.ema_alpha < 1.0:
            if self.prev_ema_arm is None:
                ema_arm = arm
            else:
                ema_arm = self.ema_alpha * arm + (1.0 - self.ema_alpha) * self.prev_ema_arm
            self.prev_ema_arm = ema_arm.astype(np.float32, copy=True)
            arm = self.prev_ema_arm.copy()

        if self.prev_arm is not None:
            delta = arm - self.prev_arm
            clipped_delta = np.clip(delta, -self.max_delta, self.max_delta)
            if not np.allclose(delta, clipped_delta):
                delta_clamped = True
            arm = self.prev_arm + clipped_delta

        guarded[:6] = arm
        release_allowed, metrics = self._release_allowed(env, arm)

        g = float(np.clip(guarded[6], 0.0, 1.0))
        if g >= self.gripper_close_threshold:
            self.gripper_latched_closed = True
            self.closed_steps += 1
        elif self.gripper_latched_closed:
            if (not release_allowed) and g <= self.gripper_open_threshold:
                g = self.gripper_close_threshold
                gripper_locked = True
            else:
                self.gripper_latched_closed = False
                self.closed_steps = 0

        guarded[6] = g
        guarded = guarded.astype(np.float32, copy=False)
        self.prev_arm = arm.astype(np.float32, copy=True)

        if not self.gripper_latched_closed:
            self.closed_steps = 0

        self.last_info = GuardInfo(
            raw_action=raw.copy(),
            guarded_action=guarded.copy(),
            delta_clamped=delta_clamped,
            gripper_locked=gripper_locked,
            release_allowed=release_allowed,
            mug_to_plate_xy=metrics[0],
            ee_to_plate_xy=metrics[1],
            ee_speed=metrics[2],
        )

        if self.verbose and (delta_clamped or gripper_locked):
            print(
                "[guard]",
                f"delta_clamped={delta_clamped}",
                f"gripper_locked={gripper_locked}",
                f"release_allowed={release_allowed}",
            )

        return guarded

    def _release_allowed(self, env, arm: np.ndarray) -> Tuple[bool, Tuple[Optional[float], Optional[float], Optional[float]]]:
        if env is None:
            return True, (None, None, None)

        try:
            p_mug, p_plate = env.get_obj_pose()
            ee_pose = env.get_ee_pose()
        except Exception:
            return True, (None, None, None)

        p_mug = np.asarray(p_mug, dtype=np.float32)
        p_plate = np.asarray(p_plate, dtype=np.float32)
        ee_pos = np.asarray(ee_pose[:3], dtype=np.float32)

        mug_to_plate_xy = float(np.linalg.norm(p_mug[:2] - p_plate[:2]))
        ee_to_plate_xy = float(np.linalg.norm(ee_pos[:2] - p_plate[:2]))
        if self.prev_ee_pos is None:
            ee_speed = 0.0
        else:
            ee_speed = float(np.linalg.norm(ee_pos - self.prev_ee_pos))
        self.prev_ee_pos = ee_pos.copy()

        above_plate = ee_to_plate_xy <= self.release_ee_xy and ee_pos[2] >= self.release_min_ee_z
        mug_near_plate = mug_to_plate_xy <= self.release_plate_xy
        arm_still = True
        if self.prev_arm is not None:
            arm_still = bool(np.linalg.norm(arm - self.prev_arm) <= self.still_delta_norm)
        held_long_enough = self.closed_steps >= self.min_closed_steps
        ee_slow = ee_speed <= self.max_ee_speed_for_release
        release_allowed = bool(
            held_long_enough
            and ee_slow
            and (mug_near_plate or above_plate)
            and (arm_still or above_plate)
        )
        return release_allowed, (mug_to_plate_xy, ee_to_plate_xy, ee_speed)

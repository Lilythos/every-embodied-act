"""Collect keyboard teleoperation demos into a LeRobotDataset.

This script is the command-line version of 1.collect_data.ipynb.

Example:
    python collect_teleop_dataset.py --root ./demo_data_collected --num-demos 20 --overwrite
"""

import argparse
import os
import shutil
from pathlib import Path

import glfw
import numpy as np
from PIL import Image

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from mujoco_env.y_env import SimpleEnv


TELEOP_KEY_NAMES = {
    glfw.KEY_W: "W",
    glfw.KEY_S: "S",
    glfw.KEY_A: "A",
    glfw.KEY_D: "D",
    glfw.KEY_R: "R",
    glfw.KEY_F: "F",
    glfw.KEY_Q: "Q",
    glfw.KEY_E: "E",
    glfw.KEY_UP: "UP",
    glfw.KEY_DOWN: "DOWN",
    glfw.KEY_LEFT: "LEFT",
    glfw.KEY_RIGHT: "RIGHT",
    glfw.KEY_SPACE: "SPACE",
    glfw.KEY_Z: "Z",
}


def parse_seed(text: str):
    if text.lower() in {"none", "random"}:
        return None
    return int(text)


def parse_args():
    parser = argparse.ArgumentParser(description="Collect MuJoCo teleop demos as a LeRobotDataset")
    parser.add_argument("--root", default="./demo_data_collected", help="Dataset output directory")
    parser.add_argument("--repo-id", default="datawhale_eai_pnp", help="Local LeRobot dataset repo id/name")
    parser.add_argument("--task", default="Put mug cup on the plate", help="Task string saved with each frame")
    parser.add_argument("--xml", default="./asset/example_scene_y.xml", help="MuJoCo scene XML")
    parser.add_argument("--num-demos", type=int, default=5, help="Number of successful episodes to save")
    parser.add_argument("--seed", default="0", help="Reset seed. Use 'none' for random object placement")
    parser.add_argument("--fps", type=int, default=20, help="Collection FPS")
    parser.add_argument("--robot-type", default="omy", help="robot_type stored in LeRobot metadata")
    parser.add_argument("--append", action="store_true", help="Append episodes to an existing dataset")
    parser.add_argument("--overwrite", action="store_true", help="Delete and recreate the dataset root")
    parser.add_argument("--record-force-proxy", action="store_true", default=True, help="Store a signed gripper/contact force proxy")
    parser.add_argument("--no-force-proxy", dest="record_force_proxy", action="store_false", help="Do not store force_proxy")
    parser.add_argument("--teleop-pos-step", type=float, default=None, help="Override end-effector translation step in meters")
    parser.add_argument("--teleop-rot-step", type=float, default=None, help="Override end-effector rotation step in radians")
    parser.add_argument("--debug-teleop", action="store_true", help="Print key/action diagnostics while collecting")
    parser.add_argument("--image-writer-threads", type=int, default=4, help="LeRobot image writer threads")
    parser.add_argument("--image-writer-processes", type=int, default=0, help="LeRobot image writer processes")
    return parser.parse_args()


def format_key_list(keys):
    names = [TELEOP_KEY_NAMES.get(key, str(key)) for key in keys]
    return "[" + ", ".join(names) + "]"


def dataset_features(record_force_proxy: bool):
    features = {
        "observation.image": {
            "dtype": "image",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.wrist_image": {
            "dtype": "image",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["action"],
        },
        "obj_init": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["obj_init"],
        },
    }
    if record_force_proxy:
        features["force_proxy"] = {
            "dtype": "float32",
            "shape": (1,),
            "names": ["force_proxy"],
        }
    return features


def create_or_load_dataset(args):
    root = Path(args.root)
    if args.overwrite and root.exists():
        shutil.rmtree(root)

    if root.exists():
        if not args.append:
            raise FileExistsError(
                f"{root} already exists. Use --append to add episodes or --overwrite to recreate it."
            )
        print(f"Loading existing dataset: {root}")
        dataset = LeRobotDataset(args.repo_id, root=root)
        if args.record_force_proxy and "force_proxy" not in dataset.meta.features:
            print("[warn] Existing dataset has no force_proxy feature. Disabling force_proxy for append mode.")
            args.record_force_proxy = False
        return dataset

    print(f"Creating new dataset: {root}")
    return LeRobotDataset.create(
        repo_id=args.repo_id,
        root=root,
        robot_type=args.robot_type,
        fps=args.fps,
        features=dataset_features(args.record_force_proxy),
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )


def resize_rgb(image):
    return np.asarray(Image.fromarray(image).resize((256, 256)))


def has_gripper_mug_contact(env: SimpleEnv) -> bool:
    try:
        contact_pairs = env.env.get_contact_body_names()
    except Exception:
        return False

    finger_bodies = {
        "rh_p12_rn_r1",
        "rh_p12_rn_r2",
        "rh_p12_rn_l1",
        "rh_p12_rn_l2",
        "left_finger",
        "right_finger",
        "left_outer_knuckle",
        "right_outer_knuckle",
    }
    mug_bodies = {"object_mug_5", "body_obj_mug_5"}
    for body_a, body_b in contact_pairs:
        pair = {body_a, body_b}
        if pair & finger_bodies and pair & mug_bodies:
            return True
    return False


def compute_force_proxy(env: SimpleEnv, teleop_action: np.ndarray, prev_gripper_cmd: float) -> np.ndarray:
    """Return a signed force proxy, not a real force sensor reading.

    +1: closing while touching the mug
    -1: opening/releasing after a closed command
     0: no useful force signal
    """
    gripper_cmd = float(teleop_action[-1])
    contact = has_gripper_mug_contact(env)
    if contact and gripper_cmd >= 0.5:
        value = 1.0
    elif gripper_cmd < 0.5 and prev_gripper_cmd >= 0.5:
        value = -1.0
    else:
        value = 0.0
    return np.array([value], dtype=np.float32)


def main():
    args = parse_args()

    # Match the notebook behavior: this may create/use an ASCII junction path on Windows.
    try:
        import env_config  # noqa: F401
    except Exception as exc:
        print(f"[warn] env_config import failed, continuing in current directory: {exc}")

    seed = parse_seed(str(args.seed))
    dataset = create_or_load_dataset(args)

    if args.teleop_pos_step is not None:
        os.environ["TELEOP_POS_STEP"] = str(args.teleop_pos_step)
    if args.teleop_rot_step is not None:
        os.environ["TELEOP_ROT_STEP"] = str(args.teleop_rot_step)

    env = SimpleEnv(args.xml, seed=seed, state_type="joint_angle")
    saved_count = 0
    record_flag = False
    prev_gripper_cmd = 0.0
    control_tick = 0

    env.grab_image()
    env.render(teleop=True)

    print("Click the Tabletop/MuJoCo window once before pressing keys.")
    print("Move end-effector: W/S = X-/X+, A/D = Y-/Y+, R/F = Z+/Z-.")
    print("Rotate: arrows and Q/E. SPACE toggles gripper. Z clears the current episode.")
    print(f"Teleop step: pos={env.teleop_pos_step:.4f} m/tick, rot={env.teleop_rot_step:.4f} rad/tick, fps={args.fps}.")
    print("An episode is saved only when env.check_success() becomes True.")

    while env.env.is_viewer_alive() and saved_count < args.num_demos:
        env.step_env()
        if not env.env.loop_every(HZ=args.fps):
            continue
        control_tick += 1

        if env.check_success() and record_flag:
            dataset.save_episode()
            saved_count += 1
            print(f"Episode saved ({saved_count}/{args.num_demos})")
            env.reset(seed=seed)
            dataset.clear_episode_buffer()
            record_flag = False
            prev_gripper_cmd = 0.0
            continue

        teleop_action, reset = env.teleop_robot()
        if args.debug_teleop:
            pressed = env.env.get_key_pressed_list()
            repeated = env.env.get_key_repeated_list()
            has_motion = np.linalg.norm(teleop_action[:6]) > env.action_deadband
            if has_motion or pressed or repeated or reset:
                print(
                    "[teleop] "
                    f"pressed={format_key_list(pressed)} "
                    f"repeat={format_key_list(repeated)} "
                    f"dpos={np.round(teleop_action[:3], 4).tolist()} "
                    f"drot={np.round(teleop_action[3:6], 4).tolist()} "
                    f"gripper={float(teleop_action[-1]):.1f}"
                )
            elif control_tick % max(args.fps * 5, 1) == 0:
                print("[teleop] no key input detected. Click the Tabletop window, then hold W/S/A/D/R/F.")

        if reset:
            env.reset(seed=seed)
            dataset.clear_episode_buffer()
            record_flag = False
            prev_gripper_cmd = 0.0
            print("Episode buffer cleared")
            continue

        if not record_flag and (np.linalg.norm(teleop_action[:6]) > 0.0 or teleop_action[-1] != 0.0):
            record_flag = True
            print("Start recording")

        ee_pose = env.get_ee_pose()
        agent_image, wrist_image = env.grab_image()
        joint_action = env.step(teleop_action)

        if record_flag:
            frame = {
                "observation.image": resize_rgb(agent_image),
                "observation.wrist_image": resize_rgb(wrist_image),
                "observation.state": ee_pose.astype(np.float32),
                "action": joint_action.astype(np.float32),
                "obj_init": env.obj_init_pose.astype(np.float32),
            }
            if args.record_force_proxy:
                frame["force_proxy"] = compute_force_proxy(env, teleop_action, prev_gripper_cmd)
            dataset.add_frame(frame, task=args.task)

        prev_gripper_cmd = float(teleop_action[-1])
        env.render(teleop=True)

    print(f"Collection finished. Saved {saved_count} new episode(s) to {Path(args.root).resolve()}.")


if __name__ == "__main__":
    main()

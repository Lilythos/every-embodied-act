"""Run a guarded ACT policy rollout in the local MuJoCo tabletop scene.


Example:
    conda activate C:\\conda_envs\\embodied_env
    cd C:\\every-embodied\\every-embodied\\06-策略抓取或抓取VLA\\大模型控制、VLA、VLM\\04mujoco复现ACT、Pi0、SmolVLA
    python run_act_sim_guarded.py --checkpoint ckpt/demo_act_fixed47

"""


import argparse
import csv
import re
from datetime import datetime
from pathlib import Path


import numpy as np
import torch
import torchvision
from PIL import Image


from act_rollout_guard import ActRolloutGuard
from lerobot.common.policies.act.modeling_act import ACTPolicy
from mujoco_env.y_env import SimpleEnv



def parse_args():
    parser = argparse.ArgumentParser(description="Guarded ACT rollout in MuJoCo")
    parser.add_argument("--checkpoint", default="ckpt/demo_act_fixed47", help="Policy checkpoint directory")
    parser.add_argument("--xml", default="asset/example_scene_y.xml", help="MuJoCo scene XML")
    parser.add_argument("--task", default="Put mug cup on the plate", help="Task string passed to ACT")
    parser.add_argument("--seed", type=int, default=0, help="Environment reset seed")
    parser.add_argument("--hz", type=int, default=20, help="Policy control frequency")
    parser.add_argument("--max-steps", type=int, default=600, help="Maximum policy steps before stopping")
    parser.add_argument("--max-delta", type=float, default=0.12, help="Max per-step change for each arm action dimension")
    parser.add_argument("--ema-alpha", type=float, default=1.0, help="Arm action EMA alpha; 1.0 disables smoothing")
    parser.add_argument("--min-closed-steps", type=int, default=20, help="Minimum closed-gripper policy ticks before release")
    parser.add_argument("--release-plate-xy", type=float, default=0.12, help="Allow release when mug is this close to plate in XY")
    parser.add_argument("--release-ee-xy", type=float, default=0.18, help="Allow release when end effector is this close to plate in XY")
    parser.add_argument("--release-min-ee-z", type=float, default=0.86, help="Allow release only above this EE z when using EE/plate rule")
    parser.add_argument("--max-ee-speed-for-release", type=float, default=0.025, help="Allow gripper release only when EE speed is below this value")
    parser.add_argument("--log-dir", default="rollout_logs", help="Directory for rollout CSV logs")
    parser.add_argument("--run-name", default=None, help="Optional name prefix for this rollout")
    parser.add_argument("--no-guard", action="store_true", help="Disable action guard for direct policy rollout")
    parser.add_argument("--no-render", action="store_true", help="Do not call env.render() each policy tick")
    parser.add_argument("--verbose-guard", action="store_true", help="Print guard interventions")
    return parser.parse_args()


def resolve_path(project_root: Path, path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return project_root / path


def build_observation(env: SimpleEnv, device: str, task: str, timestamp: float, img_transform):
    state = env.get_ee_pose()
    image, wrist_image = env.grab_image()

    image = Image.fromarray(image).resize((256, 256))
    wrist_image = Image.fromarray(wrist_image).resize((256, 256))

    return {
        "observation.state": torch.tensor([state], dtype=torch.float32, device=device),
        "observation.image": img_transform(image).unsqueeze(0).to(device),
        "observation.wrist_image": img_transform(wrist_image).unsqueeze(0).to(device),
        "task": [task],
        "timestamp": torch.tensor([timestamp], dtype=torch.float32, device=device),
    }


def safe_name(text: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return name.strip("_") or "rollout"


def get_debug(env: SimpleEnv) -> dict:
    if hasattr(env, "get_success_debug"):
        return env.get_success_debug()
    return {"success": bool(env.check_success())}


def action_columns(prefix: str, action: np.ndarray) -> dict:
    return {f"{prefix}_{idx}": float(action[idx]) for idx in range(7)}


def write_step_log(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_summary(path: Path, row: dict) -> None:
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    checkpoint = resolve_path(project_root, args.checkpoint)
    xml_path = resolve_path(project_root, args.xml)
    log_dir = resolve_path(project_root, args.log_dir)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not xml_path.exists():
        raise FileNotFoundError(f"XML scene not found: {xml_path}")
    log_dir.mkdir(parents=True, exist_ok=True)

    guard_tag = "noguard" if args.no_guard else f"guard_md{args.max_delta:g}"
    run_name = args.run_name or f"{checkpoint.name}_seed{args.seed}_{guard_tag}_{datetime.now():%Y%m%d_%H%M%S}"
    run_name = safe_name(run_name)
    step_log_path = log_dir / f"{run_name}_steps.csv"
    summary_log_path = log_dir / "summary.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")
    print(f"checkpoint = {checkpoint}")
    print("dataset_stats = checkpoint buffers")
    print(f"step_log = {step_log_path}")

    policy = ACTPolicy.from_pretrained(str(checkpoint)).to(device)
    policy.eval()
    policy.reset()

    env = SimpleEnv(str(xml_path), action_type="joint_angle")
    env.reset(seed=args.seed)

    guard = None
    if not args.no_guard:
        guard = ActRolloutGuard(
            max_delta=args.max_delta,
            ema_alpha=args.ema_alpha,
            min_closed_steps=args.min_closed_steps,
            release_plate_xy=args.release_plate_xy,
            release_ee_xy=args.release_ee_xy,
            release_min_ee_z=args.release_min_ee_z,
            max_ee_speed_for_release=args.max_ee_speed_for_release,
            verbose=args.verbose_guard,
        )
        guard.reset()

    img_transform = torchvision.transforms.ToTensor()
    step = 0
    success = False
    end_reason = "running"
    guard_delta_clamps = 0
    guard_gripper_locks = 0
    step_rows = []
    print("rollout started")

    while env.env.is_viewer_alive() and step < args.max_steps:
        env.step_env()
        if not env.env.loop_every(HZ=args.hz):
            continue

        if env.check_success():
            print(f"success before action at step {step}")
            success = True
            end_reason = "success_before_action"
            break

        obs = build_observation(
            env=env,
            device=device,
            task=args.task,
            timestamp=step / float(args.hz),
            img_transform=img_transform,
        )

        with torch.inference_mode():
            raw_action = policy.select_action(obs)[0].detach().cpu().numpy()

        if guard is None:
            action = raw_action.astype(np.float32, copy=True)
            guard_info = None
        else:
            action = guard.step(raw_action, env=env)
            guard_info = guard.last_info
            if guard_info is not None:
                guard_delta_clamps += int(guard_info.delta_clamped)
                guard_gripper_locks += int(guard_info.gripper_locked)

        env.step(action)
        if not args.no_render:
            env.render()

        dbg = get_debug(env)
        success = bool(dbg.get("success", env.check_success()))
        row = {
            "run_name": run_name,
            "checkpoint": checkpoint.name,
            "seed": args.seed,
            "step": step,
            "timestamp": step / float(args.hz),
            "guard_enabled": guard is not None,
            "delta_clamped": bool(guard_info.delta_clamped) if guard_info is not None else False,
            "gripper_locked": bool(guard_info.gripper_locked) if guard_info is not None else False,
            "release_allowed": bool(guard_info.release_allowed) if guard_info is not None else True,
            "xy_dist": float(dbg.get("xy_dist", np.nan)),
            "z_dist": float(dbg.get("z_dist", np.nan)),
            "ee_z": float(dbg.get("ee_z", np.nan)),
            "gripper_open_norm": float(dbg.get("gripper_open_norm", np.nan)),
            "success": success,
        }
        row.update(action_columns("raw_action", raw_action))
        row.update(action_columns("guarded_action", action))
        step_rows.append(row)

        if step % args.hz == 0 and hasattr(env, "get_success_debug"):
            print(
                f"step={step}",
                f"xy={dbg.get('xy_dist', np.nan):.3f}",
                f"z={dbg.get('z_dist', np.nan):.3f}",
                f"ee_z={dbg.get('ee_z', np.nan):.3f}",
                f"grip_open={dbg.get('gripper_open_norm', np.nan):.3f}",
            )

        step += 1
        if success:
            print(f"Success at step {step}")
            end_reason = "success_after_action"
            break
    else:
        if step >= args.max_steps:
            end_reason = "max_steps"
        else:
            end_reason = "viewer_closed"

    final_dbg = get_debug(env)
    success = bool(final_dbg.get("success", env.check_success()))
    summary = {
        "run_name": run_name,
        "checkpoint": checkpoint.name,
        "checkpoint_path": str(checkpoint),
        "seed": args.seed,
        "guard_enabled": guard is not None,
        "max_delta": args.max_delta,
        "ema_alpha": args.ema_alpha,
        "min_closed_steps": args.min_closed_steps,
        "release_plate_xy": args.release_plate_xy,
        "release_ee_xy": args.release_ee_xy,
        "release_min_ee_z": args.release_min_ee_z,
        "max_ee_speed_for_release": args.max_ee_speed_for_release,
        "steps": step,
        "success": success,
        "end_reason": end_reason,
        "guard_delta_clamps": guard_delta_clamps,
        "guard_gripper_locks": guard_gripper_locks,
        "final_xy_dist": float(final_dbg.get("xy_dist", np.nan)),
        "final_z_dist": float(final_dbg.get("z_dist", np.nan)),
        "final_ee_z": float(final_dbg.get("ee_z", np.nan)),
        "final_gripper_open_norm": float(final_dbg.get("gripper_open_norm", np.nan)),
    }

    write_step_log(step_log_path, step_rows)
    append_summary(summary_log_path, summary)
    print(f"rollout ended: success={success}, reason={end_reason}, steps={step}")
    print(f"saved step log: {step_log_path}")
    print(f"updated summary: {summary_log_path}")


if __name__ == "__main__":
    main()

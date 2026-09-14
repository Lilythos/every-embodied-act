"""Train an ACT policy from a local LeRobotDataset.

Example:
    python train_act_from_dataset.py --dataset-root ./demo_data_collected --checkpoint-dir ./ckpt/act_fullflow_v1
"""

import argparse
import io
import json
import os
import tempfile
import time
from pathlib import Path

# Set HuggingFace cache before importing LeRobot/datasets.
_HF_TMP = Path(tempfile.gettempdir())
os.environ["HF_DATASETS_CACHE"] = str(_HF_TMP / "lerobot_hf_cache_train_script")
os.environ["HF_HOME"] = str(_HF_TMP / "lerobot_hf_home_train_script")

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Train ACT from a local LeRobotDataset")
    parser.add_argument("--dataset-root", default="./demo_data_collected", help="LeRobot dataset root")
    parser.add_argument("--repo-id", default="datawhale_eai_pnp", help="LeRobot dataset repo id/name")
    parser.add_argument("--checkpoint-dir", default="./ckpt/act_fullflow_v1", help="Output checkpoint directory")
    parser.add_argument("--training-steps", type=int, default=5000, help="Number of optimizer steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--chunk-size", type=int, default=100, help="ACT action chunk size")
    parser.add_argument("--n-action-steps", type=int, default=100, help="ACT action steps used at inference")
    parser.add_argument("--stats-mode", choices=["dataset", "identity"], default="dataset", help="Normalization stats source")
    parser.add_argument("--image-noise-std", type=float, default=0.02, help="Gaussian image noise std; 0 disables it")
    parser.add_argument("--save-every", type=int, default=500, help="Save numbered checkpoint every N steps; 0 disables")
    parser.add_argument("--log-every", type=int, default=50, help="Print/log loss every N steps")
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cpu")
    parser.add_argument("--seed", type=int, default=1000, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Build dataset/model and run one forward pass without saving")
    return parser.parse_args()


def setup_hf_cache():
    tmp = Path(tempfile.gettempdir())
    os.environ["HF_DATASETS_CACHE"] = str(tmp / "lerobot_hf_cache_train_script")
    os.environ["HF_HOME"] = str(tmp / "lerobot_hf_home_train_script")
    Path(os.environ["HF_DATASETS_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)


def patch_lerobot_image_decode():
    import lerobot.common.datasets.lerobot_dataset as ds_mod
    import lerobot.common.datasets.utils as ds_utils

    def hf_transform_to_torch_patched(items_dict):
        to_tensor = T.ToTensor()
        for key in items_dict:
            first_item = items_dict[key][0]
            if isinstance(first_item, Image.Image):
                items_dict[key] = [to_tensor(img) for img in items_dict[key]]
            elif isinstance(first_item, dict) and ("bytes" in first_item or "path" in first_item):
                out = []
                for value in items_dict[key]:
                    if isinstance(value, dict) and value.get("bytes") is not None:
                        img = Image.open(io.BytesIO(value["bytes"])).convert("RGB")
                        out.append(to_tensor(img))
                    else:
                        out.append(value)
                items_dict[key] = out
            elif first_item is None:
                pass
            else:
                items_dict[key] = [
                    value if isinstance(value, str) else torch.tensor(value) for value in items_dict[key]
                ]
        return items_dict

    ds_utils.hf_transform_to_torch = hf_transform_to_torch_patched
    ds_mod.hf_transform_to_torch = hf_transform_to_torch_patched


class AddGaussianNoise:
    def __init__(self, std: float):
        self.std = float(std)

    def __call__(self, tensor):
        if self.std <= 0:
            return tensor
        return tensor + torch.randn_like(tensor) * self.std


class Clamp01:
    def __call__(self, tensor):
        return tensor.clamp(0, 1)


def make_image_transform(noise_std: float):
    return T.Compose([AddGaussianNoise(noise_std), Clamp01()])


def make_identity_stats(stats):
    identity = {}
    for key, value in stats.items():
        if "mean" in value and "std" in value:
            mean = torch.as_tensor(value["mean"])
            std = torch.as_tensor(value["std"])
            identity[key] = {
                "mean": torch.zeros_like(mean, dtype=torch.float32),
                "std": torch.ones_like(std, dtype=torch.float32),
            }
    return identity


def select_policy_features(dataset_features):
    policy_features = dataset_to_policy_features(dataset_features)
    input_keys = ["observation.image", "observation.wrist_image", "observation.state"]
    output_keys = ["action"]

    missing = [key for key in input_keys + output_keys if key not in policy_features]
    if missing:
        raise KeyError(f"Dataset is missing required feature(s): {missing}")

    input_features = {key: policy_features[key] for key in input_keys}
    output_features = {key: policy_features[key] for key in output_keys}
    return input_features, output_features


def save_training_artifacts(
    checkpoint_dir: Path,
    args,
    dataset,
    dataset_metadata,
    losses: list[float],
    total_time_s: float,
):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = checkpoint_dir / "training_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        for step, loss in enumerate(losses, start=1):
            f.write(f"{step},{loss:.8f}\n")

    summary = {
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "repo_id": args.repo_id,
        "total_episodes": getattr(dataset, "num_episodes", None),
        "total_frames": getattr(dataset, "num_frames", None),
        "features": dataset_metadata.features,
        "chunk_size": args.chunk_size,
        "n_action_steps": args.n_action_steps,
        "batch_size": args.batch_size,
        "training_steps": args.training_steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "stats_mode": args.stats_mode,
        "final_loss": float(losses[-1]) if losses else float("nan"),
        "avg_loss_last_50": float(np.mean(losses[-50:])) if losses else float("nan"),
        "total_time_s": round(float(total_time_s), 3),
        "seed": args.seed,
    }
    with open(checkpoint_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)


def main():
    args = parse_args()
    setup_hf_cache()
    patch_lerobot_image_decode()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    dataset_root = Path(args.dataset_root)
    checkpoint_dir = Path(args.checkpoint_dir)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"device = {device}")
    print(f"dataset_root = {dataset_root.resolve()}")
    print(f"checkpoint_dir = {checkpoint_dir.resolve()}")

    dataset_metadata = LeRobotDatasetMetadata(args.repo_id, root=dataset_root)
    input_features, output_features = select_policy_features(dataset_metadata.features)

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        pretrained_backbone_weights=None,
    )
    delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)

    stats = dataset_metadata.stats
    if args.stats_mode == "identity":
        stats = make_identity_stats(dataset_metadata.stats)

    policy = ACTPolicy(cfg, dataset_stats=stats).to(device)
    policy.train()

    dataset = LeRobotDataset(
        args.repo_id,
        delta_timestamps=delta_timestamps,
        root=dataset_root,
        image_transforms=make_image_transform(args.image_noise_std),
    )

    num_workers = 0 if os.name == "nt" else 4
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    if len(dataloader) == 0:
        raise RuntimeError("DataLoader is empty. Check dataset size and batch size.")

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.dry_run:
        batch = next(iter(dataloader))
        batch = {key: (value.to(device) if isinstance(value, torch.Tensor) else value) for key, value in batch.items()}
        policy.eval()
        with torch.inference_mode():
            loss, _ = policy.forward(batch)
        print(f"dry_run_loss = {float(loss.detach().cpu().item()):.6f}")
        print("dry run ok")
        return

    losses = []
    step = 0
    start_time = time.perf_counter()

    pbar = tqdm(total=args.training_steps, desc="ACT Training", dynamic_ncols=True)
    while step < args.training_steps:
        for batch in dataloader:
            batch = {key: (value.to(device) if isinstance(value, torch.Tensor) else value) for key, value in batch.items()}

            optimizer.zero_grad(set_to_none=True)
            loss, _ = policy.forward(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
            optimizer.step()

            loss_value = float(loss.detach().cpu().item())
            losses.append(loss_value)
            step += 1
            pbar.update(1)

            if step % args.log_every == 0 or step == 1:
                pbar.set_postfix(loss=f"{loss_value:.4f}")

            if args.save_every > 0 and step % args.save_every == 0:
                step_dir = checkpoint_dir / "checkpoints" / f"step_{step:06d}"
                step_dir.mkdir(parents=True, exist_ok=True)
                policy.save_pretrained(step_dir)

            if step >= args.training_steps:
                break
    pbar.close()

    policy.eval()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(checkpoint_dir)
    total_time_s = time.perf_counter() - start_time
    save_training_artifacts(checkpoint_dir, args, dataset, dataset_metadata, losses, total_time_s)
    print(f"Training finished. Saved checkpoint to: {checkpoint_dir.resolve()}")


if __name__ == "__main__":
    main()

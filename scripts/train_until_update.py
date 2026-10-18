import argparse
import math
import os
import subprocess
import sys
import uuid

import torch

from scripts.config import TrainingPipelineCfg
from scripts.utils import check_dir


def add_train_profile(
    cmd: list[str],
    profile: str,
    cfg: TrainingPipelineCfg,
) -> dict[str, str]:
    env_update = {
        "HYDRA_FULL_ERROR": "1",
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
    }

    if profile == "cpu":
        env_update["CUDA_VISIBLE_DEVICES"] = ""

        cmd.extend([
            "common.cpu=true",
            "common.fp16=false",
            "distributed_training.distributed_world_size=1",
            f"dataset.max_tokens={cfg.cpu_max_tokens}",
            f"dataset.num_workers={cfg.cpu_num_workers}",
        ])

        return env_update

    if profile == "gpu":
        env_update["CUDA_VISIBLE_DEVICES"] = cfg.a30_cuda_devices

        cmd.extend([
            "common.cpu=false",
            "common.fp16=true",
            "distributed_training.distributed_world_size=2",
            f"dataset.max_tokens={cfg.a30_max_tokens}",
            f"dataset.num_workers={cfg.a30_num_workers}",
        ])

        return env_update

    raise ValueError(f"unknown profile: {profile}")


def get_checkpoint_position(checkpoint_path):
    if not checkpoint_path.exists():
        return 1, 0

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    extra = checkpoint.get("extra_state", {})
    iterator = extra.get("train_iterator", {})
    history = checkpoint.get("optimizer_history", [])

    if (
        iterator.get("version") != 2
        or iterator.get("iterations_in_epoch") != 0
        or checkpoint.get("last_optimizer_state") is None
        or not history
    ):
        raise RuntimeError(
            "Expected a full checkpoint at a completed epoch boundary. "
            "Start this experiment in a fresh checkpoint directory."
        )

    valid_loss = extra.get("val_loss")

    if valid_loss is None or not math.isfinite(float(valid_loss)):
        raise RuntimeError(
            "Checkpoint has no finite boundary validation loss"
        )

    return (
        int(iterator["epoch"]),
        int(history[-1]["num_updates"]),
    )


def train_one_chunk(
    chunk_id: int,
    pass_id: int,
    total_chunks: int,
    cfg: TrainingPipelineCfg,
    profile: str,
) -> None:
    check_dir(cfg.checkpoint_dir)

    data_dir = cfg.data_dir.resolve()
    label_dir = cfg.label_dir.resolve()
    checkpoint_dir = cfg.checkpoint_dir.resolve()

    fairseq_root = cfg.fairseq_root.resolve()
    hubert_cfg_dir = cfg.hubert_cfg_dir.resolve()
    hubert_dir = cfg.hubert_dir.resolve()

    hydra_train = fairseq_root / "fairseq_cli/hydra_train.py"
    checkpoint_last = checkpoint_dir / "checkpoint_last.pt"

    target_epoch = (pass_id - 1) * total_chunks + chunk_id
    next_epoch, start_update = get_checkpoint_position(checkpoint_last)

    if next_epoch != target_epoch:
        raise RuntimeError(
            f"Checkpoint expects epoch {next_epoch}, "
            f"but requested epoch {target_epoch}. "
            "Check --start_chunk, --start_pass, and total_chunks."
        )

    if profile == "gpu":
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            "--nproc_per_node=2",
            "--master_addr=127.0.0.1",
            "--master_port=29500",
            str(hydra_train),
        ]
    else:
        cmd = [
            sys.executable,
            "-u",
            str(hydra_train),
        ]

    cmd.extend([
        "--config-dir",
        str(hubert_cfg_dir),
        "--config-name",
        cfg.hubert_cfg,

        f"task.data={data_dir}",
        f"task.label_dir={label_dir}",
        'task.labels=["km"]',
        f"model.label_rate={cfg.label_rate}",
        "task.min_sample_size=null",

        f"checkpoint.save_dir={checkpoint_dir}",
        f"optimization.max_epoch={target_epoch}",
        "optimization.max_update=0",
        f"optimization.lr=[{cfg.learning_rate}]",
        "optimization.stop_time_hours=0",
        "optimization.stop_min_lr=-1",
        "optimization.skip_remainder_batch=false",
        f"+lr_scheduler.total_num_update={cfg.lr_total_updates}",
        f"optimization.update_freq=[{cfg.update_freq}]",
        f"lr_scheduler.warmup_updates={cfg.warmup_updates}",

        "dataset.required_batch_size_multiple=1",
        "dataset.disable_validation=false",
        "dataset.valid_subset=valid",
        "dataset.validate_interval=1",
        "dataset.validate_interval_updates=0",
        "dataset.validate_after_updates=0",
        "dataset.fixed_validation_seed=1337",
        "dataset.skip_invalid_size_inputs_valid_test=false",

        "checkpoint.save_interval=1",
        "checkpoint.save_interval_updates=0",
        "checkpoint.no_save=false",
        "checkpoint.no_save_optimizer_state=false",
        "checkpoint.no_last_checkpoints=false",
        "checkpoint.write_checkpoints_asynchronously=false",
        "checkpoint.patience=-1",
        "checkpoint.reset_optimizer=false",
        "checkpoint.reset_lr_scheduler=false",
        "checkpoint.reset_dataloader=false",

        "common.log_interval=1",
        f"common.wandb_project={cfg.wandb_project}",
        f"common.user_dir={hubert_dir}",
        "common.suppress_crashes=false",
        "hydra.run.dir=.",
    ])

    env_update = add_train_profile(cmd, profile, cfg)

    if checkpoint_last.exists():
        cmd.append(
            f"checkpoint.restore_file={checkpoint_last}"
        )
    else:
        print(
            "no checkpoint_last.pt found, starting fresh",
            flush=True,
        )

    env = os.environ.copy()
    env.update(env_update)
    env.update({
        "WANDB_RUN_GROUP": cfg.wandb_group,
        "WANDB_RUN_ID": uuid.uuid4().hex[:12],
        "WANDB_RESUME": "never",
        "WANDB_NAME": f"pass-{pass_id:02d}-chunk-{chunk_id:04d}",
    })

    print()
    print("cmd:", flush=True)
    print(" ".join(cmd), flush=True)
    print(flush=True)

    subprocess.run(
        cmd,
        env=env,
        check=True,
    )

    next_epoch, end_update = get_checkpoint_position(checkpoint_last)

    if next_epoch != target_epoch + 1 or end_update <= start_update:
        raise RuntimeError(
            "Fairseq did not complete exactly one chunk epoch"
        )

    print(
        f"chunk {chunk_id} finished | "
        f"updates {start_update} -> {end_update}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--pass-id", type=int, default=1)
    parser.add_argument("--total-chunks", type=int, default=None)
    parser.add_argument("--profile", choices=["cpu", "gpu"], default="cpu")

    args = parser.parse_args()
    cfg = TrainingPipelineCfg()

    total_chunks = (
        cfg.total_chunks
        if args.total_chunks is None
        else args.total_chunks
    )

    train_one_chunk(
        chunk_id=args.chunk_id,
        pass_id=args.pass_id,
        total_chunks=total_chunks,
        cfg=cfg,
        profile=args.profile,
    )


if __name__ == "__main__":
    main()

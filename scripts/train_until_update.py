import argparse
import json
import os
import re
import subprocess
import sys

import torch

from scripts.config import TrainingPipelineCfg
from scripts.utils import check_dir


TRAIN_LOSS_PATTERN = re.compile(
    r'"train_loss":\s*"?(?P<loss>[-+0-9.eE]+)'
)


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


def get_checkpoint_updates(checkpoint_path) -> int:
    if not checkpoint_path.exists():
        return 0

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    history = checkpoint.get("optimizer_history", [])

    if not history:
        return 0

    return int(history[-1].get("num_updates", 0))


def run_train_step(
    target_update: int,
    cfg: TrainingPipelineCfg,
    profile: str,
) -> float:
    check_dir(cfg.checkpoint_dir)

    data_dir = cfg.data_dir.resolve()
    label_dir = cfg.label_dir.resolve()
    checkpoint_dir = cfg.checkpoint_dir.resolve()

    fairseq_root = cfg.fairseq_root.resolve()
    hubert_cfg_dir = cfg.hubert_cfg_dir.resolve()
    hubert_dir = cfg.hubert_dir.resolve()

    hydra_train = fairseq_root / "fairseq_cli/hydra_train.py"
    checkpoint_last = checkpoint_dir / "checkpoint_last.pt"

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

        f"checkpoint.save_dir={checkpoint_dir}",
        f"optimization.max_update={target_update}",
        "dataset.disable_validation=true",

        f"common.user_dir={hubert_dir}",

        "hydra.run.dir=.",
    ])

    env_update = add_train_profile(cmd, profile, cfg)

    if checkpoint_last.exists():
        cmd.extend([
            f"checkpoint.restore_file={checkpoint_last}",
            "checkpoint.reset_dataloader=true",
        ])
    else:
        print(
            "no checkpoint_last.pt found, starting fresh",
            flush=True,
        )

    env = os.environ.copy()
    env.update(env_update)

    print()
    print("cmd:", flush=True)
    print(" ".join(cmd), flush=True)
    print(flush=True)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    train_loss = None

    if process.stdout is None:
        raise RuntimeError(
            "could not read Fairseq output"
        )

    for line in process.stdout:
        print(line, end="", flush=True)

        match = TRAIN_LOSS_PATTERN.search(line)

        if match:
            train_loss = float(
                match.group("loss")
            )

    return_code = process.wait()

    if return_code != 0:
        raise subprocess.CalledProcessError(
            return_code,
            cmd,
        )

    if train_loss is None:
        raise RuntimeError(
            "Fairseq finished training but "
            "no train_loss was found"
        )

    return train_loss


def get_progress_path(
    cfg: TrainingPipelineCfg,
):
    return (
        cfg.checkpoint_dir
        / "training_progress.json"
    )


def save_progress(
    chunk_id: int,
    start_update: int,
    current_update: int,
    segment_losses: list[float],
    segment_updates: list[int],
    baseline_loss: float | None,
    cfg: TrainingPipelineCfg,
) -> None:
    progress_path = get_progress_path(cfg)

    progress = {
        "chunk_id": chunk_id,
        "start_update": start_update,
        "current_update": current_update,
        "baseline_loss": baseline_loss,
        "segment_losses": segment_losses,
        "segment_updates": segment_updates,
    }

    tmp_path = progress_path.with_suffix(
        ".tmp"
    )

    with tmp_path.open("w") as file:
        json.dump(
            progress,
            file,
            indent=2,
        )

    tmp_path.replace(progress_path)


def load_progress(
    chunk_id: int,
    cfg: TrainingPipelineCfg,
):
    progress_path = get_progress_path(cfg)

    if not progress_path.exists():
        return None

    with progress_path.open("r") as file:
        progress = json.load(file)

    if progress.get("chunk_id") != chunk_id:
        return None

    return progress


def clear_progress(
    chunk_id: int,
    cfg: TrainingPipelineCfg,
) -> None:
    progress_path = get_progress_path(cfg)

    if not progress_path.exists():
        return

    with progress_path.open("r") as file:
        progress = json.load(file)

    if progress.get("chunk_id") == chunk_id:
        progress_path.unlink()


def save_metrics(
    chunk_id: int,
    checkpoint_name: str,
    baseline_loss: float,
    final_loss: float,
    segment_losses: list[float],
    segment_updates: list[int],
    start_update: int,
    end_update: int,
    stop_reason: str,
    cfg: TrainingPipelineCfg,
) -> None:
    metrics_path = (
        cfg.checkpoint_dir
        / "training_metrics.json"
    )

    loss_drop = (
        baseline_loss - final_loss
    ) / baseline_loss

    loss_history = [
        {
            "update": update,
            "loss": loss,
        }
        for update, loss in zip(
            segment_updates,
            segment_losses,
        )
    ]

    record = {
        "chunk_id": chunk_id,
        "checkpoint": checkpoint_name,

        "baseline_loss": baseline_loss,
        "final_rolling_loss": final_loss,

        "loss_drop_percent":
            loss_drop * 100,

        "updates_this_chunk":
            end_update - start_update,

        "cumulative_updates":
            end_update,

        "loss_target_percent":
            cfg.loss_drop_target * 100,

        "loss_step_updates":
            cfg.loss_step_updates,

        "loss_window_steps":
            cfg.loss_window_steps,

        "target_reached":
            loss_drop >= cfg.loss_drop_target,

        "stop_reason":
            stop_reason,

        "segment_losses":
            segment_losses,

        "loss_history":
            loss_history,
    }

    if metrics_path.exists():
        with metrics_path.open("r") as file:
            metrics = json.load(file)
    else:
        metrics = []

    metrics = [
        item
        for item in metrics
        if item["chunk_id"] != chunk_id
    ]

    metrics.append(record)

    metrics.sort(
        key=lambda item: item["chunk_id"]
    )

    with metrics_path.open("w") as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    print()
    print(
        f"saved training metrics: "
        f"{metrics_path}",
        flush=True,
    )


def train_until_loss(
    chunk_id: int,
    cfg: TrainingPipelineCfg,
    profile: str = "cpu",
) -> None:
    checkpoint_last = (
        cfg.checkpoint_dir
        / "checkpoint_last.pt"
    )

    checkpoint_update = (
        get_checkpoint_updates(
            checkpoint_last
        )
    )

    progress = load_progress(
        chunk_id,
        cfg,
    )

    if progress is not None:
        start_update = int(
            progress["start_update"]
        )

        current_update = (
            checkpoint_update
        )

        segment_losses = [
            float(loss)
            for loss in progress[
                "segment_losses"
            ]
        ]

        segment_updates = [
            int(update)
            for update in progress[
                "segment_updates"
            ]
        ]

        baseline_loss = progress.get(
            "baseline_loss"
        )

        if baseline_loss is not None:
            baseline_loss = float(
                baseline_loss
            )

        print()
        print(
            f"resuming chunk {chunk_id}",
            flush=True,
        )
        print(
            f"original chunk start: "
            f"{start_update}",
            flush=True,
        )
        print(
            f"checkpoint update: "
            f"{current_update}",
            flush=True,
        )
        print(
            f"restored "
            f"{len(segment_losses)} "
            f"loss measurements",
            flush=True,
        )

        if baseline_loss is not None:
            print(
                f"restored baseline: "
                f"{baseline_loss:.4f}",
                flush=True,
            )

    else:
        start_update = checkpoint_update
        current_update = start_update

        segment_losses = []
        segment_updates = []
        baseline_loss = None

        print()
        print(
            f"starting chunk {chunk_id} "
            f"at update {start_update}",
            flush=True,
        )

        save_progress(
            chunk_id=chunk_id,
            start_update=start_update,
            current_update=current_update,
            segment_losses=segment_losses,
            segment_updates=segment_updates,
            baseline_loss=baseline_loss,
            cfg=cfg,
        )

    final_loss = None
    stop_reason = None

    print(
        f"target loss reduction: "
        f"{cfg.loss_drop_target * 100:.1f}%",
        flush=True,
    )

    print(
        f"maximum updates this chunk: "
        f"{cfg.max_updates_per_chunk}",
        flush=True,
    )

    while True:
        updates_this_chunk = (
            current_update
            - start_update
        )

        if (
            updates_this_chunk
            >= cfg.max_updates_per_chunk
        ):
            stop_reason = "max_updates"

            if (
                len(segment_losses)
                >= cfg.loss_window_steps
            ):
                final_loss = sum(
                    segment_losses[
                        -cfg.loss_window_steps:
                    ]
                ) / cfg.loss_window_steps
            else:
                final_loss = (
                    segment_losses[-1]
                )

            break

        target_update = min(
            current_update
            + cfg.loss_step_updates,

            start_update
            + cfg.max_updates_per_chunk,
        )

        loss = run_train_step(
            target_update=target_update,
            cfg=cfg,
            profile=profile,
        )

        current_update = (
            get_checkpoint_updates(
                checkpoint_last
            )
        )

        segment_losses.append(loss)
        segment_updates.append(
            current_update
        )

        print()
        print(
            f"chunk {chunk_id} | "
            f"update {current_update} | "
            f"loss {loss:.4f}",
            flush=True,
        )

        if (
            baseline_loss is None
            and len(segment_losses)
            >= cfg.loss_window_steps
        ):
            baseline_loss = sum(
                segment_losses[
                    :cfg.loss_window_steps
                ]
            ) / cfg.loss_window_steps

            print(
                f"baseline rolling loss: "
                f"{baseline_loss:.4f}",
                flush=True,
            )

        save_progress(
            chunk_id=chunk_id,
            start_update=start_update,
            current_update=current_update,
            segment_losses=segment_losses,
            segment_updates=segment_updates,
            baseline_loss=baseline_loss,
            cfg=cfg,
        )

        if baseline_loss is None:
            continue

        rolling_loss = sum(
            segment_losses[
                -cfg.loss_window_steps:
            ]
        ) / cfg.loss_window_steps

        updates_this_chunk = (
            current_update
            - start_update
        )

        loss_drop = (
            baseline_loss
            - rolling_loss
        ) / baseline_loss

        print(
            f"rolling loss: "
            f"{rolling_loss:.4f} | "
            f"drop: "
            f"{loss_drop * 100:.2f}%",
            flush=True,
        )

        if (
            updates_this_chunk
            >= cfg.min_updates_per_chunk
            and loss_drop
            >= cfg.loss_drop_target
        ):
            final_loss = rolling_loss
            stop_reason = "loss_target"
            break

    if baseline_loss is None:
        raise RuntimeError(
            f"chunk {chunk_id} ended before "
            "a baseline loss could be "
            "established"
        )

    if final_loss is None:
        raise RuntimeError(
            f"chunk {chunk_id} ended "
            "without a final loss"
        )

    checkpoint_name = (
        f"checkpoint_chunk_"
        f"{chunk_id:04d}.pt"
    )

    save_metrics(
        chunk_id=chunk_id,
        checkpoint_name=checkpoint_name,
        baseline_loss=baseline_loss,
        final_loss=final_loss,
        segment_losses=segment_losses,
        segment_updates=segment_updates,
        start_update=start_update,
        end_update=current_update,
        stop_reason=stop_reason,
        cfg=cfg,
    )

    clear_progress(
        chunk_id,
        cfg,
    )

    loss_drop = (
        baseline_loss
        - final_loss
    ) / baseline_loss

    print()

    if stop_reason == "loss_target":
        print(
            f"chunk {chunk_id} reached "
            f"{loss_drop * 100:.2f}% "
            f"loss reduction",
            flush=True,
        )

    elif stop_reason == "max_updates":
        print(
            f"chunk {chunk_id} reached the "
            f"{cfg.max_updates_per_chunk} "
            f"update limit with "
            f"{loss_drop * 100:.2f}% "
            f"loss reduction",
            flush=True,
        )

        print(
            "saving checkpoint and "
            "continuing to the next chunk",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--chunk-id",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--profile",
        choices=["cpu", "gpu"],
        default="cpu",
    )

    args = parser.parse_args()
    cfg = TrainingPipelineCfg()

    train_until_loss(
        chunk_id=args.chunk_id,
        cfg=cfg,
        profile=args.profile,
    )


if __name__ == "__main__":
    main()
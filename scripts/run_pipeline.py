import argparse
import os
import sys

from scripts.config import TrainingPipelineCfg
from scripts.utils import cmd_exec

def save_developmental_checkpoint(
    checkpoint_last,
    checkpoint_age,
) -> None:
    import torch

    checkpoint = torch.load(
        checkpoint_last,
        map_location="cpu",
        weights_only=False,
    )

    checkpoint["last_optimizer_state"] = None

    torch.save(
        checkpoint,
        checkpoint_age,
    )

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--start_chunk", type=int, default=1)
    parser.add_argument("--total_chunks", type=int, default=None)

    args = parser.parse_args()
    cfg = TrainingPipelineCfg()

    total_chunks = (
        cfg.total_chunks
        if args.total_chunks is None
        else args.total_chunks
    )

    if not cfg.kmeans_path.exists():
        print("no kmeans model found, learning it", flush=True)

        cmd_exec([
            "python",
            "-m",
            "scripts.learn_kmeans_once",
            "--seed_chunk_id",
            "1",
        ])
    else:
        print("found kmeans model, reusing it", flush=True)

    valid_tsv = cfg.data_dir / "valid.tsv"
    valid_km = cfg.label_dir / "valid.km"

    if not valid_tsv.exists():
        cmd_exec([
            "python",
            "-m",
            "scripts.prep_chunk",
            "--mode",
            "valid",
        ])
    else:
        print("found valid.tsv, reusing it", flush=True)

    if not valid_km.exists():
        cmd_exec([
            "python",
            "-m",
            "scripts.make_km_labels",
            "--split",
            "valid",
        ])
    else:
        print("found valid.km, reusing it", flush=True)

    for chunk_id in range(args.start_chunk, total_chunks + 1):
        print()
        print(
            f"starting chunk {chunk_id}/{total_chunks}",
            flush=True,
        )

        cmd_exec([
            "python",
            "-m",
            "scripts.prep_chunk",
            "--mode",
            "train",
            "--chunk_id",
            str(chunk_id),
        ])

        cmd_exec([
            "python",
            "-m",
            "scripts.make_km_labels",
            "--split",
            "train",
        ])

        cmd_exec([
            "python",
            "-m",
            "scripts.train_until_update",
            "--chunk-id",
            str(chunk_id),
            "--profile",
            args.profile,
        ])

        checkpoint_last = (
            cfg.checkpoint_dir / "checkpoint_last.pt"
        )

        checkpoint_age = (
            cfg.checkpoint_dir
            / f"checkpoint_chunk_{chunk_id:04d}.pt"
        )

        if not checkpoint_last.exists():
            raise FileNotFoundError(
                f"Training finished, but "
                f"{checkpoint_last} was not created"
            )

        save_developmental_checkpoint(
            checkpoint_last,
            checkpoint_age,
        )

        print(
            f"saved developmental checkpoint: "
            f"{checkpoint_age}",
            flush=True,
        )

        print(
            f"finished chunk {chunk_id}",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        sys.stdout.flush()
        sys.stderr.flush()
        raise

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
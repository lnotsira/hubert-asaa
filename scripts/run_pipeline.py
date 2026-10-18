import argparse
import os
import sys

from scripts.config import TrainingPipelineCfg
from scripts.utils import cmd_exec
from scripts.train_until_update import get_checkpoint_position

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
    parser.add_argument("--start_pass", type=int, default=1)
    parser.add_argument("--total_chunks", type=int, default=None)

    args = parser.parse_args()
    cfg = TrainingPipelineCfg()

    total_chunks = (
        cfg.total_chunks
        if args.total_chunks is None
        else args.total_chunks
    )

    if (
        total_chunks < 1
        or not 1 <= args.start_chunk <= total_chunks
        or not 1 <= args.start_pass <= cfg.passes
    ):
        raise ValueError("Invalid chunk or pass range")

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

    for pass_id in range(args.start_pass, cfg.passes + 1):
        first_chunk = args.start_chunk if pass_id == args.start_pass else 1

        for chunk_id in range(first_chunk, total_chunks + 1):
            print()
            print(
                f"starting pass {pass_id}, chunk {chunk_id}/{total_chunks}",
                flush=True,
            )

            target_epoch = (pass_id - 1) * total_chunks + chunk_id
            next_epoch, _ = get_checkpoint_position(
                cfg.checkpoint_dir / "checkpoint_last.pt"
            )

            if next_epoch != target_epoch:
                raise RuntimeError(
                    f"Checkpoint expects epoch {next_epoch}; "
                    "set --start_pass and --start_chunk to the next chunk."
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
                "--pass-id",
                str(pass_id),
                "--total-chunks",
                str(total_chunks),
                "--profile",
                args.profile,
            ])

            checkpoint_last = (
                cfg.checkpoint_dir / "checkpoint_last.pt"
            )

            checkpoint_name = (
                f"checkpoint_chunk_{chunk_id:04d}.pt"
                if pass_id == 1
                else f"checkpoint_pass_{pass_id:02d}_chunk_{chunk_id:04d}.pt"
            )

            checkpoint_age = cfg.checkpoint_dir / checkpoint_name

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
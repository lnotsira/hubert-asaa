import argparse
import os
import sys

from pathlib import Path

import soundfile as sf
from datasets import Audio
from datasets import load_dataset as hf_load_dataset

from scripts.config import TrainingPipelineCfg
from scripts.utils import check_dir, wipe_dir


def load_hf_dataset(
    cfg: TrainingPipelineCfg,
    split: str,
):
    ds = hf_load_dataset(
        cfg.dataset_name,
        cfg.dataset_subset,
        split=split,
        streaming=True,
    )

    ds = ds.cast_column(
        cfg.audio_column,
        Audio(sampling_rate=cfg.sample_rate),
    )

    return ds


def make_audio_filename(
    split: str,
    chunk_id: int,
    index: int,
    extension: str,
) -> str:
    if split == "valid":
        return f"valid_file_{index:08d}.{extension}"

    return (
        f"chunk_{chunk_id:06d}_"
        f"file_{index:08d}.{extension}"
    )


def write_audio_file(
    audio_path: Path,
    arr,
    cfg: TrainingPipelineCfg,
) -> None:
    extension = cfg.audio_format

    if extension == "flac":
        sf.write(
            audio_path,
            arr,
            cfg.sample_rate,
            format="FLAC",
        )

    elif extension == "wav":
        sf.write(
            audio_path,
            arr,
            cfg.sample_rate,
            format="WAV",
        )

    else:
        raise ValueError(
            f"unsupported audio format: {extension}"
        )


def make_manifest(
    stream,
    split: str,
    audio_dir: Path,
    manifest_path: Path,
    cfg: TrainingPipelineCfg,
    limit: int,
    chunk_id: int = 0,
) -> None:
    check_dir(manifest_path.parent)
    wipe_dir(audio_dir)

    extension = cfg.audio_format
    saved_count = 0

    with manifest_path.open("w") as manifest:
        manifest.write(
            str(audio_dir.resolve()) + "\n"
        )

        for row in stream:
            if saved_count >= limit:
                break

            audio = row[cfg.audio_column]
            arr = audio["array"]

            audio_name = make_audio_filename(
                split=split,
                chunk_id=chunk_id,
                index=saved_count,
                extension=extension,
            )

            audio_path = audio_dir / audio_name

            write_audio_file(
                audio_path,
                arr,
                cfg,
            )

            manifest.write(
                f"{audio_name}\t{len(arr)}\n"
            )

            saved_count += 1

            if saved_count % 100 == 0:
                print(
                    f"saved {saved_count}/{limit}",
                    flush=True,
                )

    if saved_count == 0:
        raise RuntimeError(
            f"no audio files were written for {split}"
        )

    print(
        f"finished {split}: {saved_count} files",
        flush=True,
    )


def prep_train_chunk(
    chunk_id: int,
    cfg: TrainingPipelineCfg,
) -> None:
    ds = load_hf_dataset(
        cfg,
        cfg.train_split,
    )

    ds = ds.shuffle(
        seed=cfg.shuffle_seed,
        buffer_size=cfg.shuffle_buffer_size,
    )

    start_idx = (
        (chunk_id - 1)
        * cfg.chunk_size
    )

    stream = (
        ds
        .skip(start_idx)
        .take(cfg.chunk_size)
    )

    make_manifest(
        stream=stream,
        split="train",
        audio_dir=cfg.train_audio_dir,
        manifest_path=cfg.data_dir / "train.tsv",
        cfg=cfg,
        limit=cfg.chunk_size,
        chunk_id=chunk_id,
    )


def prep_valid(
    cfg: TrainingPipelineCfg,
) -> None:
    ds = load_hf_dataset(
        cfg,
        cfg.valid_split,
    )

    stream = ds.take(
        cfg.valid_size
    )

    make_manifest(
        stream=stream,
        split="valid",
        audio_dir=cfg.valid_audio_dir,
        manifest_path=cfg.data_dir / "valid.tsv",
        cfg=cfg,
        limit=cfg.valid_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["train", "valid"],
        required=True,
    )
    parser.add_argument(
        "--chunk_id",
        type=int,
        default=1,
    )

    args = parser.parse_args()
    cfg = TrainingPipelineCfg()

    if args.mode == "train":
        prep_train_chunk(
            args.chunk_id,
            cfg,
        )
    else:
        prep_valid(cfg)


if __name__ == "__main__":
    main()

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
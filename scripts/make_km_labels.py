import argparse
import shutil

from pathlib import Path

from scripts.config import TrainingPipelineCfg
from scripts.utils import check_dir, wipe_dir, cmd_exec

def fix_filename(label_dir: str | Path, split: str) -> Path:
    label_dir = Path(label_dir)

    expected = label_dir / f"{split}.km"

    files = sorted(label_dir.glob(f"{split}_*.km"))

    if expected.exists():
        expected.unlink()

    if len(files) == 1:
        shutil.move(str(files[0]), str(expected))
        print(f"[labels] renamed {files[0].name} -> {expected.name}")
        return expected

    if len(files) == 0:
        raise FileNotFoundError(
            f"No generated label file found for split={split}. "
            f"Expecting {split}_0_1.km in {label_dir}"
        )

    raise RuntimeError(
        f"Multiple label files found for split={split}: "
        f"{[p.name for p in files]}. "
    )

def write_dict(cfg: TrainingPipelineCfg) -> None:
    check_dir(cfg.label_dir)

    dict_path = cfg.label_dir.joinpath("dict.km.txt")

    with open(dict_path, "w") as f:
        for i in range(cfg.clusters):
            f.write(f"{i} 1\n")

def make_km_labels(split: str, cfg: TrainingPipelineCfg) -> None:
    check_dir(cfg.label_dir)
    wipe_dir(cfg.feat_dir)

    dump_mfcc = cfg.fairseq_root.joinpath(
        "examples/hubert/simple_kmeans/dump_mfcc_feature.py"
    )
    dump_km = cfg.fairseq_root.joinpath(
        "examples/hubert/simple_kmeans/dump_km_label.py"
    )

    cmd_exec([
        "python",
        str(dump_mfcc),
        str(cfg.data_dir),
        split,
        "1",
        "0",
        str(cfg.feat_dir),
    ])

    cmd_exec([
        "python",
        str(dump_km),
        str(cfg.feat_dir),
        split,
        str(cfg.kmeans_path),
        "1",
        "0",
        str(cfg.label_dir),
    ])

    write_dict(cfg)

    fix_filename(cfg.label_dir, split)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "valid"], required=True)

    args = parser.parse_args()
    cfg = TrainingPipelineCfg()

    make_km_labels(args.split, cfg)


if __name__ == "__main__":
    main()

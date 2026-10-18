import argparse

from scripts.config import TrainingPipelineCfg
from scripts.prep_chunk import prep_train_chunk
from scripts.utils import check_dir, wipe_dir, cmd_exec


def learn_kmeans(seed_chunk_id: int, cfg: TrainingPipelineCfg) -> None:
    print("1")
    check_dir(cfg.kmeans_path.parent)
    print("2")
    seed_feat_dir = cfg.feat_dir.parent.joinpath("km_seed")
    print("3")
    wipe_dir(seed_feat_dir)
    print("4")
    prep_train_chunk(seed_chunk_id, cfg)

    print("5")
    dump_mfcc = cfg.fairseq_root.joinpath(
        "examples/hubert/simple_kmeans/dump_mfcc_feature.py"
    )
    print("6")
    learn_kmeans_script = cfg.fairseq_root.joinpath(
        "examples/hubert/simple_kmeans/learn_kmeans.py"
    )

    cmd_exec([
        "python",
        str(dump_mfcc),
        str(cfg.data_dir),
        "train",
        "1",
        "0",
        str(seed_feat_dir),
    ])

    cmd_exec([
        "python",
        str(learn_kmeans_script),
        str(seed_feat_dir),
        "train",
        "1",
        str(cfg.kmeans_path),
        str(cfg.clusters),
        "--percent",
        "0.1",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_chunk_id", type=int, default=1)

    args = parser.parse_args()
    cfg = TrainingPipelineCfg()

    learn_kmeans(args.seed_chunk_id, cfg)


if __name__ == "__main__":
    main()

from dataclasses import dataclass
from pathlib import Path

@dataclass
class TrainingPipelineCfg:

    # HuggingFace
    dataset_name: str = "disco-eth/WorldSpeech"
    dataset_subset: str = "en_us"
    train_split: str = "train"
    valid_split: str = "test"
    audio_column: str = "audio"
    sample_rate: int = 16000

    audio_format: str = "flac"

    # Chunks
    chunk_size: int = 10000
    total_chunks: int = 66
    valid_size: int = 5000

    # Chunk Seeding
    shuffle_seed: int = 42
    shuffle_buffer_size: int = 100

    # Training passes and logging
    passes: int = 1
    wandb_project: str = "hubert-development"
    wandb_group: str = "hubert-one-pass"
    lr_total_updates: int = 25000
    learning_rate: float = 0.0001
    update_freq: int = 16
    warmup_updates: int = 2000

    # Paths
    data_dir: Path = Path("data")
    train_audio_dir: Path = Path("data/audio/train")
    valid_audio_dir: Path = Path("data/audio/valid")
    label_dir: Path = Path("data/labels")
    feat_dir: Path = Path("feats/")
    kmeans_path: Path = Path("kmeans/km.bin")
    checkpoint_dir: Path = Path("checkpoints_one_pass")

    fairseq_root: Path = Path("external/fairseq")
    hubert_cfg_dir: Path = Path("external/fairseq/examples/hubert/config/pretrain")
    hubert_cfg: str = "hubert_base_librispeech"
    hubert_dir: Path = Path("external/fairseq/examples/hubert")

    # HuBERT
    clusters: int = 100
    label_rate: int = 100

    # Training
    cpu_max_tokens: int = 320000
    cpu_num_workers: int = 0

    a30_max_tokens: int = 320000
    a30_num_workers: int = 1
    a30_cuda_devices: str = "0,1"

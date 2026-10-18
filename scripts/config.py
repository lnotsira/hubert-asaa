from dataclasses import dataclass
from pathlib import Path

@dataclass
class TrainingPipelineCfg:
    #
    # Central config for the HuBERT pretraining pipeline.
    #
    # Idea is that we can easily mess around with things like changing the
    # dataset and subset, switching a path somewhere else, and also so I don't
    # have to manually retype everything that's defined here in other files.
    # - aris.
    #

    # HuggingFace
    dataset_name: str = "disco-eth/WorldSpeech"
    dataset_subset: str = "en_us"
    train_split: str = "train"
    valid_split: str = "test" # WorldSpeech already has a split dedicated to
                              # validation so we'll use it here.
    audio_column: str = "audio"
    sample_rate: int = 16000

    #
    # fairseq supports both wav and flac, however I decided that since we're
    # trying to save on storage anyways, we'll opt for using FLAC since it is
    # a lossless format rather than a totally uncompressed format like WAV (but
    # I'm keeping this here just in case something decides to break on us)
    #
    audio_format: str = "flac"

    # Chunks
    chunk_size: int = 10000 # Number of audio clips per chunk
    total_chunks: int = 66 # Total number of chunks the pipeline should process
    valid_size: int = 5000

    # Chunk Seeding
    shuffle_seed: int = 42
    shuffle_buffer_size: int = 100

    # Loss-based training
    loss_drop_target: float = 0.05
    loss_step_updates: int = 100
    loss_window_steps: int = 3
    min_updates_per_chunk: int = 300
    max_updates_per_chunk: int = 5000
    
    # Paths
    data_dir: Path = Path("data")
    train_audio_dir: Path = Path("data/audio/train")
    valid_audio_dir: Path = Path("data/audio/valid")
    label_dir: Path = Path("data/labels")
    feat_dir: Path = Path("feats/")
    kmeans_path: Path = Path("kmeans/km.bin")
    checkpoint_dir: Path = Path("checkpoints")

    fairseq_root: Path = Path("external/fairseq")
    hubert_cfg_dir: Path = Path("external/fairseq/examples/hubert/config/pretrain")
    hubert_cfg: str = "hubert_base_librispeech"
    hubert_dir: Path = Path("external/fairseq/examples/hubert")

    # HuBERT
    clusters: int = 100 # number of kmeans clusters
    label_rate: int = 100

    # Training
    # cpu is mainly for smoke tests. a30 optimized for the jupyter environment.
    cpu_max_tokens: int = 320000
    cpu_num_workers: int = 0

    a30_max_tokens: int = 320000
    a30_num_workers: int = 1
    a30_cuda_devices: str = "0,1"

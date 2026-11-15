import argparse
import csv
import json

import torch

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

if __package__:
    from . import extract_activations as base
else:
    import extract_activations as base

def read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8-sig") as manifest:
        reader = csv.DictReader(manifest)
        required_columns = {"stimulus", "step", "wav_path"}

        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                "Manifest requires stimulus, step, and wav_path columns"
            )

        rows = list(reader)

    steps = sorted(int(row["step"]) for row in rows)

    if len(rows) != 7 or steps != list(range(1, 8)):
        raise ValueError(
            "Expected exactly seven stimuli with steps 1 through 7"
        )

    names = [row["stimulus"] for row in rows]

    if len(set(names)) != 7:
        raise ValueError("Stimulus names must be unique")

    for name in names:
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"Invalid stimulus filename: {name}")

    return rows

def load_audio(wav_path: Path) -> torch.Tensor:
    info = sf.info(wav_path)

    if info.samplerate != 16000 or info.channels != 1:
        raise ValueError(f"Expected mono 16 kHz audio: {wav_path}")

    waveform, _, _ = base.load_audio(wav_path)
    return waveform

def get_frame_timing(model: torch.nn.Module, num_samples: int) -> tuple[int, int, int]:
    num_frames = num_samples
    frame_stride = 1
    receptive_field = 1

    for block in model.feature_extractor.conv_layers:
        convolutions = [
            module
            for module in block.modules()
            if isinstance(module, torch.nn.Conv1d)
        ]

        if len(convolutions) != 1:
            raise ValueError(
                "Expected one convolution per feature-extractor block"
            )

        convolution = convolutions[0]
        kernel_size = convolution.kernel_size[0]
        stride = convolution.stride[0]
        dilation = convolution.dilation[0]
        padding = convolution.padding[0]

        if padding != 0:
            raise ValueError(
                "Timing currently requires unpadded feature convolutions"
            )

        kernel_width = dilation * (kernel_size - 1)
        num_frames = (num_frames - kernel_width - 1) // stride + 1
        receptive_field += kernel_width * frame_stride
        frame_stride *= stride

    if num_frames <= 0:
        raise ValueError("Audio is too short for the feature extractor")

    return num_frames, frame_stride, receptive_field

@torch.inference_mode()
def extract_vot_activations(model: torch.nn.Module, waveform: torch.Tensor, normalize: bool) -> tuple[torch.Tensor, int, int]:
    num_frames, frame_stride, receptive_field = get_frame_timing(
        model,
        waveform.numel(),
    )

    source = base.prepare_model_input(
        waveform=waveform,
        normalize=normalize,
        device=torch.device("cpu"),
    )

    requested_layers = base.parse_requested_layers(
        "all",
        base.get_encoder_layer_count(model),
    )

    _, layer_activations = base.extract_activations(
        model=model,
        source=source,
        requested_layers=requested_layers,
        storage_dtype=torch.float32,
    )

    hidden_states = torch.stack([
        layer_activations[layer_number]
        for layer_number in requested_layers
    ])

    if (tuple(hidden_states.shape) != (12, num_frames, 768) or not torch.isfinite(hidden_states).all()):
        raise ValueError(f"Invalid hidden states: {tuple(hidden_states.shape)}")

    return hidden_states, frame_stride, receptive_field

def make_metadata(
    row: dict[str, str],
    manifest_path: Path,
    wav_path: Path,
    checkpoint_path: Path,
    num_samples: int,
    normalize: bool,
    hidden_states: torch.Tensor,
    frame_stride: int,
    receptive_field: int,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "manifest_row": row,
        "manifest": str(manifest_path),
        "wav_path": str(wav_path),
        "checkpoint": str(checkpoint_path.resolve()),
        "sample_rate": 16000,
        "num_samples": num_samples,
        "duration_s": num_samples / 16000,
        "normalized": normalize,
        "mask": False,
        "device": "cpu",
        "shape": list(hidden_states.shape),
        "axes": ["layer", "frame", "channel"],
        "hidden_state_definition": (
            "Fairseq extract_features output for each output_layer, using extract_activations.py"
        ),
        "frame_stride_samples": frame_stride,
        "frame_receptive_field_samples": receptive_field,
        "timing_definition": (
            "convolution windows [start, end); centers use window midpoint; "
            "transformer context is global"
        ),
    }

def save_output(output_path: Path, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    if output_path.suffix == ".npz":
        np.savez_compressed(
            output_path,
            **arrays,
            metadata_json=json.dumps(metadata),
        )

        with np.load(output_path, allow_pickle=False) as saved:
            for key, value in arrays.items():
                np.testing.assert_array_equal(saved[key], value)

            if json.loads(str(saved["metadata_json"])) != metadata:
                raise ValueError(f"Saved metadata does not match: {output_path}")

    else:
        payload = {
            key: torch.from_numpy(value.copy())
            for key, value in arrays.items()
        }
        payload["metadata"] = metadata

        torch.save(payload, output_path)

        saved = torch.load(
            output_path,
            map_location="cpu",
            weights_only=True,
        )

        for key in arrays:
            torch.testing.assert_close(
                saved[key],
                payload[key],
                rtol=0,
                atol=0,
            )

        if saved["metadata"] != metadata:
            raise ValueError(f"Saved metadata does not match: {output_path}")

def extract_vot(checkpoint_path: Path, manifest_path: Path, output_dir: Path, output_format: str) -> None:
    manifest_path = manifest_path.resolve()
    rows = read_manifest(manifest_path)
    audio = []

    for row in rows:
        wav_path = manifest_path.parent.joinpath(row["wav_path"]).resolve()
        waveform = load_audio(wav_path)
        audio.append((row, wav_path, waveform))

    checkpoint_output_dir = output_dir / checkpoint_path.stem
    extensions = ["npz", "pt"] if output_format == "both" else [output_format]

    for row in rows:
        for extension in extensions:
            output_path = checkpoint_output_dir / f"{row['stimulus']}.{extension}"
            if output_path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite {output_path}; "
                    "choose a new --output-dir"
                )

    model, saved_cfg, task = base.load_hubert_model(
        checkpoint_path=checkpoint_path,
        device=torch.device("cpu"),
    )

    if base.get_encoder_layer_count(model) != 12:
        raise ValueError("Expected 12 transformer layers")

    task_cfg = getattr(saved_cfg, "task", getattr(task, "cfg", None))

    if getattr(task_cfg, "sample_rate", 16000) != 16000:
        raise ValueError("Checkpoint expects a different sample rate")

    normalize = base.checkpoint_uses_waveform_normalization(saved_cfg, task)
    checkpoint_output_dir.mkdir(parents=True, exist_ok=True)

    for row, wav_path, waveform in audio:
        hidden_states, frame_stride, receptive_field = extract_vot_activations(
            model=model,
            waveform=waveform,
            normalize=normalize,
        )

        frame_start = np.arange(hidden_states.shape[1], dtype=np.int64)
        frame_start *= frame_stride
        frame_end = frame_start + receptive_field

        metadata = make_metadata(
            row=row,
            manifest_path=manifest_path,
            wav_path=wav_path,
            checkpoint_path=checkpoint_path,
            num_samples=waveform.numel(),
            normalize=normalize,
            hidden_states=hidden_states,
            frame_stride=frame_stride,
            receptive_field=receptive_field,
        )

        arrays = {
            "hidden_states": hidden_states.numpy(),
            "layer_numbers": np.arange(1, 13),
            "frame_start_sample": frame_start,
            "frame_end_sample": frame_end,
            "frame_start_s": frame_start / 16000,
            "frame_end_s": frame_end / 16000,
            "frame_center_s": (frame_start + receptive_field / 2) / 16000,
        }

        for extension in extensions:
            output_path = checkpoint_output_dir / f"{row['stimulus']}.{extension}"

            save_output(
                output_path=output_path,
                arrays=arrays,
                metadata=metadata,
            )

        print(
            f"{row['stimulus']}: {tuple(hidden_states.shape)} "
            f"verified ({', '.join(extensions)})",
            flush=True,
        )

    print(
        f"saved seven stimuli to {checkpoint_output_dir.resolve()}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("Stimuli/manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("activations/vot"),
    )
    parser.add_argument(
        "--format",
        choices=["npz", "pt", "both"],
        default="both",
    )

    args = parser.parse_args()

    extract_vot(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        output_format=args.format,
    )

if __name__ == "__main__":
    main()

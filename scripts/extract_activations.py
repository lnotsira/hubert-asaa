from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import fairseq
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly


TARGET_SAMPLE_RATE = 16_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--wav-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("activations"))
    
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--layers", default="all")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16"),
        default="float32",
    )

    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> None:
    if not args.checkpoint_dir.is_dir():
        raise NotADirectoryError(f"Checkpoint directory not found: {args.checkpoint_dir}")

    if not args.wav_dir.is_dir():
        raise NotADirectoryError(f"WAV directory not found: {args.wav_dir}")


def load_audio(
    wav_path: Path,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
) -> tuple[torch.Tensor, int, int]:
    waveform, original_sample_rate = sf.read(
        wav_path,
        dtype="float32",
        always_2d=True,
    )

    waveform = waveform.mean(axis=1)

    if waveform.size == 0:
        raise ValueError(f"WAV file contains no samples: {wav_path}")

    if not np.isfinite(waveform).all():
        raise ValueError(f"WAV file contains NaN or infinite values: {wav_path}")

    if original_sample_rate != target_sample_rate:
        divisor = np.gcd(original_sample_rate, target_sample_rate)
        up = target_sample_rate // divisor
        down = original_sample_rate // divisor

        waveform = resample_poly(waveform, up=up, down=down).astype(
            np.float32,
            copy=False,
        )

    waveform_tensor = torch.from_numpy(
        np.ascontiguousarray(waveform, dtype=np.float32)
    )

    return waveform_tensor, original_sample_rate, target_sample_rate


def load_hubert_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, Any]:
    models, saved_cfg, task = (
        fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [str(checkpoint_path.resolve())]
        )
    )

    if len(models) != 1:
        raise RuntimeError(
            f"Expected one model in the checkpoint, but loaded {len(models)}."
        )

    model = models[0]
    model.eval()
    model.to(device)

    return model, saved_cfg, task


def get_encoder_layer_count(model: torch.nn.Module) -> int:
    if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
        return len(model.encoder.layers)

    model_cfg = getattr(model, "cfg", None)
    encoder_layers = getattr(model_cfg, "encoder_layers", None)

    if encoder_layers is not None:
        return int(encoder_layers)

    raise RuntimeError(
        "Could not determine the number of HuBERT transformer layers."
    )


def parse_requested_layers(
    layers_arg: str,
    encoder_layer_count: int,
) -> list[int]:
    if layers_arg.strip().lower() == "all":
        return list(range(1, encoder_layer_count + 1))

    try:
        requested = sorted(
            {
                int(value.strip())
                for value in layers_arg.split(",")
                if value.strip()
            }
        )
    except ValueError as exc:
        raise ValueError(
            f"Invalid --layers value: {layers_arg!r}. "
            'Use "all", "6", or something like "3,6,9,12".'
        ) from exc

    if not requested:
        raise ValueError("No transformer layers were requested.")

    invalid = [
        layer
        for layer in requested
        if layer < 1 or layer > encoder_layer_count
    ]

    if invalid:
        raise ValueError(
            f"Invalid layers {invalid}. This model has transformer layers "
            f"1 through {encoder_layer_count}."
        )

    return requested


def checkpoint_uses_waveform_normalization(
    saved_cfg: Any,
    task: Any,
) -> bool:
    task_cfg = getattr(saved_cfg, "task", None)

    if task_cfg is not None and hasattr(task_cfg, "normalize"):
        return bool(task_cfg.normalize)

    runtime_task_cfg = getattr(task, "cfg", None)

    if runtime_task_cfg is not None and hasattr(runtime_task_cfg, "normalize"):
        return bool(runtime_task_cfg.normalize)

    return False


def prepare_model_input(
    waveform: torch.Tensor,
    normalize: bool,
    device: torch.device,
) -> torch.Tensor:
    if normalize:
        waveform = F.layer_norm(waveform, waveform.shape)

    return waveform.unsqueeze(0).to(device)


def storage_dtype_from_name(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    return torch.float32


@torch.inference_mode()
def extract_activations(
    model: torch.nn.Module,
    source: torch.Tensor,
    requested_layers: list[int],
    storage_dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    input_features, _ = model.extract_features(
        source=source,
        padding_mask=None,
        mask=False,
        ret_conv=True,
        output_layer=1,
    )

    input_features = (
        input_features.squeeze(0)
        .detach()
        .to(device="cpu", dtype=storage_dtype)
        .contiguous()
    )

    layer_activations: dict[int, torch.Tensor] = {}

    for layer_number in requested_layers:
        hidden_states, _ = model.extract_features(
            source=source,
            padding_mask=None,
            mask=False,
            ret_conv=False,
            output_layer=layer_number,
        )

        hidden_states = (
            hidden_states.squeeze(0)
            .detach()
            .to(device="cpu", dtype=storage_dtype)
            .contiguous()
        )

        layer_activations[layer_number] = hidden_states

    return input_features, layer_activations


def save_output(
    output_path: Path,
    checkpoint_path: Path,
    wav_path: Path,
    waveform: torch.Tensor,
    original_sample_rate: int,
    model_sample_rate: int,
    normalized: bool,
    requested_layers: list[int],
    input_features: torch.Tensor,
    layer_activations: dict[int, torch.Tensor],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "format_version": 1,
        "checkpoint": {
            "name": checkpoint_path.name,
            "path": str(checkpoint_path.resolve()),
        },
        "audio": {
            "name": wav_path.name,
            "path": str(wav_path.resolve()),
            "original_sample_rate": original_sample_rate,
            "model_sample_rate": model_sample_rate,
            "num_samples": int(waveform.numel()),
            "duration_seconds": float(
                waveform.numel() / model_sample_rate
            ),
            "normalized": normalized,
        },
        "extraction": {
            "mask": False,
            "requested_transformer_layers": requested_layers,
            "input_features_shape": list(input_features.shape),
        },
        "input_features": input_features,
        "layers": layer_activations,
    }

    torch.save(payload, output_path)

    metadata_path = output_path.with_suffix(".json")

    metadata = {
        "format_version": payload["format_version"],
        "checkpoint": payload["checkpoint"],
        "audio": payload["audio"],
        "extraction": {
            **payload["extraction"],
            "saved_layer_shapes": {
                str(layer): list(states.shape)
                for layer, states in layer_activations.items()
            },
            "saved_dtype": str(input_features.dtype),
        },
        "activation_file": output_path.name,
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def main() -> None:
    args = parse_args()
    validate_paths(args)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")

    checkpoints = sorted(args.checkpoint_dir.glob("checkpoint_chunk_*.pt"))
    wav_files = sorted(
        path
        for path in args.wav_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".wav"
    )

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint_chunk_*.pt files found in {args.checkpoint_dir}")

    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in {args.wav_dir}")

    device = torch.device(args.device)
    storage_dtype = storage_dtype_from_name(args.dtype)

    print(f"Checkpoints: {len(checkpoints)}")
    print(f"WAV files: {len(wav_files)}")
    print(f"Device: {device}")

    for checkpoint_path in checkpoints:
        print(f"\n--- Loading checkpoint: {checkpoint_path.name} ---")

        model, saved_cfg, task = load_hubert_model(
            checkpoint_path=checkpoint_path,
            device=device,
        )

        encoder_layer_count = get_encoder_layer_count(model)
        requested_layers = parse_requested_layers(args.layers, encoder_layer_count)
        normalize = checkpoint_uses_waveform_normalization(saved_cfg, task)
        
        checkpoint_output_dir = args.output_dir / checkpoint_path.stem

        print(f"Transformer layers: {encoder_layer_count}")
        print(f"Requested layers: {requested_layers}")
        print(f"Waveform normalization: {normalize}")

        for wav_path in wav_files:
            waveform, original_sample_rate, model_sample_rate = load_audio(wav_path)
            
            source = prepare_model_input(
                waveform=waveform,
                normalize=normalize,
                device=device,
            )

            input_features, layer_activations = extract_activations(
                model=model,
                source=source,
                requested_layers=requested_layers,
                storage_dtype=storage_dtype,
            )

            rel_wav_path = wav_path.relative_to(args.wav_dir)
            output_path = checkpoint_output_dir / rel_wav_path.parent / f"{wav_path.stem}.pt"

            save_output(
                output_path=output_path,
                checkpoint_path=checkpoint_path,
                wav_path=wav_path,
                waveform=waveform,
                original_sample_rate=original_sample_rate,
                model_sample_rate=model_sample_rate,
                normalized=normalize,
                requested_layers=requested_layers,
                input_features=input_features,
                layer_activations=layer_activations,
            )
            print(f"  Extracted and saved: {rel_wav_path.name}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nFinished extracting all activations.")


if __name__ == "__main__":
    main()

import argparse
import csv

from pathlib import Path

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from scipy.signal import spectrogram


windows = {
    "w050_150": (0.05, 0.15),
    "w050_250": (0.05, 0.25),
    "w050_350": (0.05, 0.35)
}


def read_manifest(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    rows.sort(key=lambda r: int(r["step"]))

    steps = [int(r["step"]) for r in rows]

    if steps != list(range(1, 8)):
        raise ValueError("Expected original steps 1 through 7")

    if len({r["stimulus"] for r in rows}) != 7:
        raise ValueError("Stimulus names must be unique")

    return rows


def select_steps(rows, selection):
    if selection == "all":
        chosen = list(range(1, 8))
    elif selection == "odd":
        chosen = [1, 3, 5, 7]
    elif selection == "even":
        chosen = [2, 4, 6]
    else:
        chosen = sorted(set(
            int(s.strip()) for s in selection.split(",")
        ))

    if len(chosen) < 3:
        raise ValueError("At least three steps required")

    if any(s not in range(1, 8) for s in chosen):
        raise ValueError("Steps must be between 1 and 7")

    selected = [
        row for row in rows
        if int(row["step"]) in chosen
    ]

    steps = np.array(chosen)
    vot = steps.astype(float) * 10

    if selection in ("all", "odd", "even"):
        name = selection
    else:
        name = "steps_" + "_".join(map(str, chosen))

    return selected, steps, vot, name


def load_pooled(path, start, end):
    with np.load(path, allow_pickle=False) as d:
        x = d["hidden_states"]
        t = d["frame_center_s"]

        if x.ndim != 3 or x.shape[0] != 12 or x.shape[2] != 768:
            raise ValueError(f"Invalid activations: {path}")

        if x.shape[1] != len(t):
            raise ValueError(f"Frame mismatch: {path}")

        mask = (t >= start) & (t <= end)

        if not mask.any():
            raise ValueError(f"No frames in window: {path}")

        x = x[:, mask, :].mean(axis=1)

    if not np.isfinite(x).all():
        raise ValueError(f"Nonfinite activations: {path}")

    return x


def load_continuum(path, rows, start, end):
    return np.stack([
        load_pooled(
            path / (row["stimulus"] + ".npz"),
            start,
            end
        )
        for row in rows
    ], axis=1)


def load_phase_pair(path0, path10, rows0, rows10, start, end):
    a = load_continuum(path0, rows0, start, end)
    b = load_continuum(path10, rows10, start + 0.01, end + 0.01)

    return (a + b) / 2

def scalar(x):
    # x: (checkpoints, layers, steps, dimensions)

    axis = x[:, :, -1] - x[:, :, 0]

    denominator = np.sum(axis * axis, axis=-1)

    numerator = np.sum(
        (x - x[:, :, :1]) * axis[:, :, None],
        axis=-1
    )

    scores = np.divide(
        numerator,
        denominator[:, :, None],
        out=np.full_like(numerator, np.nan),
        where=denominator[:, :, None] > 1e-12
    )

    return scores, np.sqrt(denominator)


def total_variation(scores):
    diff = np.abs(np.diff(scores, axis=-1))
    total = diff.sum(axis=-1)

    return np.divide(
        diff.max(axis=-1),
        total,
        out=np.full_like(total, np.nan),
        where=total > 0
    )


def monotonic(scores):
    return (
        np.isfinite(scores).all(axis=-1)
        & (np.diff(scores, axis=-1) >= -1e-6).all(axis=-1)
    )


def boundary(scores, valid, vot):
    out = np.full(scores.shape[:-1], np.nan)

    for index in np.ndindex(out.shape):
        if not valid[index]:
            continue

        row = scores[index]
        crossings = []

        for i in range(len(vot) - 1):
            a, b = row[i:i + 2]

            if a == b:
                continue

            if a <= 0.5 <= b:
                p = vot[i] + (
                    (0.5 - a) / (b - a)
                ) * (vot[i + 1] - vot[i])

                crossings.append(p)

        crossings = np.unique(
            np.round(crossings, 8)
        )

        if len(crossings) == 1:
            out[index] = crossings[0]

    return out


def analyze(x, vot):
    scores, endpoints = scalar(x)

    raw = np.linalg.norm(
        np.diff(x, axis=2),
        axis=-1
    )

    path_length = raw.sum(axis=-1)

    fractions = np.divide(
        raw,
        path_length[:, :, None],
        out=np.full_like(raw, np.nan),
        where=path_length[:, :, None] > 0
    )

    ratio = np.divide(
        path_length,
        endpoints,
        out=np.full_like(endpoints, np.nan),
        where=endpoints > 1e-12
    )

    valid = monotonic(scores)
    steep = total_variation(scores)

    return {
        "scores": scores,
        "max_to_total_variation_ratio": steep,
        "monotonic": valid,
        "boundary_ms": boundary(scores, valid, vot),
        "endpoint_distance": endpoints,
        "adjacent_raw": raw,
        "adjacent_fraction": fractions,
        "path_length": path_length,
        "path_ratio": ratio
    }


def logmel(path, start, end, hop):
    x, sr = sf.read(path, dtype="float32")

    if sr != 16000 or x.ndim != 1:
        raise ValueError(f"Expected mono 16 kHz: {path}")

    f, t, power = spectrogram(
        x,
        fs=sr,
        window="hann",
        nperseg=400,
        noverlap=400 - hop,
        nfft=512,
        scaling="spectrum"
    )

    def hz_to_mel(x):
        return 2595 * np.log10(1 + x / 700)

    def mel_to_hz(x):
        return 700 * (10 ** (x / 2595) - 1)

    edges = mel_to_hz(
        np.linspace(
            hz_to_mel(0),
            hz_to_mel(sr / 2),
            42
        )
    )

    filters = np.zeros((40, len(f)))

    for i in range(40):
        left, mid, right = edges[i:i + 3]

        filters[i] = np.maximum(
            0,
            np.minimum(
                (f - left) / (mid - left),
                (right - f) / (right - mid)
            )
        )

    mel = np.log(
        np.maximum(filters @ power, 1e-10)
    )

    mask = (t >= start) & (t <= end)

    if not mask.any():
        raise ValueError(f"No acoustic frames: {path}")

    return mel[:, mask].mean(axis=-1)


def acoustic_baseline(manifest, rows, vot, start, end, hop):
    x = []

    for row in rows:
        path = (
            manifest.parent / row["wav_path"]
        ).resolve()

        x.append(logmel(path, start, end, hop))

    # Same four axes as HuBERT: (C, L, N, D)
    x = np.stack(x)[None, None]

    return analyze(x, vot)


def get_checkpoints(root, selected):
    if selected:
        chunks = [
            int(s.strip())
            for s in selected.split(",")
        ]

        if len(chunks) != len(set(chunks)):
            raise ValueError("Duplicate checkpoints")

        paths = [
            root / f"checkpoint_chunk_{i:04d}"
            for i in chunks
        ]
    else:
        paths = sorted(
            root.glob("checkpoint_chunk_*"),
            key=lambda p: int(p.name.rsplit("_", 1)[-1])
        )

    if not paths:
        raise FileNotFoundError("No checkpoints found")

    for path in paths:
        if not path.is_dir():
            raise FileNotFoundError(path)

    return paths


def upsert_csv(path, fields, rows, keys):
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {}

    if path.exists():
        with path.open(newline="") as f:
            reader = csv.DictReader(f)

            if reader.fieldnames != fields:
                raise ValueError(f"CSV columns do not match: {path}")

            for row in reader:
                key = tuple(str(row[k]) for k in keys)
                data[key] = row

    for row in rows:
        row = {
            k: (
                "" if isinstance(v, (float, np.floating))
                and not np.isfinite(v) else v
            )
            for k, v in row.items()
        }

        key = tuple(str(row[k]) for k in keys)
        data[key] = row

    tmp = path.with_suffix(".tmp")

    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data.values())

    tmp.replace(path)

def write_metrics(root, chunk, steps, vot, window, subset, result):
    projections = []
    distances = []
    metrics = []

    linear = (vot - vot[0]) / (vot[-1] - vot[0])

    for i in range(12):
        layer = i + 1

        common = {
            "checkpoint": chunk,
            "layer": layer,
            "window": window,
            "subset": subset
        }

        metrics.append({
            **common,
            "max_adjacent_change_fraction": result["max_to_total_variation_ratio"][0, i],
            "monotonic": bool(result["monotonic"][0, i]),
            "projection_midpoint_ms": result["boundary_ms"][0, i],
            "endpoint_distance": result["endpoint_distance"][0, i],
            "path_length": result["path_length"][0, i],
            "path_tortuosity_ratio": result["path_ratio"][0, i]
        })

        for j, step in enumerate(steps):
            projections.append({
                **common,
                "step": int(step),
                "vot_ms": int(vot[j]),
                "projection": result["scores"][0, i, j],
                "linear_reference": linear[j]
            })

        for j in range(len(steps) - 1):
            distances.append({
                **common,
                "from_vot_ms": int(vot[j]),
                "to_vot_ms": int(vot[j + 1]),
                "midpoint_ms": (vot[j] + vot[j + 1]) / 2,
                "adjacent_distance": result["adjacent_raw"][0, i, j],
                "adjacent_fraction": result["adjacent_fraction"][0, i, j]
            })

    upsert_csv(
        root / "projections.csv",
        list(projections[0]),
        projections,
        ["checkpoint", "layer", "window", "subset", "step"]
    )

    upsert_csv(
        root / "distances.csv",
        list(distances[0]),
        distances,
        ["checkpoint", "layer", "window", "subset", "from_vot_ms"]
    )

    upsert_csv(
        root / "metrics.csv",
        list(metrics[0]),
        metrics,
        ["checkpoint", "layer", "window", "subset"]
    )

    valid = result["monotonic"][0, 2:8]
    values = result["max_to_total_variation_ratio"][0, 2:8]
    values = values[valid & np.isfinite(values)]

    comparison = [{
        "checkpoint": chunk,
        "window": window,
        "subset": subset,
        "monotonic_layers_3_8": len(values),
        "mean_max_adjacent_change_fraction": (
            values.mean() if len(values) else np.nan
        )
    }]

    upsert_csv(
        root / "window_comparison.csv",
        list(comparison[0]),
        comparison,
        ["checkpoint", "window", "subset"]
    )


def write_baselines(root, steps, vot, window, subset, base10, base20):
    rows = []
    linear = (vot - vot[0]) / (vot[-1] - vot[0])

    for hop, result in ((160, base10), (320, base20)):
        for j, step in enumerate(steps):
            has_next = j < len(steps) - 1

            rows.append({
                "window": window,
                "subset": subset,
                "hop_ms": hop / 16,
                "step": int(step),
                "vot_ms": int(vot[j]),
                "projection": result["scores"][0, 0, j],
                "linear_reference": linear[j],
                "adjacent_distance": (
                    result["adjacent_raw"][0, 0, j]
                    if has_next else np.nan
                ),
                "adjacent_fraction": (
                    result["adjacent_fraction"][0, 0, j]
                    if has_next else np.nan
                )
            })

    upsert_csv(
        root / "acoustic.csv",
        list(rows[0]),
        rows,
        ["window", "subset", "hop_ms", "step"]
    )



def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("stimuli/praat/manifest_p0.csv")
    )

    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("activations/vot_p0")
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/vot_phase/csv")
    )

    p.add_argument("--checkpoints")
    p.add_argument("--acoustic", action="store_true")

    args = p.parse_args()

    manifest0 = args.manifest.resolve()
    manifest10 = manifest0.with_name("manifest_p10.csv")

    rows0 = read_manifest(manifest0)
    rows10 = read_manifest(manifest10)

    paths0 = get_checkpoints(args.input_dir, args.checkpoints)
    paths10 = get_checkpoints(
        args.input_dir.parent / "vot_p10",
        args.checkpoints
    )

    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)

    for path0, path10 in zip(paths0, paths10):
        if path0.name != path10.name:
            raise ValueError("Checkpoint mismatch between phases")

        ck = int(path0.name.rsplit("_", 1)[-1])
        print(f"Analyzing checkpoint {ck}", flush=True)

        for window, (start, end) in windows.items():
            pooled = load_phase_pair(
                path0, path10, rows0, rows10, start, end
            )

            selected, steps, vot, _ = select_steps(rows0, "all")

            x = pooled[:, steps - 1, :][None]
            result = analyze(x, vot)

            write_metrics(
                root, ck, steps, vot, window, "all", result
            )

    if args.acoustic:
        manifest = manifest0.with_name("manifest.csv")
        rows = read_manifest(manifest)

        for window, (start, end) in windows.items():
            for subset in ("all", "odd", "even"):
                selected, steps, vot, _ = select_steps(rows, subset)

                base10 = acoustic_baseline(
                    manifest, selected, vot, start, end, 160
                )

                base20 = acoustic_baseline(
                    manifest, selected, vot, start, end, 320
                )

                write_baselines(
                    root, steps, vot, window, subset,
                    base10, base20
                )

    print(f"Saved CSVs to {root.resolve()}")



if __name__ == "__main__":
    main()


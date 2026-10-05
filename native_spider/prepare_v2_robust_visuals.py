#!/usr/bin/env python3
"""Prepare complete-span video contact sheets and fixed-grid plots for one v2 condition."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

GRID = [(x, y) for x in (-5, 0, 5) for y in (-5, 0, 5)]
REAL_STEPS = 391
TOTAL_STEPS = 450


def contact_sheets(video: Path, out: Path, stem: str) -> dict:
    if not video.is_file():
        return {"status": "PENDING", "video": str(video), "missing": True}
    reported_frames, reported_duration = imageio_ffmpeg.count_frames_and_secs(str(video))
    if reported_frames <= 0:
        raise ValueError(f"video reports no frames: {video}")
    indices = np.linspace(0, reported_frames - 1, min(20, reported_frames), dtype=int).tolist()
    selected = set(indices)
    retained: dict[int, Image.Image] = {}
    reader = imageio_ffmpeg.read_frames(str(video), pix_fmt="rgb24")
    metadata = next(reader)
    width, height = metadata["size"]
    fps = float(metadata["fps"])
    decoded = 0
    for frame_bytes in reader:
        if decoded in selected:
            retained[decoded] = Image.frombytes("RGB", (width, height), frame_bytes)
        decoded += 1
    if decoded != reported_frames or len(retained) != len(indices):
        raise ValueError(f"incomplete decode for {video}: decoded {decoded}/{reported_frames}")

    tile_w = 720
    tile_h = max(1, round(tile_w * height / width))
    label_h = 38
    paths = []
    for sheet_i, group in enumerate((indices[:10], indices[10:]), start=1):
        if not group:
            continue
        cols, rows = 2, 5
        sheet = Image.new("RGB", (cols * tile_w, rows * (tile_h + label_h)), "white")
        draw = ImageDraw.Draw(sheet)
        for slot, frame_i in enumerate(group):
            col, row = slot % cols, slot // cols
            x, y = col * tile_w, row * (tile_h + label_h)
            thumb = retained[frame_i].resize((tile_w, tile_h), Image.Resampling.LANCZOS)
            sheet.paste(thumb, (x, y + label_h))
            draw.text((x + 6, y + 7), f"frame {frame_i} | {frame_i / fps:.2f} s", fill="black")
        path = out / f"{stem}_contact_sheet_{sheet_i}.jpg"
        sheet.save(path, quality=94)
        paths.append(str(path))
    return {
        "status": "DECODED",
        "video": str(video),
        "video_bytes": video.stat().st_size,
        "decoded_frames": decoded,
        "reported_frames": reported_frames,
        "reported_duration_s": reported_duration,
        "fps": fps,
        "decoded_duration_s": decoded / fps,
        "frame_size_px": [width, height],
        "sample_frame_indices": indices,
        "sample_time_s": [round(i / fps, 6) for i in indices],
        "contact_sheets": paths,
        "sheet_layout": f"two sheets up to 2x5; tile {tile_w}x{tile_h} preserves source aspect ratio",
        "interpretation": "Full video decoded; contact sheets show evenly spaced samples across the full duration, not full-speed visual review.",
    }


def load_metrics(csv_path: Path) -> dict[int, dict[str, str]]:
    with csv_path.open(newline="") as f:
        rows = {int(row["world_index"]): row for row in csv.DictReader(f)}
    return rows


def make_plots(result_dir: Path, out: Path) -> dict:
    evaluation = result_dir / "evaluation"
    traj_path = evaluation / "position_probe_trajectories.npz"
    ref_path = result_dir / "official_loaded_reference.npz"
    metrics_path = evaluation / "position_probe_metrics.csv"
    missing = [str(p) for p in (traj_path, ref_path, metrics_path) if not p.is_file()]
    if missing:
        return {"status": "PENDING", "missing": missing}

    with np.load(traj_path) as data:
        qpos = np.asarray(data["qpos"], dtype=float)
        dt = float(data["sim_dt_s"])
    with np.load(ref_path) as data:
        reference = np.asarray(data["qpos"], dtype=float)
        ref_dt = float(data["sim_dt"])
    if qpos.shape[:2] != (TOTAL_STEPS, 9) or qpos.shape[2] != 25:
        raise ValueError(f"unexpected saved probe qpos shape: {qpos.shape}")
    indices = np.rint((np.arange(1, TOTAL_STEPS + 1) * dt) / ref_dt).astype(int)
    if not np.allclose(indices * ref_dt, np.arange(1, TOTAL_STEPS + 1) * dt, atol=ref_dt * 1e-3):
        raise ValueError("probe physical steps do not align to reference time grid")
    ref = reference[indices]
    actual_obj = qpos[:, :, 18:21]
    ref_obj = ref[:, None, 18:21]
    root_error_cm = np.linalg.norm(actual_obj - ref_obj, axis=2) * 100.0
    quat = qpos[:, :, 21:25]
    ref_quat = ref[:, None, 21:25]
    quat_n = np.linalg.norm(quat, axis=2)
    ref_n = np.linalg.norm(ref_quat, axis=2)
    cosine = np.abs(np.sum(quat * ref_quat, axis=2) / (quat_n * ref_n))
    rot_error_deg = np.rad2deg(2 * np.arccos(np.clip(cosine, 0.0, 1.0)))
    time_s = np.arange(1, TOTAL_STEPS + 1) * dt
    metric_rows = load_metrics(metrics_path)

    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    for i, ax in enumerate(axes.flat):
        ax.plot(time_s, actual_obj[:, i, 2], color="tab:blue", linewidth=1.2, label="sim root z")
        ax.plot(time_s, ref_obj[:, 0, 2], color="black", linestyle="--", linewidth=1, label="reference z")
        ax.axvline(REAL_STEPS * dt, color="grey", linestyle=":", linewidth=1)
        ax.axvspan(REAL_STEPS * dt, TOTAL_STEPS * dt, color="grey", alpha=0.12)
        ax.set_title(f"world {i}: ({GRID[i][0]:+d}, {GRID[i][1]:+d}) mm")
        ax.grid(alpha=0.2)
        if i // 3 == 2:
            ax.set_xlabel("Physical time (s)")
        if i % 3 == 0:
            ax.set_ylabel("Object root z (m)")
        mr = metric_rows.get(i, {})
        if i == 0:
            ax.legend(fontsize=8, loc="best")
        ax.text(0.98, 0.04, f"lift={mr.get('lift_proxy_all_steps_at_least_0p10m', '?')}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    fig.suptitle("Nominal/robust fixed-control evaluation: object height by XY world\nshaded tail is steps 392–450; reference metrics cover steps 1–391")
    fig.tight_layout()
    height_path = out / "position_height_9grid.png"
    fig.savefig(height_path, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    for i, ax in enumerate(axes.flat):
        ax.plot(time_s, root_error_cm[:, i], color="tab:red", linewidth=1.1, label="root position error")
        ax.set_ylabel("Root error (cm)")
        ax2 = ax.twinx()
        ax2.plot(time_s, rot_error_deg[:, i], color="tab:purple", alpha=0.75, linewidth=0.9, label="SO(3) error")
        ax2.set_ylabel("SO(3) error (deg)")
        ax.axvline(REAL_STEPS * dt, color="grey", linestyle=":", linewidth=1)
        ax.axvspan(REAL_STEPS * dt, TOTAL_STEPS * dt, color="grey", alpha=0.12)
        ax.set_title(f"world {i}: ({GRID[i][0]:+d}, {GRID[i][1]:+d}) mm")
        ax.grid(alpha=0.2)
        if i // 3 == 2:
            ax.set_xlabel("Physical time (s)")
        if i == 0:
            ax.legend(loc="upper left", fontsize=8)
            ax2.legend(loc="upper right", fontsize=8)
    fig.suptitle("Object root position and orientation error\nmetrics align physical steps 1–450; frozen proxy uses steps 1–391")
    fig.tight_layout()
    error_path = out / "tracking_error_9grid.png"
    fig.savefig(error_path, dpi=150)
    plt.close(fig)
    return {
        "status": "GENERATED",
        "trajectory": str(traj_path),
        "reference": str(ref_path),
        "metrics_csv": str(metrics_path),
        "trajectory_shape": list(qpos.shape),
        "simulation_dt_s": dt,
        "reference_dt_s": ref_dt,
        "steps_aligned": [1, TOTAL_STEPS],
        "scored_real_reference_steps": [1, REAL_STEPS],
        "padded_tail_steps": [REAL_STEPS + 1, TOTAL_STEPS],
        "height_plot": str(height_path),
        "error_plot": str(error_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True,
                        help="Condition directory, e.g. results/native_v2_01/robust01/nominal_s0")
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    out = result_dir / "visuals"
    out.mkdir(parents=True, exist_ok=True)
    plan_video = result_dir / "visualization_mjwp_fast.mp4"
    eval_video = result_dir / "evaluation/position_probe_9grid.mp4"
    manifest = {
        "condition_dir": str(result_dir),
        "planning_video": contact_sheets(plan_video, out, "planning"),
        "evaluation_9grid_video": contact_sheets(eval_video, out, "evaluation_9grid"),
        "plots": make_plots(result_dir, out),
        "visual_scope": "Samples and derived plots are for review; full decoding does not imply full-speed video review.",
    }
    (out / "visual_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

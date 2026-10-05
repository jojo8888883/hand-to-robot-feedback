#!/usr/bin/env python3
"""Run the frozen 3x3 XY initial-object-position probe with one shared control sequence."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve()
REMOTE_ROOT = Path(os.environ.get("SPIDER_ROOT", "/mnt/sdc/dty_user/hand_to_robot_20261002"))
if SCRIPT.parent.name == "scripts" and (SCRIPT.parent.parent / "native_spider").is_dir():
    PROJECT_ROOT = SCRIPT.parent.parent
    SPIDER_DIR = PROJECT_ROOT / "native_spider"
else:
    SPIDER_DIR = SCRIPT.parent
    PROJECT_ROOT = SPIDER_DIR.parent
DEFAULT_SPEC = SPIDER_DIR / "V2_DIAGNOSTIC_RUN_SPEC.json"
DEFAULT_CONTROLS = PROJECT_ROOT / "results/native_v2_01/control_replay/continuous_control/trajectory_replay.npz"
DEFAULT_REFERENCE = PROJECT_ROOT / "results/native_v2_01/compat_diagnostic_s0/official_loaded_reference.npz"
REAL_STEPS = 391
TOTAL_STEPS = 450
SIM_DT_FALLBACK = 0.01
OFFSETS_M = (-0.005, 0.0, 0.005)
PAIR_MARGIN = -0.005


def make_grid() -> np.ndarray:
    """x-major Cartesian grid; y varies within each x, both in meters."""
    return np.asarray([(x, y) for x in OFFSETS_M for y in OFFSETS_M], dtype=np.float64)


def quat_angle_rad(actual: np.ndarray, target: np.ndarray) -> np.ndarray:
    """SO(3) geodesic angle for MuJoCo wxyz quaternions."""
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    dot = np.abs(np.sum(a * b, axis=-1))
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


def summarize_world(qpos: np.ndarray, initial_qpos: np.ndarray, reference: np.ndarray) -> dict:
    """Compute protocol metrics over output rows 1:392 (391 saved states)."""
    real = np.asarray(qpos[:REAL_STEPS], dtype=np.float64)
    target = np.asarray(reference[1 : REAL_STEPS + 1], dtype=np.float64)
    if real.shape != (REAL_STEPS, 25) or target.shape != (REAL_STEPS, 25):
        raise ValueError(f"expected aligned (391,25) arrays, got {real.shape} and {target.shape}")
    pos_error = np.linalg.norm(real[:, 18:21] - target[:, 18:21], axis=1)
    rot_error = quat_angle_rad(real[:, 21:25], target[:, 21:25])
    rise = real[:, 20] - float(initial_qpos[20])
    last50_rise = qpos[341:391, 20].astype(np.float64) - float(initial_qpos[20])
    tail_rise = qpos[391:, 20].astype(np.float64) - float(initial_qpos[20])
    last50_pos_error = pos_error[341:391]
    last50_rot_error = rot_error[341:391]
    return {
        "real_reference_window": {
            "physical_steps_start_end_inclusive": [1, 391],
            "samples": 391,
            "root_position_error_mean_m": float(pos_error.mean()),
            "root_position_error_mean_cm": float(pos_error.mean() * 100.0),
            "root_position_error_max_m": float(pos_error.max()),
            "root_position_error_max_cm": float(pos_error.max() * 100.0),
            "so3_error_mean_rad": float(rot_error.mean()),
            "so3_error_max_rad": float(rot_error.max()),
            "max_root_rise_m": float(rise.max()),
            "track_reference_proxy_all_steps": bool(np.all(pos_error < 0.10) and np.all(rot_error < 0.5)),
        },
        "last_0p5s_lift_window": {
            "output_slice_zero_based": [341, 391],
            "physical_steps_start_end_inclusive": [342, 391],
            "samples": int(len(last50_rise)),
            "minimum_root_rise_m": float(last50_rise.min()),
            "lift_proxy_all_steps_at_least_0p10m": bool(np.all(last50_rise >= 0.10)),
            "root_position_error_mean_cm": float(last50_pos_error.mean() * 100.0),
            "root_position_error_max_cm": float(last50_pos_error.max() * 100.0),
            "so3_error_mean_rad": float(last50_rot_error.mean()),
            "so3_error_max_rad": float(last50_rot_error.max()),
            "track_reference_proxy_all_steps": bool(
                np.all(last50_pos_error < 0.10) and np.all(last50_rot_error < 0.5)
            ),
        },
        "padded_tail_after_reference": {
            "physical_steps_start_end_inclusive": [392, TOTAL_STEPS],
            "samples": int(len(tail_rise)),
            "maximum_root_rise_m": float(tail_rise.max()) if len(tail_rise) else None,
            "tracking_metrics_interpreted": False,
        },
    }


def read_probe_inputs(controls_path: Path, reference_path: Path, spec_path: Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    spec = json.loads(spec_path.read_text())
    with np.load(controls_path) as archive:
        controls = np.asarray(archive["ctrl"]).reshape(-1, 18).astype(np.float32)
        baseline_qpos = np.asarray(archive["qpos"]).reshape(-1, 25).astype(np.float64)
        baseline_initial = np.asarray(archive["initial_qpos"]).reshape(25).astype(np.float64)
    with np.load(reference_path) as archive:
        reference = np.asarray(archive["qpos"], dtype=np.float64)
        sim_dt = float(np.asarray(archive["sim_dt"]).item()) if "sim_dt" in archive else SIM_DT_FALLBACK
    return spec, controls, baseline_qpos, baseline_initial, reference, sim_dt


def preflight(args: argparse.Namespace) -> int:
    """Pure NumPy/data validation; deliberately imports no MuJoCo/Warp/Torch modules."""
    remote_spec = REMOTE_ROOT / "native_spider/V2_DIAGNOSTIC_RUN_SPEC.json"
    remote_controls = REMOTE_ROOT / "results/native_v2_01/control_replay/continuous_control/trajectory_replay.npz"
    remote_reference = REMOTE_ROOT / "results/native_v2_01/compat_diagnostic_s0/official_loaded_reference.npz"
    spec_path = Path(args.spec) if args.spec else (DEFAULT_SPEC if DEFAULT_SPEC.is_file() else remote_spec)
    controls_path = Path(args.controls) if args.controls else (DEFAULT_CONTROLS if DEFAULT_CONTROLS.is_file() else remote_controls)
    reference_path = Path(args.reference) if args.reference else (DEFAULT_REFERENCE if DEFAULT_REFERENCE.is_file() else remote_reference)
    spec, controls, baseline_qpos, baseline_initial, reference, sim_dt = read_probe_inputs(
        controls_path, reference_path, spec_path
    )
    if spec.get("source_revision") != "44717007de41cbef7565dff7ff9f4453557a2d3d":
        raise ValueError("unexpected official source revision in run spec")
    if controls.shape != (TOTAL_STEPS, 18) or not np.isfinite(controls).all():
        raise ValueError(f"expected finite controls {(TOTAL_STEPS, 18)}, got {controls.shape}")
    if baseline_qpos.shape != (TOTAL_STEPS, 25) or not np.isfinite(baseline_qpos).all():
        raise ValueError("previous continuous replay must provide finite (450,25) qpos")
    if reference.shape[0] < REAL_STEPS + 1 or reference.shape[1] != 25:
        raise ValueError(f"reference must cover qpos rows 1:392, got {reference.shape}")
    if not np.isfinite(reference[: REAL_STEPS + 1]).all() or sim_dt <= 0:
        raise ValueError("reference window or sim_dt is invalid")
    grid = make_grid()
    if grid.shape != (9, 2) or len({tuple(x) for x in grid.tolist()}) != 9:
        raise ValueError("position grid must contain nine unique XY points")
    expected = np.asarray([(x, y) for x in OFFSETS_M for y in OFFSETS_M], dtype=np.float64)
    if not np.array_equal(grid, expected):
        raise ValueError("grid order differs from protocol x-major/y-minor order")

    # Exercise the pure-array scoring path on the previously saved continuous replay only.
    # This is preflight evidence, not a result for any of the nine new worlds.
    baseline_metrics = summarize_world(baseline_qpos, baseline_initial, reference)
    record = {
        "status": "PASS_NO_SIMULATION",
        "gpu_used": False,
        "probe_results_created": False,
        "official_source_revision": spec["source_revision"],
        "controls_source": str(controls_path),
        "controls_shape": list(controls.shape),
        "baseline_qpos_shape": list(baseline_qpos.shape),
        "reference_source": str(reference_path),
        "reference_qpos_shape": list(reference.shape),
        "real_reference_window": {"reference_rows": [1, 392], "samples": REAL_STEPS, "sim_dt_s": sim_dt},
        "grid_order_x_then_y_m": grid.tolist(),
        "pair_margin_m": PAIR_MARGIN,
        "pure_numpy_metric_smoke_on_prior_continuous_replay": baseline_metrics,
        "metric_smoke_note": "Previously saved one-world replay only; values are not measurements of the nine probe worlds.",
    }
    output = Path(args.preflight_output) if args.preflight_output else SPIDER_DIR / "V2_POSITION_PROBE_PREFLIGHT.json"
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="CPU-only input/grid/metric preflight; no simulator imports")
    parser.add_argument("--spec", help="run spec JSON path")
    parser.add_argument("--controls", help="saved 450-control trajectory NPZ")
    parser.add_argument("--reference", help="official_loaded_reference.npz path")
    parser.add_argument("--preflight-output", help="CPU preflight JSON destination")
    parser.add_argument("--output-dir", help="Separate output directory for a newly frozen comparison")
    parser.add_argument("--job-id", default="h2r-20261004-spider-native-v2-position-probe-s0")
    parser.add_argument("--protocol", default="NATIVE_V2_POSITION_PROBE_PROTOCOL.md")
    args = parser.parse_args()
    if args.preflight:
        return preflight(args)
    return run_probe(args)


def run_probe(args: argparse.Namespace) -> int:
    """Remote-only MJWP execution path. The root agent controls when this is invoked."""
    root = REMOTE_ROOT
    repo = root / "repos/spider-native"
    spec_path = Path(args.spec) if args.spec else root / "native_spider/V2_DIAGNOSTIC_RUN_SPEC.json"
    controls_path = Path(args.controls) if args.controls else root / "results/native_v2_01/control_replay/continuous_control/trajectory_replay.npz"
    reference_path = Path(args.reference) if args.reference else root / "results/native_v2_01/compat_diagnostic_s0/official_loaded_reference.npz"
    spec = json.loads(spec_path.read_text())
    with np.load(controls_path) as archive:
        controls = np.asarray(archive["ctrl"]).reshape(-1, 18).astype(np.float32)
    with np.load(reference_path) as archive:
        reference_np = np.asarray(archive["qpos"], dtype=np.float64)
        sim_dt = float(np.asarray(archive["sim_dt"]).item())
    if controls.shape != (TOTAL_STEPS, 18) or not np.isfinite(controls).all():
        raise ValueError(f"saved controls must be finite {(TOTAL_STEPS, 18)}, got {controls.shape}")
    if reference_np.shape[0] < REAL_STEPS + 1 or reference_np.shape[1] != 25:
        raise ValueError("official reference does not cover requested 391-step window")

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    import imageio.v2 as imageio
    import mujoco
    import torch
    import warp as wp
    from PIL import Image, ImageDraw
    from hydra import compose, initialize_config_dir
    from spider.config import Config, filter_config_fields, process_config
    from spider.io import load_data
    import spider.simulators.mjwp as sim
    from spider.viewers import render_image, setup_renderer

    overrides = list(spec["author_cli_arguments"])
    with initialize_config_dir(config_dir=str(repo / "examples/config"), version_base=None):
        cfg = Config(**filter_config_fields(dict(compose(config_name="default", overrides=overrides))))
    cfg = process_config(cfg)
    cfg.num_samples = 9
    cfg.seed = int(spec.get("seed", 0))

    original_setup_model = sim.setup_mj_model

    def compatible_setup_model(config):
        model = original_setup_model(config)
        model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_INVDISCRETE
        return model

    sim.setup_mj_model = compatible_setup_model
    reference_tensors = load_data(cfg, cfg.data_path)
    env = sim.setup_env(cfg, reference_tensors)
    model = env.model_cpu
    render_data = mujoco.MjData(model)
    reference_data = mujoco.MjData(model)
    renderer = setup_renderer(cfg, model)

    # Save the state produced by setup_env's one CPU initialization step.
    initial_qpos = sim.get_qpos(cfg, env).detach().clone()
    initial_qvel = sim.get_qvel(cfg, env).detach().clone()
    initial_time = wp.to_torch(env.data_wp.time).detach().clone()
    initial_qpos_np = initial_qpos.cpu().numpy().copy()
    initial_qvel_np = initial_qvel.cpu().numpy().copy()
    initial_time_np = initial_time.cpu().numpy().copy()
    grid = make_grid()

    # Freeze contact margin, then perturb only object root x/y with a qpos copy.
    sim.load_env_params(cfg, env, {"pair_margin": PAIR_MARGIN})
    qpos_changed = initial_qpos.clone()
    offsets_t = torch.as_tensor(grid, dtype=qpos_changed.dtype, device=qpos_changed.device)
    qpos_changed[:, 18:20] = qpos_changed[:, 18:20] + offsets_t
    wp.copy(env.data_wp.qpos, wp.from_torch(qpos_changed.contiguous()))
    initial_probe_qpos = sim.get_qpos(cfg, env).detach().cpu().numpy().copy()
    qvel_after_offsets = sim.get_qvel(cfg, env).detach().cpu().numpy().copy()
    time_after_offsets = wp.to_torch(env.data_wp.time).detach().cpu().numpy().copy()
    expected_qpos = initial_qpos_np.copy()
    expected_qpos[:, 18:20] += grid.astype(expected_qpos.dtype)
    untouched = np.ones(25, dtype=bool)
    untouched[18:20] = False
    if not np.array_equal(initial_probe_qpos[:, untouched], initial_qpos_np[:, untouched]):
        raise RuntimeError("XY initialization changed non-XY qpos fields")
    if not np.array_equal(qvel_after_offsets, initial_qvel_np):
        raise RuntimeError("XY initialization changed qvel")
    if not np.array_equal(time_after_offsets, initial_time_np):
        raise RuntimeError("XY initialization changed simulation time")
    if not np.allclose(initial_probe_qpos[:, 18:20], expected_qpos[:, 18:20], atol=0.0, rtol=0.0):
        raise RuntimeError("XY initialization did not match the nine protocol offsets")

    out = Path(args.output_dir) if args.output_dir else root / "results/native_v2_01/position_probe_s0"
    out.mkdir(parents=True, exist_ok=True)
    progress_path = out / "progress.json"
    progress_path.write_text(json.dumps({"stage": "RUNNING", "completed_steps": 0, "grid_worlds": 9}) + "\n")
    qpos_rows = np.empty((TOTAL_STEPS, 9, 25), dtype=np.float32)
    qvel_rows = np.empty((TOTAL_STEPS, 9, 24), dtype=np.float32)
    time_rows = np.empty((TOTAL_STEPS, 9), dtype=np.float64)
    tile_size = (480, 320)
    video_path = out / "position_probe_9grid.mp4"
    writer = imageio.get_writer(str(video_path), fps=50, codec="libx264", quality=8, macro_block_size=16)
    try:
        for step, control in enumerate(controls):
            control_t = torch.from_numpy(control).to(cfg.device)
            sim.step_env(cfg, env, control_t)
            q = sim.get_qpos(cfg, env).detach().cpu().numpy().copy()
            v = sim.get_qvel(cfg, env).detach().cpu().numpy().copy()
            t = wp.to_torch(env.data_wp.time).detach().cpu().numpy().copy()
            if not np.isfinite(q).all() or not np.isfinite(v).all() or not np.isfinite(t).all():
                raise RuntimeError(f"nonfinite state at saved control step {step + 1}")
            qpos_rows[step] = q
            qvel_rows[step] = v
            time_rows[step] = t - initial_time_np

            # One frame per two physical steps, showing the sim half for all nine worlds.
            if (step + 1) % 2 == 0:
                canvas = Image.new("RGB", (3 * tile_size[0], 3 * tile_size[1]), (0, 0, 0))
                for world, (dx, dy) in enumerate(grid):
                    render_data.qpos[:] = q[world]
                    render_data.qvel[:] = v[world]
                    render_data.ctrl[:] = control
                    render_data.time = float(t[world])
                    mujoco.mj_forward(model, render_data)
                    ref_row = reference_tensors[0][min(step + 1, len(reference_tensors[0]) - 1)]
                    reference_data.qpos[:] = ref_row.detach().cpu().numpy()
                    image = np.asarray(render_image(cfg, renderer, model, render_data, reference_data))
                    sim_half = image[:, image.shape[1] // 2 :]
                    tile = Image.fromarray(sim_half).resize(tile_size, Image.Resampling.LANCZOS)
                    draw = ImageDraw.Draw(tile)
                    label = f"x={dx * 1000:+.0f} mm  y={dy * 1000:+.0f} mm"
                    draw.rectangle((4, 4, 204, 30), fill=(0, 0, 0))
                    draw.text((9, 8), label, fill=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0))
                    canvas.paste(tile, ((world % 3) * tile_size[0], (world // 3) * tile_size[1]))
                writer.append_data(np.asarray(canvas))

            if (step + 1) % 50 == 0:
                progress_path.write_text(json.dumps({"stage": "RUNNING", "completed_steps": step + 1, "grid_worlds": 9}) + "\n")
    finally:
        writer.close()
        if hasattr(renderer, "close"):
            renderer.close()

    offsets = grid.astype(np.float32)
    np.savez_compressed(
        out / "position_probe_trajectories.npz",
        qpos=qpos_rows,
        qvel=qvel_rows,
        ctrl=controls.astype(np.float32),
        relative_time=time_rows,
        initial_qpos=initial_probe_qpos.astype(np.float32),
        initial_qvel=initial_qvel_np.astype(np.float32),
        initial_time=initial_time_np.astype(np.float64),
        xy_offsets_m=offsets,
        pair_margin_m=np.asarray(PAIR_MARGIN),
        sim_dt_s=np.asarray(sim_dt),
    )

    worlds = []
    csv_path = out / "position_probe_metrics.csv"
    csv_rows = []
    for world, (dx, dy) in enumerate(grid):
        metrics = summarize_world(qpos_rows[:, world], initial_probe_qpos[world], reference_np)
        entry = {
            "world_index": world,
            "x_offset_m": float(dx),
            "y_offset_m": float(dy),
            "x_offset_mm": float(dx * 1000.0),
            "y_offset_mm": float(dy * 1000.0),
            **metrics,
        }
        worlds.append(entry)
        real = entry["real_reference_window"]
        lift = entry["last_0p5s_lift_window"]
        tail = entry["padded_tail_after_reference"]
        csv_rows.append({
            "world_index": world,
            "x_offset_mm": dx * 1000.0,
            "y_offset_mm": dy * 1000.0,
            "real_steps": REAL_STEPS,
            "root_position_error_mean_cm": real["root_position_error_mean_cm"],
            "root_position_error_max_cm": real["root_position_error_max_cm"],
            "so3_error_mean_rad": real["so3_error_mean_rad"],
            "so3_error_max_rad": real["so3_error_max_rad"],
            "max_root_rise_m": real["max_root_rise_m"],
            "last_0p5s_min_root_rise_m": lift["minimum_root_rise_m"],
            "lift_proxy_all_steps_at_least_0p10m": lift["lift_proxy_all_steps_at_least_0p10m"],
            "track_reference_proxy_all_steps": real["track_reference_proxy_all_steps"],
            "last_0p5s_root_position_error_mean_cm": lift["root_position_error_mean_cm"],
            "last_0p5s_root_position_error_max_cm": lift["root_position_error_max_cm"],
            "last_0p5s_so3_error_mean_rad": lift["so3_error_mean_rad"],
            "last_0p5s_so3_error_max_rad": lift["so3_error_max_rad"],
            "last_0p5s_track_reference_proxy_all_steps": lift["track_reference_proxy_all_steps"],
            "tail_59_max_root_rise_m": tail["maximum_root_rise_m"],
        })
    with csv_path.open("w", newline="") as f:
        writer_csv = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        writer_csv.writeheader()
        writer_csv.writerows(csv_rows)

    summary = {
        "status": "COMPLETE",
        "protocol": args.protocol,
        "condition": out.name,
        "job_id": args.job_id,
        "source_revision": spec["source_revision"],
        "dataset_revision": spec["dataset_revision"],
        "input_controls": str(controls_path),
        "input_reference": str(reference_path),
        "output_trajectory": str(out / "position_probe_trajectories.npz"),
        "output_metrics_csv": str(csv_path),
        "output_video": str(video_path),
        "video": {"frames": TOTAL_STEPS // 2, "fps": 50, "tile_grid": [3, 3], "tile_size_px": list(tile_size), "label_units": "mm", "sim_panel_only": True},
        "initialization": {
            "setup_env_calls": 1,
            "worlds": 9,
            "grid_order": "x-major, y-minor",
            "xy_offsets_m": grid.tolist(),
            "pair_margin_m": PAIR_MARGIN,
            "only_object_xy_changed_after_setup": True,
            "qvel_preserved_exactly_after_xy_offsets": True,
            "time_preserved_exactly_after_xy_offsets": True,
        },
        "execution": {"steps": TOTAL_STEPS, "same_control_for_all_worlds": True, "optimizer_called": False, "sim_dt_s": sim_dt, "time_saved_relative_to_setup_env": True},
        "real_reference_window": {"saved_output_rows": [0, REAL_STEPS], "physical_steps": [1, REAL_STEPS], "reference_rows": [1, REAL_STEPS + 1], "samples": REAL_STEPS},
        "padded_tail_steps": TOTAL_STEPS - REAL_STEPS,
        "worlds": worlds,
        "claim_boundary": "Fixed 5 mm XY initial-position grid; lift and tracking proxies only, not grasp/task success.",
        "elapsed_seconds": time.time() - started if "started" in globals() else None,
    }
    (out / "condition_result.json").write_text(json.dumps(summary, indent=2) + "\n")
    progress_path.write_text(json.dumps({"stage": "COMPLETE", "completed_steps": TOTAL_STEPS, "grid_worlds": 9}) + "\n")
    print(json.dumps({"status": "COMPLETE", "output": str(out), "worlds": len(worlds), "video": str(video_path)}, indent=2))
    return 0


if __name__ == "__main__":
    started = time.time()
    raise SystemExit(main())

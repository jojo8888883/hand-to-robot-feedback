#!/usr/bin/env python3
"""Materialize the revision-pinned OakInk-v2 trial inputs and optionally rsync them."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPIDER_DIR = HERE.parent
MANIFEST = HERE / "asset_manifest.json"
DATASET_DIR = SPIDER_DIR / "v2_dataset"
REMOTE_HOST = "dgvlaA800"
REMOTE_ROOT = "/mnt/sdc/dty_user/hand_to_robot_20261002/native_spider/v2_dataset"
ROBOT_SOURCE_ROOT = SPIDER_DIR / "example_datasets"


def run(command: list[str]) -> str:
    return subprocess.run(command, check=True, text=True, capture_output=True).stdout


def materialize(manifest: dict) -> dict:
    entries = manifest["minimum_direct_runner_assets"]["files"]
    metadata = manifest["reference_metadata_not_needed_by_scene_or_kinematic_input"]
    link = next(x for x in entries if x.get("kind") == "symlink_blob")
    local_link = DATASET_DIR / link["path"]
    local_link.parent.mkdir(parents=True, exist_ok=True)
    if local_link.exists() or local_link.is_symlink():
        if not local_link.is_symlink() or str(local_link.readlink()) != link["link_target"]:
            raise RuntimeError(f"unexpected existing path at {local_link}")
    else:
        local_link.symlink_to(link["link_target"])

    reused_robot_meshes = []
    downloaded = []
    for item in entries + metadata:
        if item.get("kind") == "symlink_blob":
            continue
        rel = item["path"]
        dest = DATASET_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        src = ROBOT_SOURCE_ROOT / rel
        if rel.startswith("processed/oakink/assets/robots/xhand/assets/") and src.is_file():
            if src.stat().st_size != item["size_bytes"]:
                raise RuntimeError(f"reusable robot mesh size mismatch: {src}")
            shutil.copyfile(src, dest)
            reused_robot_meshes.append(rel)
        else:
            url = item.get("url")
            if not url:
                raise RuntimeError(f"missing pinned URL for {rel}")
            request = urllib.request.Request(url, headers={"User-Agent": "native-v2-asset-prep/1.0"})
            with urllib.request.urlopen(request, timeout=45) as response:
                data = response.read()
            if len(data) != item["size_bytes"]:
                raise RuntimeError(
                    f"download size mismatch for {rel}: got {len(data)}, expected {item['size_bytes']}"
                )
            dest.write_bytes(data)
            downloaded.append({"path": rel, "bytes": len(data)})

    checked = []
    for item in entries + metadata:
        if item.get("kind") == "symlink_blob":
            p = DATASET_DIR / item["path"]
            if not p.is_symlink() or str(p.readlink()) != item["link_target"]:
                raise RuntimeError(f"symlink mismatch at {p}")
            checked.append({"path": item["path"], "kind": "symlink", "target": item["link_target"]})
        else:
            p = DATASET_DIR / item["path"]
            if not p.is_file() or p.stat().st_size != item["size_bytes"]:
                raise RuntimeError(f"local asset missing or wrong size: {p}")
            checked.append({"path": item["path"], "kind": "file", "size_bytes": p.stat().st_size})

    return {
        "asset_entries": checked,
        "regular_file_count": sum(x["kind"] == "file" for x in checked),
        "symlink_count": sum(x["kind"] == "symlink" for x in checked),
        "regular_file_bytes": sum(x.get("size_bytes", 0) for x in checked),
        "downloaded": downloaded,
        "reused_robot_meshes": reused_robot_meshes,
        "unexpected_saved_outputs_present": [
            str(p.relative_to(DATASET_DIR))
            for p in DATASET_DIR.rglob("*")
            if p.is_file()
            and p.name in {
                "trajectory_mjwp_fast.npz",
                "trajectory_ikrollout.npz",
                "visualization_mjwp_fast.mp4",
                "visualization_ik.mp4",
            }
        ],
    }


def sync_and_verify(local: dict) -> dict:
    ssh_opts = ["-o", "IPQoS=none", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
    remote_parent = str(Path(REMOTE_ROOT).parent)
    run(["ssh", *ssh_opts, REMOTE_HOST, "mkdir", "-p", remote_parent])
    rsync_ssh = "ssh -o IPQoS=none -o BatchMode=yes -o ConnectTimeout=15"
    run(["rsync", "-av", "--links", "-e", rsync_ssh, f"{DATASET_DIR}/", f"{REMOTE_HOST}:{REMOTE_ROOT}/"])

    remote_cmd = (
        f"cd {shlex.quote(REMOTE_ROOT)} && "
        "find . -type f -printf '%P\\t%s\\n' && "
        "printf '__ROBOT_LINK__\\t%s\\n' \"$(readlink processed/oakinkv2/assets/robots)\""
    )
    output = run(["ssh", *ssh_opts, REMOTE_HOST, remote_cmd])
    actual_files: dict[str, int] = {}
    robot_link = None
    for line in output.splitlines():
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        if parts[0] == "__ROBOT_LINK__":
            robot_link = parts[1]
        else:
            actual_files[parts[0].removeprefix("./")] = int(parts[1])
    expected_files = {
        x["path"]: x["size_bytes"]
        for x in local["asset_entries"]
        if x["kind"] == "file"
    }
    expected_link = next(x["target"] for x in local["asset_entries"] if x["kind"] == "symlink")
    entries = [
        {"path": p, "expected_size_bytes": size, "actual_size_bytes": actual_files.get(p), "ok": actual_files.get(p) == size}
        for p, size in expected_files.items()
    ]
    remote_check = {
        "all_expected_files_ok": all(x["ok"] for x in entries),
        "file_entries": entries,
        "robot_link_target": robot_link,
        "robot_link_ok": robot_link == expected_link,
        "unexpected_remote_files": sorted(set(actual_files) - set(expected_files)),
    }
    if not remote_check["all_expected_files_ok"] or not remote_check["robot_link_ok"]:
        raise RuntimeError("remote asset verification failed")
    return {"rsync": "completed_without_delete", "remote_check": remote_check}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync-remote", action="store_true", help="rsync to the configured remote and verify sizes/link")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    started = time.time()
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    local = materialize(manifest)
    if local["unexpected_saved_outputs_present"]:
        raise RuntimeError("saved-run outputs unexpectedly mixed into the dataset")
    result = {
        "status": "LOCAL_READY",
        "dataset_revision": manifest["dataset"]["pinned_revision"],
        "source_commit": manifest["official_code"]["commit"],
        "local_dataset_dir": str(DATASET_DIR),
        "remote_dataset_dir": REMOTE_ROOT if args.sync_remote else None,
        **local,
        "remote": None,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    if args.sync_remote:
        result["remote"] = sync_and_verify(local)
        result["status"] = "LOCAL_AND_REMOTE_READY"
        result["elapsed_seconds"] = round(time.time() - started, 3)
    (HERE / "DOWNLOAD_RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "regular_file_count", "symlink_count", "regular_file_bytes", "remote_dataset_dir", "elapsed_seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"download_inputs: {exc}", file=sys.stderr)
        raise

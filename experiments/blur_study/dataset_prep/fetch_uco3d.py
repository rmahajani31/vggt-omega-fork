"""Fetch the manifest recordings from the authors' unified uCO3D re-annotation using the study shard index."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

import requests

from convert_common import MANIFEST, UNIFIED, manifest_rows
from unified_io import validate_scene

INDEX = Path(__file__).resolve().parent / "uco3d_shard_index.json"
REPO_URL = "https://huggingface.co/datasets/facebook/uco3d/resolve/main/vggt_omega_anno/{shard}.tar"
USER_AGENT = "vggt-blur-study-fetch/1.0"
VERSION = "uco3d-vggt-omega-anno-1"
NAME = re.compile(r"^(?P<seq>.+)_rgb_video_(?P<md5>[0-9a-f]{32})$")


class RateLimited(RuntimeError):
    def __init__(self, retry_after: float):
        super().__init__(f"HTTP 429, retry after {retry_after:.0f}s")
        self.retry_after = retry_after


def check_rate_limit(response):
    if response.status_code == 429:
        for header in ("Retry-After", "RateLimit-Reset", "X-RateLimit-Reset"):
            try:
                wait = float(response.headers[header])
            except (KeyError, ValueError):
                continue
            raise RateLimited(min(600.0, max(1.0, wait)))
        raise RateLimited(30.0)


def _safe_members(tar: tarfile.TarFile, root: str):
    for member in tar:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or (path.parts and path.parts[0] != root):
            raise ValueError(f"Unsafe tar member {member.name}")
        if not (member.isfile() or member.isdir()):
            raise ValueError(f"Unexpected tar member type {member.name}")
        yield member


def download_and_extract(entry: dict, staging: Path, session) -> Path:
    """Stream one indexed archive into staging and return its scene directory."""
    url = REPO_URL.format(shard=entry["shard"])
    start, end = entry["offset"], entry["offset"] + entry["size"] - 1
    root = entry["member"][: -len(".tar")]
    with session.get(url, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=120) as r:
        check_rate_limit(r)
        if r.status_code != 206:
            raise RuntimeError(f"Range request failed with {r.status_code}")
        r.raw.decode_content = True
        with tarfile.open(fileobj=r.raw, mode="r|") as tar:
            for member in _safe_members(tar, root):
                tar.extract(member, staging, filter="data")
    return staging / root


def fetch_one(catalog_id: str, entry: dict, dest: Path, records: Path, session) -> str:
    final = dest / catalog_id
    record = records / f"{catalog_id}.json"
    if final.is_dir() and record.is_file() and json.loads(record.read_text()).get("converter_version") == VERSION:
        return "skipped"
    # Short staging name: nested sequence paths otherwise approach the Windows path limit.
    staging = dest.parent / "_staging" / "uco3d" / f"{NAME.match(catalog_id)['md5'][:12]}.part"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    scene = download_and_extract(entry, staging, session)
    validation = validate_scene(scene)
    if validation["errors"]:
        raise ValueError(f"{catalog_id} failed validation: {validation['errors'][:3]}")
    if final.exists():
        shutil.rmtree(final)
    dest.mkdir(parents=True, exist_ok=True)
    scene.rename(final)
    shutil.rmtree(staging)
    records.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({
        "scene": catalog_id, "status": "converted", "converter_version": VERSION,
        "source": {"repo": "facebook/uco3d", "path": f"vggt_omega_anno/{entry['shard']}.tar",
                   "member": entry["member"], "offset": entry["offset"], "size": entry["size"]},
        "validation": validation, "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=1), encoding="utf-8")
    return "converted"


def fetch(args) -> None:
    if not INDEX.is_file():
        raise SystemExit(f"Missing study shard index: {INDEX}")
    index = json.loads(INDEX.read_text())
    ids = [r["catalog_id"] for r in manifest_rows("uCO3D", args.manifest)]
    if args.scenes:
        ids = [i for i in ids if i in set(args.scenes)]
    missing = [i for i in ids if i not in index["members"]]
    if missing:
        raise SystemExit(f"{len(missing)} ids are not in the index: {missing[:5]}")
    total = sum(index["members"][i]["size"] for i in ids)
    print(f"{len(ids)} sequences, {total / 1e9:.1f} GB from huggingface.co/datasets/facebook/uco3d "
          f"({args.workers} parallel streams)")
    dest = Path(args.dest).resolve()
    records = dest.parent / "_records" / "uco3d"
    local = threading.local()

    def session():
        if not hasattr(local, "session"):
            local.session = requests.Session()
            local.session.headers["User-Agent"] = USER_AGENT
        return local.session

    def work(catalog_id):
        failures = 0
        while failures < 4:
            try:
                return catalog_id, fetch_one(catalog_id, index["members"][catalog_id], dest, records, session())
            except RateLimited as e:  # not counted as a failure: wait as instructed and retry
                print(f"   {catalog_id}: rate limited, waiting {e.retry_after:.0f}s", flush=True)
                time.sleep(e.retry_after)
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"   {catalog_id}: attempt {failures} failed: {type(e).__name__}: {e}", flush=True)
                time.sleep(10 * failures)
        return catalog_id, "failed"

    counts = {"converted": 0, "skipped": 0, "failed": 0}
    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for k, future in enumerate(as_completed([pool.submit(work, i) for i in ids])):
            catalog_id, status = future.result()
            counts[status] += 1
            print(f"[{k + 1}/{len(ids)}] {catalog_id}: {status} ({(time.time() - started) / 60:.1f} min)",
                  flush=True)
    print(f"uco3d: {counts}")
    if counts["failed"]:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fetch",))
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--dest", type=Path, default=UNIFIED / "uco3d")
    parser.add_argument("--scenes", nargs="+")
    parser.add_argument("--workers", type=int, default=4, help="parallel download streams")
    args = parser.parse_args()
    fetch(args)


if __name__ == "__main__":
    main()

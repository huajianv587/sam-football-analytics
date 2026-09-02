#!/usr/bin/env python3
"""Download a fixed model artifact with verified byte ranges and checksum."""

import argparse
import hashlib
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


CHUNK_SIZE = 8 * 1024 * 1024


def request_range(url: str, start: int, end: int) -> tuple[bytes, int]:
    for attempt in range(20):
        try:
            request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 206:
                    raise RuntimeError(f"server ignored byte range: HTTP {response.status}")
                total = int(response.headers["Content-Range"].rsplit("/", 1)[1])
                data = response.read()
                if len(data) != end - start + 1:
                    raise RuntimeError(f"short range response: {len(data)} bytes")
                return data, total
        except (OSError, RuntimeError, urllib.error.HTTPError) as exc:
            if attempt == 19:
                raise RuntimeError(f"range {start}-{end} failed") from exc
            time.sleep(5)
    raise AssertionError("unreachable")


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path, expected_md5: str) -> None:
    if destination.is_file() and md5(destination) == expected_md5:
        print(f"verified {destination}")
        return
    if destination.exists():
        destination.unlink()
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.parent.mkdir(parents=True, exist_ok=True)
    if partial.exists():
        aligned = partial.stat().st_size // CHUNK_SIZE * CHUNK_SIZE
        with partial.open("r+b") as handle:
            handle.truncate(aligned)
    start = partial.stat().st_size if partial.exists() else 0
    _, total = request_range(url, 0, 0)
    while start < total:
        end = min(total - 1, start + CHUNK_SIZE - 1)
        data, reported_total = request_range(url, start, end)
        if reported_total != total:
            raise RuntimeError("remote model size changed during download")
        with partial.open("ab") as handle:
            handle.write(data)
        start = end + 1
        print(f"{destination.name}: {start / total:.1%}", flush=True)
    actual_md5 = md5(partial)
    if actual_md5 != expected_md5:
        raise RuntimeError(f"checksum mismatch for {destination.name}: {actual_md5}")
    os.replace(partial, destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("destination", type=Path)
    parser.add_argument("md5")
    arguments = parser.parse_args()
    download(arguments.url, arguments.destination, arguments.md5)

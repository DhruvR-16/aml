"""Download the official UJIIndoorLoc archive from the UCI repository."""

from __future__ import annotations

import argparse
import hashlib
import io
import ssl
import urllib.request
import zipfile
from pathlib import Path

import certifi

URL = "https://archive.ics.uci.edu/static/public/310/ujiindoorloc.zip"
EXPECTED = {"trainingData.csv", "validationData.csv"}


def download(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(URL, headers={"User-Agent": "aml-project/1.0"})
    with urllib.request.urlopen(
        request, timeout=60, context=ssl.create_default_context(cafile=certifi.where())
    ) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        found = {Path(name).name: name for name in archive.namelist()}
        if not EXPECTED.issubset(found):
            raise RuntimeError(f"UCI archive did not contain {sorted(EXPECTED)}")
        for filename in sorted(EXPECTED):
            payload = archive.read(found[filename])
            output = destination / filename
            output.write_bytes(payload)
            print(
                f"Wrote {output} ({len(payload):,} bytes; sha256 {hashlib.sha256(payload).hexdigest()})"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/raw",
    )
    download(parser.parse_args().output_dir)

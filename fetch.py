"""Download the selected replay data for one experiment into its outputs/ folder."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parent
RELEASES = "https://github.com/dimentary/llm-robotics-playground/releases/download"


def main():
    manifest = json.loads((ROOT / "recordings.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=manifest["artifacts"])
    args = parser.parse_args()
    recording = manifest["artifacts"][args.experiment]
    asset = recording["replay"]
    output = ROOT / "experiments" / args.experiment / "outputs"
    expected = set(recording["replay_files"])
    existing = sorted(name for name in expected if (output / name).exists())
    if existing:
        parser.error(
            "Replay files already exist in outputs/. Move them aside first: "
            + ", ".join(existing)
        )
    url = f"{RELEASES}/{manifest['tag']}/{asset['file']}"
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / asset["file"]
        digest = hashlib.sha256()
        print(f"Downloading {asset['file']} …", flush=True)
        request = urllib.request.Request(
            url, headers={"User-Agent": "llm-robotics-playground"}
        )
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            archive.open("wb") as target,
        ):
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
                digest.update(chunk)
        if (
            archive.stat().st_size != asset["bytes"]
            or digest.hexdigest() != asset["sha256"]
        ):
            raise RuntimeError(
                "Download checksum/size mismatch; no files were extracted."
            )
        with tarfile.open(archive, "r:gz") as bundle:
            members = bundle.getmembers()
            if len(members) != len(expected) or {m.name for m in members} != expected:
                raise RuntimeError("Unexpected replay bundle contents.")
            if any(not m.isfile() or Path(m.name).name != m.name for m in members):
                raise RuntimeError(
                    "Replay bundle must contain flat regular files only."
                )
            output.mkdir(parents=True, exist_ok=True)
            bundle.extractall(output, filter="data")
    print(f"Ready: {output}")


if __name__ == "__main__":
    main()

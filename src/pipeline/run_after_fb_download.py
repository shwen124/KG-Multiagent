"""Wait for Zenodo zip, verify MD5, then run CWQ FB+CVT-REV pipeline stages."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import time
from pathlib import Path

# Avoid Windows console UnicodeEncodeError on non-ASCII paths.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
ZIP = ROOT / "data" / "Freebase" / "raw" / "idirlab-freebases.zip"
EXPECTED_MD5 = "170689b7aad9f029566a4deb36605b01"
EXPECTED_SIZE = 14148416296


def md5_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def wait_for_zip(poll_sec: int = 60) -> None:
    print(f"Waiting for {ZIP} ...", flush=True)
    while True:
        if ZIP.exists():
            size = ZIP.stat().st_size
            pct = 100.0 * size / EXPECTED_SIZE
            print(f"  size={size/1e9:.2f} GB ({pct:.1f}%)", flush=True)
            if size >= EXPECTED_SIZE * 0.999:
                # give curl a moment to finish closing
                time.sleep(5)
                if ZIP.stat().st_size >= EXPECTED_SIZE * 0.999:
                    return
        time.sleep(poll_sec)


def main() -> None:
    wait_for_zip()
    size = ZIP.stat().st_size
    print(f"Download appears complete: {size/1e9:.2f} GB", flush=True)
    print("Computing MD5 (may take several minutes) ...", flush=True)
    digest = md5_file(ZIP)
    print(f"MD5={digest}", flush=True)
    if digest.lower() != EXPECTED_MD5.lower():
        print("MD5 MISMATCH — aborting. Re-download with curl -C -", flush=True)
        sys.exit(2)
    print("MD5 OK", flush=True)

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.pipeline.process_cwq_fb_cvt_rev",
        "--stage",
        "all",
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    main()

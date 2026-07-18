"""Fetch the BootsTAPIR tracker (Apache-2.0) into third_party/tapir.

The default learned tracker is BootsTAPIR — commercial-safe, Apache-2.0. Its
code and 209 MB checkpoint are NOT vendored in git (kept out to respect the
upstream repo and GitHub's file-size limit), so run this once after cloning,
before `python ui.py` or a PyInstaller build:

    python setup_tracker.py

It clones ibaiGorordo/Tapir-Pytorch-Inference (Apache-2.0) and downloads the
dm-tapnet causal checkpoint (Apache-2.0). Re-running is a no-op if both exist.
Without this, the pipeline falls back to the classic KLT front-end (still
works, weaker on soft footage) or --tracker cotracker (non-commercial).
"""
import os
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TP = os.path.join(HERE, "third_party", "tapir")
REPO = "https://github.com/ibaiGorordo/Tapir-Pytorch-Inference.git"
CKPT_URL = ("https://storage.googleapis.com/dm-tapnet/"
            "causal_bootstapir_checkpoint.pt")
CKPT = os.path.join(TP, "models", "causal_bootstapir_checkpoint.pt")


def main():
    os.makedirs(os.path.dirname(TP), exist_ok=True)
    if not os.path.exists(os.path.join(TP, "tapnet", "tapir_inference.py")):
        print(f"[setup] cloning tracker repo -> {TP}")
        subprocess.run(["git", "clone", "--depth", "1", REPO, TP], check=True)
    else:
        print("[setup] tracker repo already present")

    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    if os.path.exists(CKPT) and os.path.getsize(CKPT) > 1_000_000:
        print("[setup] checkpoint already present")
    else:
        print(f"[setup] downloading checkpoint (~209 MB) -> {CKPT}")

        def _progress(n, bs, total):
            if total > 0:
                pct = min(100, 100 * n * bs / total)
                sys.stdout.write(f"\r  {pct:5.1f}%")
                sys.stdout.flush()
        urllib.request.urlretrieve(CKPT_URL, CKPT, _progress)
        print()
    print("[setup] done — BootsTAPIR ready (Apache-2.0, commercial-safe)")


if __name__ == "__main__":
    main()

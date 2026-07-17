"""
AeroTrack frozen-app entry point.
=================================
Default: launches the UI (native window).
`AeroTrack.exe --run <module> [args…]`: dispatches to a pipeline module —
this is how the frozen app runs its own subprocesses (there is no separate
python.exe inside a PyInstaller bundle; the exe re-invokes itself).
"""

import os
import runpy
import sys


def _ensure_streams():
    """Windowed (no-console) exes have sys.stdout/stderr = None; any print()
    would crash.

    When this exe is re-invoked as a pipeline step (``--run <module>``) the
    parent captures our output through a PIPE — i.e. OS fd 1/2 are valid even
    though Python's sys.stdout is None. The render-queue progress bar reads
    that piped output live, so attach to the real fds FIRST. Only fall back
    to a log file when there is genuinely no usable stdout (the main windowed
    app, launched with no console and no pipe)."""
    if sys.stdout is not None and sys.stderr is not None:
        return

    def _from_fd(fd):
        try:
            s = os.fdopen(fd, "w", buffering=1, encoding="utf-8",
                          errors="replace")
            s.write("")
            s.flush()
            return s
        except Exception:
            return None

    if sys.stdout is None:
        sys.stdout = _from_fd(1)
    if sys.stderr is None:
        sys.stderr = _from_fd(2)
    if sys.stdout is not None and sys.stderr is not None:
        return

    # no usable pipe/console — log to a file so print() never crashes
    try:
        f = open(os.path.join(os.path.dirname(sys.executable),
                              "aerotrack_log.txt"), "a",
                 encoding="utf-8", buffering=1)
    except OSError:
        f = open(os.devnull, "w")
    if sys.stdout is None:
        sys.stdout = f
    if sys.stderr is None:
        sys.stderr = f


def main():
    _ensure_streams()
    if len(sys.argv) > 2 and sys.argv[1] == "--run":
        module = sys.argv[2]
        sys.argv = [module] + sys.argv[3:]
        runpy.run_module(module, run_name="__main__")
        return
    runpy.run_module("ui", run_name="__main__")


if __name__ == "__main__":
    main()

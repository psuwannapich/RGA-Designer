"""Run model-generated code in a killable subprocess (a thread cannot be stopped on timeout)."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

_KILL_GRACE = 2.0


def _kill(proc):
    try:
        os.killpg(os.getpgid(proc.pid), 9)   # the whole group: the code may spawn children
    except (ProcessLookupError, PermissionError):
        proc.kill()
    proc.communicate(timeout=_KILL_GRACE)


def run_snippet(code: str, timeout: int = 5) -> tuple[bool, str]:
    """Run *code* in a fresh interpreter; returns (exited with 0, output) or (False, "TIMEOUT")."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(code)
        path = fh.name
    try:
        proc = subprocess.Popen([sys.executable, "-I", "-S", path], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, start_new_session=True)
        try:
            out, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill(proc)
            return False, "TIMEOUT"
        return proc.returncode == 0, (out or "").strip()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# The result goes to its own file, so printed output can't be mistaken for it.
# No -S: generated code may import site-packages (e.g. sympy).
_CAPTURE_WRAPPER = r"""
import sys
src_path, out_path = sys.argv[1], sys.argv[2]
with open(src_path) as fh:
    src = fh.read()
ns = {}
status, payload = "NONE", ""
try:
    exec(src, {}, ns)
except Exception as e:
    status, payload = "ERR", str(e)
else:
    if "answer" in ns:
        try:
            status, payload = "OK", str(ns["answer"])
        except Exception as e:
            status, payload = "ERR", str(e)
with open(out_path, "w") as fh:
    fh.write(status + "\n" + payload)
"""


def run_snippet_capture(code: str, timeout: int = 5) -> tuple[str, str]:
    """Run *code* and return the value it bound to `answer`.

    Returns (status, payload): "OK" with str(answer), "NONE" (no `answer`),
    "ERR" with the exception text, or "TIMEOUT".
    """
    src_path = wrap_path = out_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(code)
            src_path = fh.name
        with tempfile.NamedTemporaryFile("w", suffix="_wrap.py", delete=False) as fh:
            fh.write(_CAPTURE_WRAPPER)
            wrap_path = fh.name
        with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as fh:
            out_path = fh.name
        proc = subprocess.Popen([sys.executable, "-I", wrap_path, src_path, out_path],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                start_new_session=True)
        try:
            proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill(proc)
            return "TIMEOUT", ""
        try:
            with open(out_path) as fh:
                raw = fh.read()
        except OSError:
            return "ERR", "no result written"
        status, _, payload = raw.partition("\n")
        return (status or "ERR"), payload
    finally:
        for p in (src_path, wrap_path, out_path):
            if p:
                try:
                    os.unlink(p)
                except OSError:
                    pass

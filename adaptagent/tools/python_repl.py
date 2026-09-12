"""Sandboxed Python interpreter: restricted builtins, no network, timeouts."""

from __future__ import annotations

import multiprocessing
import sys
from typing import Any

from .base import Tool, tool


def _run_code(code: str, result_queue) -> None:  # type: ignore[valid-type]
    import io
    import contextlib

    captured = io.StringIO()
    allowed_builtins = {
        "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict, "divmod": divmod,
        "enumerate": enumerate, "filter": filter, "float": float, "int": int, "isinstance": isinstance,
        "len": len, "list": list, "map": map, "max": max, "min": min, "print": print, "range": range,
        "repr": repr, "reversed": reversed, "round": round, "set": set, "sorted": sorted,
        "str": str, "sum": sum, "tuple": tuple, "type": type, "zip": zip, "True": True, "False": False,
        "None": None,
    }
    import math as _math
    import random as _random
    import json as _json
    import statistics as _statistics
    import datetime as _datetime
    modules = {
        "math": _math, "random": _random, "json": _json,
        "statistics": _statistics, "datetime": _datetime,
    }
    globals_dict: dict[str, Any] = {"__builtins__": allowed_builtins, **modules}
    try:
        with contextlib.redirect_stdout(captured):
            exec(compile(code, "<sandbox>", "exec"), globals_dict)  # noqa: S102
        result_queue.put(("ok", captured.getvalue()))
    except SystemExit as exc:
        result_queue.put(("ok", f"SystemExit({exc.code})"))
    except BaseException as exc:  # noqa: BLE001
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _execute(code: str, timeout: int = 20) -> str:
    sys_platform = sys.platform
    if sys_platform == "win32":
        ctx = multiprocessing.get_context("spawn")
    else:
        ctx = multiprocessing.get_context("fork")
    q: Any = ctx.SimpleQueue()
    p = ctx.Process(target=_run_code, args=(code, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return "Error: timeout"
    try:
        status, output = q.get()
        return output if output else "(no output)"
    except Exception:
        return "(process finished without output)"


@tool
def python_repl(code: str) -> str:
    """Execute Python code in a secure sandbox.

    code: Python source code to execute. print() output is returned. \
    Allowed modules: math, random, json, statistics, datetime. No network, no filesystem.
    """
    return _execute(code, timeout=20)


def PythonREPLTool() -> Tool:
    return python_repl

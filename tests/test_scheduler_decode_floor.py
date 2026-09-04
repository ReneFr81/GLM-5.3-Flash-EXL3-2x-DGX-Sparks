#!/usr/bin/env python3
"""Apply overlay/patch_scheduler_decode_floor.py to a copy of scheduler.py."""
from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PATCH = next(
    p
    for p in (
        HERE / "patch_scheduler_decode_floor.py",
        HERE.parent / "overlay" / "patch_scheduler_decode_floor.py",
    )
    if p.is_file()
)
SRC = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py"
)


def main() -> int:
    if not PATCH.is_file():
        raise SystemExit(f"missing {PATCH}")
    src = Path(os.environ.get("GLM53_SCHEDULER_PY_SRC", SRC))
    if not src.is_file():
        # Host unit test: copy from a live container if present.
        raise SystemExit(f"missing scheduler.py at {src}")
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "scheduler.py"
        shutil.copyfile(src, dst)
        env = os.environ.copy()
        env["GLM53_SCHEDULER_PY"] = str(dst)
        env["GLM53_MIXED_PREFILL_CHUNK"] = "skip"
        subprocess.check_call([sys.executable, str(PATCH)], env=env)
        text = dst.read_text()
        assert "[glm53-decode-floor]" in text
        assert text.count("[glm53-decode-floor]") == 2
        assert "def _glm53_mixed_prefill_policy(" in text
        assert "[glm53-bounded-fairness]" in text
        tree = ast.parse(text)
        helper_node = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_glm53_mixed_prefill_policy"
        )
        namespace = {"os": os, "time": __import__("time")}
        helper_module = ast.Module(body=[helper_node], type_ignores=[])
        exec(compile(helper_module, str(dst), "exec"), namespace)
        policy = namespace["_glm53_mixed_prefill_policy"]
        decoder = SimpleNamespace(
            request_id="decode", num_computed_tokens=101, num_prompt_tokens=100
        )
        fresh = SimpleNamespace(
            request_id="fresh", num_computed_tokens=0, num_prompt_tokens=100,
            arrival_time=980.0,
        )
        old = SimpleNamespace(
            request_id="old", num_computed_tokens=0, num_prompt_tokens=100,
            arrival_time=969.0,
        )
        with patch.dict(
            os.environ,
            {
                "GLM53_MIXED_PREFILL_CHUNK": "skip",
                "GLM53_MIXED_PREFILL_MAX_WAIT_S": "30",
            },
        ), patch.object(namespace["time"], "time", return_value=1000.0):
            assert policy([decoder], fresh) == 0
            assert policy([decoder], old) is None
            assert policy([decoder], decoder) is None
        # idempotent
        subprocess.check_call([sys.executable, str(PATCH)], env=env)
        assert dst.read_text().count("[glm53-decode-floor]") == 2
    print("scheduler decode-floor patch OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

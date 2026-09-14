#!/usr/bin/env python3
"""Exercise fresh and already-patched scheduler installs without importing vLLM."""
import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
PATCH = Path(os.environ.get("GLM53_PATCH_SCRIPT", next(
    str(p) for p in (HERE / "patch_scheduler_decode_floor.py", HERE.parent / "overlay/patch_scheduler_decode_floor.py") if p.is_file()
)))
SPEC = importlib.util.spec_from_file_location("scheduler_patch", PATCH)
PATCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCHER)


def fixture(helper, running, waiting):
    return (
        "import itertools\nimport time\nimport os\n" + helper
        + "from vllm.compilation.cuda_graph import CUDAGraphStat\n"
        + "class Scheduler:\n    def run(self):\n        for request in self.running:\n"
        + running + "            pass\n"
        + "    def wait(self):\n        if True:\n            if True:\n                for request in self.running:\n"
        + waiting + "                    pass\n"
    )


def without_peer(helper):
    return helper[helper.index("\n_GLM53_GATE_CFG = None"):]


class PeerHelperTests(unittest.TestCase):
    def patched(self, source):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scheduler.py"
            path.write_text(source)
            env = {**os.environ, "GLM53_SCHEDULER_PY": str(path)}
            result = subprocess.run([sys.executable, str(PATCH)], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            actual = path.read_text()
            ast.parse(actual)
            result = subprocess.run([sys.executable, str(PATCH)], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(path.read_text(), actual, "second application must be byte-identical")
            return actual

    def assert_helpers(self, text):
        tree = ast.parse(text)
        definitions = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id.startswith("_glm53_")}
        self.assertFalse(calls - definitions, f"undefined scheduler helpers: {calls - definitions}")
        self.assertIn("_glm53_has_decoding_peer", definitions)
        self.assertEqual(text.count("def _glm53_has_decoding_peer("), 1)
        helper_nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("_glm53_")]
        ns = {"os": os, "_GLM53_GATE_CFG": None, "_GLM53_FIRST_SEEN_FALLBACK": None}
        exec(compile(ast.Module(body=helper_nodes, type_ignores=[]), "installed-helpers", "exec"), ns)
        current = SimpleNamespace(request_id="current", num_prompt_tokens=32768, num_tokens=32768, num_computed_tokens=0)
        decoder = SimpleNamespace(request_id="decoder", num_prompt_tokens=100, num_computed_tokens=120)
        prefiller = SimpleNamespace(request_id="prefiller", num_prompt_tokens=1000, num_computed_tokens=12)
        peer = ns["_glm53_has_decoding_peer"]
        self.assertFalse(peer([], current))
        self.assertFalse(peer([current, prefiller], current))
        self.assertTrue(peer([current, decoder], current))
        self.assertFalse(peer([decoder], decoder))
        same_id = SimpleNamespace(**vars(decoder))
        self.assertFalse(peer([same_id], decoder))
        # Execute the installed threshold sites, not just text-marker checks.
        os.environ["GLM53_MIXED_PREFILL_CHUNK"] = "256"
        for site in (PATCHER.RUNNING_NEW, PATCHER.WAITING_NEW):
            self.assertIn(site, text)
            code = textwrap.dedent(site).replace("continue", "return 0")
            ns.update(request=current, num_computed_tokens=0, token_budget=7168, input_budget=7168, draft_slots=0)
            exec("def schedule_site(num_new_tokens):\n" + textwrap.indent(code, "    ") + "    return num_new_tokens\n", ns)
            ns["self"] = SimpleNamespace(running=[current], scheduler_config=SimpleNamespace(long_prefill_token_threshold=1024))
            self.assertEqual(ns["schedule_site"](7168), 7168)
            ns["self"].running.append(decoder)
            self.assertEqual(ns["schedule_site"](7168), 256)

    def test_fresh_install_has_all_callable_helpers(self):
        self.assert_helpers(self.patched(fixture("", PATCHER.RUNNING_OLD, PATCHER.WAITING_OLD)))

    def test_repair_d83_marked_but_missing_helper(self):
        self.assert_helpers(self.patched(fixture(without_peer(PATCHER.HELPER), PATCHER.RUNNING_NEW, PATCHER.WAITING_NEW)))

    def test_upgrade_prior_v2_threshold_sites(self):
        running = PATCHER.RUNNING_NEW.replace(
            "if (\n                0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens\n                and _glm53_has_decoding_peer(self.running, request)\n            ):",
            "if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:",
        )
        waiting = PATCHER.WAITING_NEW.replace(" and _glm53_has_decoding_peer(self.running, request)", "")
        self.assert_helpers(self.patched(fixture(without_peer(PATCHER.HELPER), running, waiting)))

    def test_drift_fails_without_modifying_input(self):
        source = fixture(without_peer(PATCHER.HELPER), PATCHER.RUNNING_NEW.replace("input_budget", "drifted_budget"), PATCHER.WAITING_NEW)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scheduler.py"
            path.write_text(source)
            result = subprocess.run([sys.executable, str(PATCH)], env={**os.environ, "GLM53_SCHEDULER_PY": str(path)}, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(path.read_text(), source)


if __name__ == "__main__":
    unittest.main()

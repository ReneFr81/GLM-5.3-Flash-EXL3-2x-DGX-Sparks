"""Generate an explicit TP3 overlay beside an already measured native bundle.

No downloads, lifecycle commands, model changes or default configuration edits.
"""
import argparse
import hashlib
from pathlib import Path

from manifest import verify_artifacts

# Reviewed repin (thin-decode split from #182). overlay/exl3.py gained the
# opt-in GLM53_EXL3_MOE_FAST dispatch, the gate/up SUH pointer alias (created
# only in fast mode, whose equality is proven per expert at load, so the aliased
# table is content-identical) and the FAST=1 fail-closed raises. With the flag
# unset -- the launcher default, and what start-tp3.sh enforces by unsetting it
# -- the module builds exactly the pointer tables and takes exactly the paths it
# did before, so a profile generated from this source behaves like the
# previously pinned one. Refusal on any further drift is unchanged.
STOCK_SHA = "7677ab42f4a20698371b5c22d27ecb1b0b416a6860d00a137e13f25e9fd0ed40"


def prepare(stock, bundle):
    stock, bundle = Path(stock), Path(bundle)
    source = stock.read_bytes()
    if hashlib.sha256(source).hexdigest() != STOCK_SHA:
        raise ValueError("stock EXL3 source changed; review compatibility before repinning")
    verify_artifacts(bundle)
    footer = '''
# Explicit TP3 ABI2 cooperative adapter; complete manifest required on all ranks.
import runpy as _coop_runpy
import sys as _coop_sys
_coop_setup = _coop_runpy.run_path("/root/.cache/vllm/cooperative_moe/runtime.py")
_coop_setup["install"](_coop_sys.modules[__name__], library_root="/root/.cache/vllm/cooperative_moe", enabled=True)
'''
    output = bundle / "exl3-tp3.py"
    with output.open("xb") as handle:
        handle.write(source + footer.encode())
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.stock, args.bundle))

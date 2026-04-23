"""Export a trained PPO checkpoint to firmware Q16 weights.

Usage:
    python -m PySaacSim.scripts.export_model <checkpoint.zip>
        [--model-c PATH]   # firmware Model.c to patch (default: auto-detect)
        [--dry-run]        # print the patched file to stdout, don't write
        [-o PATH]          # write a standalone C fragment instead of patching

Reads a Stable-Baselines3 PPO `.zip` (linear policy, net_arch=[]), extracts
the 3x13 weight matrix and 3-vector bias, quantizes to Q16, and rewrites the
`Model_Weights` / `Model_Bias` initializers inside firmware's Model.c in place.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from stable_baselines3 import PPO

from ..gui.training.linear_policy import extract_weights, export_firmware
from ..sim.model import INPUT_NAMES, OUTPUT_NAMES


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODEL_C = _REPO_ROOT / "lab-7-tweinstein-1" / "RTOS_SensorBoard" / "Model.c"

_WEIGHTS_RE = re.compile(
    r"const fixed_t Model_Weights\[NUM_OUTPUTS\]\[NUM_INPUTS\]\s*=\s*\{[^;]*\};[^\n]*",
    re.DOTALL,
)
_BIAS_RE = re.compile(
    r"const fixed_t Model_Bias\[NUM_OUTPUTS\]\s*=\s*\{[^;]*\};[^\n]*"
)


def _split_c_source(c_source: str) -> tuple[str, str]:
    """export_to_c_source emits `<weights>\\n\\n<bias>`; split them back."""
    parts = c_source.strip().split("\n\n", 1)
    if len(parts) != 2:
        raise RuntimeError("unexpected export_to_c_source layout")
    return parts[0].rstrip(), parts[1].rstrip()


def _patch_model_c(src: str, weights_block: str, bias_block: str) -> str:
    if not _WEIGHTS_RE.search(src):
        raise RuntimeError("could not locate Model_Weights initializer in Model.c")
    if not _BIAS_RE.search(src):
        raise RuntimeError("could not locate Model_Bias initializer in Model.c")
    # re.sub treats `\` in the replacement as an escape; use a lambda to pass it literally.
    src = _WEIGHTS_RE.sub(lambda _m: weights_block, src, count=1)
    src = _BIAS_RE.sub(lambda _m: bias_block, src, count=1)
    return src


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path, help="Path to the PPO .zip")
    p.add_argument("--model-c", type=Path, default=_DEFAULT_MODEL_C,
                   help=f"Firmware Model.c to patch (default: {_DEFAULT_MODEL_C})")
    p.add_argument("--dry-run", action="store_true",
                   help="Print patched Model.c to stdout instead of writing")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Write standalone C fragment here instead of patching Model.c")
    args = p.parse_args()

    if not args.checkpoint.exists():
        print(f"error: {args.checkpoint} does not exist", file=sys.stderr)
        return 1

    model = PPO.load(str(args.checkpoint), device="cpu")
    W, b = extract_weights(model)
    Wq, bq, c_source = export_firmware(W, b)

    print(f"// Loaded policy from {args.checkpoint.name}", file=sys.stderr)
    print(f"// W float range [{W.min():+.4f}, {W.max():+.4f}]  shape {W.shape}",
          file=sys.stderr)
    print(f"// b float range [{b.min():+.4f}, {b.max():+.4f}]  shape {b.shape}",
          file=sys.stderr)
    print(f"// Wq int range  [{Wq.min():+d}, {Wq.max():+d}]", file=sys.stderr)
    print(f"// Input order : {', '.join(INPUT_NAMES)}", file=sys.stderr)
    print(f"// Output order: {', '.join(OUTPUT_NAMES)}", file=sys.stderr)

    if args.output:
        args.output.write_text(c_source)
        print(f"// wrote standalone fragment to {args.output}", file=sys.stderr)
        return 0

    if not args.model_c.exists():
        print(f"error: {args.model_c} does not exist", file=sys.stderr)
        return 1

    weights_block, bias_block = _split_c_source(c_source)
    original = args.model_c.read_text()
    patched = _patch_model_c(original, weights_block, bias_block)

    if patched == original:
        print("// Model.c already matches these weights, nothing to do",
              file=sys.stderr)
        return 0

    if args.dry_run:
        sys.stdout.write(patched)
        print(f"// [dry-run] would patch {args.model_c}", file=sys.stderr)
        return 0

    args.model_c.write_text(patched)
    print(f"// patched {args.model_c}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""D2TA gate — minimal aggregation table.

Reads the target-eval JSON produced by `eval_gate.py` plus the per-group online
`_trace.json` files, and prints one compact table to decide whether the first
round passes the gate:

    full average ΔDice >= base average ΔDice, without either victim dropping.

No theory verification here — this is the only summary needed for round one.
"""

from __future__ import annotations

import argparse
import json
import os


def _load_trace(adv_dir):
    p = os.path.join(adv_dir, "_trace.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str, default="ddta/gate_results.json")
    ap.add_argument("--groups", type=str, default=None, help='JSON {"name": "adv_dir", ...} (for traces)')
    args = ap.parse_args()

    with open(args.results) as f:
        res = json.load(f)

    clean = res["clean"]
    groups = res["groups"]
    trace_map = {}
    if args.groups:
        gmap = json.loads(args.groups)
        trace_map = {name: _load_trace(adv_dir) for name, adv_dir in gmap.items()}

    order = ["base", "offline", "online", "full", "random"]

    def _fmt(v):
        return "  -  " if v is None else f"{v:+.4f}"

    header = f"{'method':<14} {'SAMed ΔDice':>12} {'Med-SA ΔDice':>13} {'avg ΔDice':>10} {'cos':>7} {'drift↓':>8}"
    print(header)
    print("-" * len(header))
    rows = []
    for name in order:
        if name not in groups:
            continue
        g = groups[name]
        dd_s = g["samed_ddice"]
        dd_m = g["medsa_ddice"]
        avg = (dd_s + dd_m) / 2.0
        tr = trace_map.get(name, {})
        cos = tr.get("mean_cos")
        drift_reduced = tr.get("drift_reduced")
        cos_s = "  -  " if cos is None else f"{cos:.3f}"
        drift_s = "  -  " if drift_reduced is None else ("yes" if drift_reduced else "no")
        print(f"{name:<14} {dd_s:>+12.4f} {dd_m:>+13.4f} {avg:>+10.4f} {cos_s:>7} {drift_s:>8}")
        rows.append((name, avg, dd_s, dd_m))

    print()
    print(f"clean: SAMed={clean['samed_dice']:.4f}  Med-SA={clean['medsa_dice']:.4f}")

    # gate verdict
    if rows:
        base = next((r for r in rows if r[0] == "base"), None)
        full = next((r for r in rows if r[0] == "full"), None)
        if base is not None and full is not None:
            verdict = "PASS (full avg ΔDice >= base)" if full[1] >= base[1] - 1e-9 else "FAIL (full < base)"
            print(f"\nGATE: {verdict}  (base avg ΔDice={base[1]:+.4f}, full avg ΔDice={full[1]:+.4f})")


if __name__ == "__main__":
    main()

"""
compare_to_layout.py

The "difference report": compares what the rover MEASURED during a mission
(the findings database written by mission_manager_node.py / segment_manager.py)
against the MANUALLY VERIFIED LAYOUT of the corridor (corridor_ground_truth.json).

For every segment it shows: reference width, true clearance, measured clearance,
the error, whether a finding was expected / detected, and the severity. It ends
with summary numbers that feed the evaluation plan (clearance error, coverage,
finding agreement).

Measured clearance of a segment = the SMALLEST clearance recorded in it (that is
where the obstruction is), or the last reading if the segment had no finding.

HOW TO RUN (from the backend folder, after a mission has finished):
    python3 compare_to_layout.py ../simulation/corridor_ground_truth.json
Optional:  --tolerance 0.10   (allowed clearance error in metres, default 0.10)
Writes difference_report.json next to this script.
"""
import argparse
import json
import os
import sqlite3
import sys

from segment_manager import DB_PATH, compute_severity

HERE = os.path.dirname(os.path.abspath(__file__))


def load_measured(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT segment_id, reference_clearance_m, surveyed, last_actual_clearance_m FROM segments")
    segs = {r[0]: {"reference": r[1], "surveyed": bool(r[2]), "last_actual": r[3]} for r in c.fetchall()}
    c.execute("SELECT segment_id, MIN(actual_clearance_m), MAX(severity), COUNT(*) FROM findings GROUP BY segment_id")
    for seg_id, min_clear, max_sev, n in c.fetchall():
        if seg_id in segs:
            segs[seg_id].update({"min_clearance": min_clear, "max_severity": max_sev, "n_findings": n})
    conn.close()
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ground_truth", nargs="?", default=None)
    ap.add_argument("--tolerance", type=float, default=0.10)
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args()

    gt_path = args.ground_truth
    if gt_path is None:
        for cand in ("corridor_ground_truth.json", os.path.join("..", "simulation", "corridor_ground_truth.json")):
            if os.path.exists(cand):
                gt_path = cand
                break
    if gt_path is None or not os.path.exists(gt_path):
        sys.exit("Could not find corridor_ground_truth.json - pass its path as the first argument.")

    truth = json.load(open(gt_path))["segments"]
    measured = load_measured(args.db)

    rows, errors = [], []
    surveyed_n = agree_n = sev_ok = sev_total = 0
    for seg_id, t in truth.items():
        m = measured.get(seg_id)
        ref, true_c = t["reference_clearance_m"], t["true_clearance_m"]
        exp_finding = (ref - true_c) > 0.02
        exp_sev = compute_severity(ref, true_c) if exp_finding else None
        if m is None or not m["surveyed"]:
            rows.append({"segment": seg_id, "reference_m": ref, "true_clearance_m": true_c, "measured_clearance_m": None,
                         "error_m": None, "finding_expected": exp_finding, "finding_detected": None,
                         "severity_expected": exp_sev, "severity_detected": None, "result": "NOT SURVEYED"})
            continue
        surveyed_n += 1
        det_finding = m.get("n_findings", 0) > 0
        meas = m["min_clearance"] if det_finding else (m["last_actual"] if m["last_actual"] is not None else ref)
        err = meas - true_c
        errors.append(abs(err))
        det_sev = m.get("max_severity") if det_finding else None
        problems = []
        if abs(err) > args.tolerance:
            problems.append(f"clearance off by {err:+.2f} m")
        if det_finding != exp_finding:
            problems.append("missed finding" if exp_finding else "false finding")
        else:
            agree_n += 1
        if exp_finding and det_finding:
            sev_total += 1
            sev_ok += int(det_sev == exp_sev)
            if det_sev != exp_sev:
                problems.append(f"severity {det_sev} vs expected {exp_sev}")
        rows.append({"segment": seg_id, "reference_m": ref, "true_clearance_m": true_c,
                     "measured_clearance_m": round(meas, 3), "error_m": round(err, 3),
                     "finding_expected": exp_finding, "finding_detected": det_finding,
                     "severity_expected": exp_sev, "severity_detected": det_sev,
                     "result": "OK" if not problems else "; ".join(problems)})

    n = len(truth)
    summary = {
        "segments_total": n, "segments_surveyed": surveyed_n,
        "coverage_pct": round(100 * surveyed_n / n) if n else 0,
        "mean_abs_clearance_error_m": round(sum(errors) / len(errors), 3) if errors else None,
        "max_abs_clearance_error_m": round(max(errors), 3) if errors else None,
        "finding_agreement": f"{agree_n}/{surveyed_n}",
        "severity_agreement": f"{sev_ok}/{sev_total}" if sev_total else "n/a",
        "tolerance_m": args.tolerance,
    }

    print("\nDIFFERENCE REPORT: measured vs verified layout\n")
    print(f"{'Seg':<4}{'Ref':>6}{'True':>7}{'Meas':>7}{'Err':>7}   {'Finding exp/det':<17}{'Sev exp/det':<13}Result")
    for r in rows:
        meas = f"{r['measured_clearance_m']:.2f}" if r["measured_clearance_m"] is not None else "-"
        err = f"{r['error_m']:+.2f}" if r["error_m"] is not None else "-"
        fd = {True: "yes", False: "no", None: "-"}
        print(f"{r['segment']:<4}{r['reference_m']:>6.2f}{r['true_clearance_m']:>7.2f}{meas:>7}{err:>7}   "
              f"{fd[r['finding_expected']] + ' / ' + fd[r['finding_detected']]:<17}"
              f"{str(r['severity_expected'] or '-') + ' / ' + str(r['severity_detected'] or '-'):<13}{r['result']}")
    print("\nSUMMARY")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    out = os.path.join(HERE, "difference_report.json")
    json.dump({"summary": summary, "segments": rows}, open(out, "w"), indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
"""Entry point of the sandbox container: `python -I /app/runner.py`.

Reads /in/job.json (analysis + parameters, already validated by the host) and /in/data.csv (the
extract: run_id, label, run_ts, then one column per feature `f<idx>`; an empty cell is a missing
value), writes /out/result.json and, for a plot, /out/plot.png. /in is mounted read-only. The
container has no network and a read-only root file system.
"""

# ruff: noqa: E402, I001
import csv
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I removes the script directory

import numpy as np
import scipy

import analyses

IN_DIR = os.environ.get("RFLENS_IN_DIR", "/in")
OUT_DIR = os.environ.get("RFLENS_OUT_DIR", "/out")
MAX_ROWS = 5000
MAX_RESULT_BYTES = 64 * 1024


def load_data(path: str) -> analyses.Data:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        if header[:3] != ["run_id", "label", "run_ts"]:
            raise analyses.AnalysisError("bad_extract", "unexpected extract header")
        rows = list(reader)
    if len(rows) > MAX_ROWS:
        raise analyses.AnalysisError(
            "too_many_rows", f"{len(rows)} rows; the maximum is {MAX_ROWS}"
        )
    names = header[3:]
    features = {}
    for k, name in enumerate(names):
        idx = int(name[1:])
        features[idx] = np.array(
            [float(r[3 + k]) if r[3 + k] != "" else np.nan for r in rows], dtype=float
        )
    return analyses.Data(
        run_ids=np.array([int(r[0]) for r in rows], dtype=np.int64),
        labels=np.array([r[1] for r in rows], dtype=object),
        run_ts=np.array([r[2] for r in rows], dtype=object),
        features=features,
    )


def write_result(payload: dict) -> None:
    text = json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
    if len(text.encode("utf-8")) > MAX_RESULT_BYTES:
        text = json.dumps(
            {
                "ok": False,
                "error": {
                    "code": "result_too_large",
                    "message": f"the result is over {MAX_RESULT_BYTES} bytes; ask for less",
                },
            }
        )
    with open(os.path.join(OUT_DIR, "result.json"), "w", encoding="utf-8") as f:
        f.write(text)


def main() -> int:
    with open(os.path.join(IN_DIR, "job.json"), encoding="utf-8") as f:
        job = json.load(f)
    name, params = job["analysis"], job["params"]
    try:
        data = load_data(os.path.join(IN_DIR, "data.csv"))
        extract_rows = data.n_rows
        data = analyses.apply_filters(data, params.get("filters") or {})
        if name == "plot":
            out_png = os.path.join(OUT_DIR, "plot.png")
            result = analyses.plot(data, params, out_png)
        elif name in analyses.ANALYSES:
            result = analyses.ANALYSES[name](data, params)
        else:
            raise analyses.AnalysisError("unknown_analysis", f"no analysis named {name!r}")
        write_result(
            {
                "ok": True,
                "analysis": name,
                "n_runs_in_extract": extract_rows,
                "n_runs_used": data.n_rows,
                "missing_policy": analyses.MISSING_POLICY,
                "versions": {
                    "python": sys.version.split()[0],
                    "numpy": np.__version__,
                    "scipy": scipy.__version__,
                },
                "result": result,
            }
        )
    except analyses.AnalysisError as e:
        write_result({"ok": False, "error": {"code": e.code, "message": e.message}})
    except Exception as e:  # report, never hide: the host logs it as an error
        traceback.print_exc()
        write_result(
            {"ok": False, "error": {"code": "internal", "message": f"{type(e).__name__}: {e}"}}
        )
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())

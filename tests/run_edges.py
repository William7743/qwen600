"""Run independent ASan and sampling probes; preserve every result, exit 1 on failures."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for case in ["tokenizer-less-than", "sampler-all", "sampler-distribution",
                 "sampler-distribution-default", "sampler-k-boundaries", "sampler-nucleus-boundary"]:
        command = [str(args.probe.resolve()), case]
        if case.startswith("tokenizer"):
            command.append(str(args.model.resolve()))
        run = subprocess.run(command, capture_output=True, text=True, timeout=60)
        log = run.stdout + run.stderr
        (args.output / (case + ".log")).write_text(log)
        result = {"case": case, "exit_code": run.returncode, "log": case + ".log"}
        if "AddressSanitizer: heap-buffer-overflow" in log:
            result["asan_error"] = "heap-buffer-overflow"
        if case.startswith("sampler-distribution") and run.stdout.strip():
            result["counts"] = json.loads(run.stdout)
        results.append(result)
        print(case, "PASS" if run.returncode == 0 else "FAIL", flush=True)
    (args.output / "edge-report.json").write_text(json.dumps(results, indent=2))
    return int(any(row["exit_code"] != 0 for row in results))


if __name__ == "__main__":
    raise SystemExit(main())

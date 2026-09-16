"""Portable regression entry point. Exit 0: scoped regressions pass, 1: tests fail, 2: setup fails.

The source model directory is read-only. Generated tokenizer files live in OUTPUT/model.
Numerical acceptance remains pending even when this command exits 0.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
import platform
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as data:
        for chunk in iter(lambda: data.read(8*1024*1024), b""): digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Local pinned HF model directory")
    parser.add_argument("--build-dir", type=Path, required=True, help="CMake build containing bin/correctness_probe")
    parser.add_argument("--asan-build-dir", type=Path, help="Separate CPU ASan build if needed")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-version-drift", action="store_true",
                        help="Explore different Python/Unicode library versions; marked non-reference")
    args = parser.parse_args()
    model = args.model.resolve(); build = args.build_dir.resolve()
    cpu = (args.asan_build_dir or args.build_dir).resolve()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    lock = json.loads((ROOT / "tests/reference.json").read_text())
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    summary = {"status": "SETUP_IN_PROGRESS", "numerical_acceptance": "PENDING",
               "scope": "at most 8192 tokens, defined regression corpus only", "steps": []}
    summary["python"] = sys.version
    summary["platform"] = platform.platform()

    def save():
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")

    def run(name, command, timeout):
        print(f"[{name}]", flush=True)
        with (output / (name+".log")).open("w") as log:
            result = subprocess.run(list(map(str, command)), cwd=ROOT, env=env,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        summary["steps"].append({"name": name, "exit_code": result.returncode, "log": name+".log"})
        save()
        print(f"[{name}] exit={result.returncode}; log={output / (name+'.log')}", flush=True)
        return result.returncode

    try:
        actual = {name: version(name) for name in lock["python_reference"]}
        drift = {name: {"expected": expected, "actual": actual[name]}
                 for name, expected in lock["python_reference"].items() if actual[name] != expected}
        summary["python_versions"] = actual
        summary["version_drift"] = drift
        if drift and not args.allow_version_drift:
            raise RuntimeError("Python versions differ from tests/reference.json; install the reference versions or explicitly use --allow-version-drift")
        import torch
        if not torch.cuda.is_available(): raise RuntimeError("The reference PyTorch runtime cannot access CUDA")
        summary["gpu"] = torch.cuda.get_device_name(0)
        summary["compute_capability"] = list(torch.cuda.get_device_capability(0))
        summary["torch_cuda"] = torch.version.cuda
        if torch.cuda.get_device_capability(0)[0] < 8:
            raise RuntimeError("These BF16 checks require an Ampere-or-newer CUDA GPU")
        probes = {"forward": build / "bin/correctness_probe",
                  "operators": build / "bin/operator_probe",
                  "tokenizer": cpu / "bin/tokenizer_probe", "edges": cpu / "bin/edge_probe"}
        for path in probes.values():
            if not path.is_file(): raise RuntimeError(f"Build the test target first: {path}")
        metadata = subprocess.run([str(probes["edges"]), "--build-info"], cwd=ROOT, env=env,
                                  capture_output=True, text=True, check=True, timeout=30)
        summary["native_libraries"] = json.loads(metadata.stdout)
        if not summary["native_libraries"]["address_sanitizer"]:
            raise RuntimeError("edge_probe lacks AddressSanitizer; rebuild with QWEN_ENABLE_ASAN=ON or supply --asan-build-dir")
        for name in ("icu", "pcre2"):
            found = summary["native_libraries"][name].split()[0]
            expected = lock["native_reference"][name]
            if found != expected:
                drift[name] = {"expected": expected, "actual": found}
        if drift and not args.allow_version_drift:
            raise RuntimeError("Native Unicode library versions differ; use the reference versions or --allow-version-drift")
        summary["version_profile"] = "NON_REFERENCE" if drift else "REFERENCE_LIBRARIES"
        summary["probe_sha256"] = {name: sha256(path) for name, path in probes.items()}
        cache = (build / "CMakeCache.txt").read_text()
        summary["cmake_configuration"] = [line for line in cache.splitlines() if line.startswith((
            "CMAKE_CUDA_COMPILER:", "CMAKE_CUDA_ARCHITECTURES:", "CMAKE_CUDA_HOST_COMPILER:",
            "CMAKE_CXX_COMPILER:", "CMAKE_BUILD_TYPE:"))]
        try:
            summary["driver"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                text=True, timeout=15).strip()
        except (OSError, subprocess.SubprocessError):
            summary["driver"] = "not available through nvidia-smi"
        summary["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        summary["git_status"] = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
        print("[preflight] checking model hashes", flush=True)
        hashes = {}
        for name, expected in lock["model"]["sha256"].items():
            hashes[name] = sha256(model / name)
            if hashes[name] != expected:
                raise RuntimeError(f"Model file mismatch: {name}; use revision {lock['model']['revision']}")
        summary["model_sha256"] = hashes
        stage = output / "model"; stage.mkdir(exist_ok=True)
        for name in lock["model"]["sha256"]:
            target = stage / name
            if target.exists() or target.is_symlink():
                if not target.is_symlink() or target.resolve() != (model/name).resolve():
                    raise RuntimeError(f"Output model staging conflicts with existing file: {target}")
            else:
                target.symlink_to(model/name)
        save()
        if run("export", [sys.executable, ROOT / "tools/export.py", stage], 180):
            raise RuntimeError("Tokenizer export failed; see export.log")
        checks = [
            ("tokenizer", "run_tokenizer.py", probes["tokenizer"], 180),
            ("edges", "run_edges.py", probes["edges"], 180),
            ("operators", "run_operators.py", probes["operators"], 1800),
            ("extended_model", "run_model_extended.py", probes["forward"], 1800),
            ("model", "run_correctness.py", probes["forward"], 1800),
        ]
        failed = False
        for name, script, probe, timeout in checks:
            failed |= run(name, [sys.executable, ROOT / "tests" / script, "--model", stage,
                                "--probe", probe, "--output", output / "results"], timeout) != 0
        summary["status"] = "FAIL" if failed else "REGRESSION_CHECKS_PASS_NUMERICAL_REVIEW_REQUIRED"
        save()
        print(summary["status"], flush=True)
        print("Full numerical correctness is NOT certified. Summary:", output / "summary.json", flush=True)
        return 1 if failed else 0
    except Exception as error:
        summary["status"] = "SETUP_OR_EXECUTION_ERROR"
        summary["error"] = str(error)
        save()
        print("ERROR:", error, file=sys.stderr)
        print("See", output / "summary.json", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

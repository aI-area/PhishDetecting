#!/usr/bin/env python3
"""Record whether auditable hardware energy telemetry is available to this account."""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import stat
import subprocess
from pathlib import Path


def run(command: list[str]) -> dict:
    result = subprocess.run(command, text=True, capture_output=True, timeout=15)
    return {"command": command, "returncode": result.returncode,
            "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("output", type=Path); args = parser.parse_args()
    counters = []
    for path in sorted(Path("/sys/class/powercap").glob("intel-rapl:*/energy_uj")) + sorted(Path("/sys/class/powercap").glob("intel-rapl:*:*/energy_uj")):
        record = {"path": str(path), "mode": stat.filemode(path.stat().st_mode), "readable": os.access(path, os.R_OK)}
        if record["readable"]:
            record["value_uj"] = int(path.read_text().strip())
        counters.append(record)
    perf = run([shutil.which("perf") or "perf", "stat", "-e", "power/energy-pkg/", "--", "sleep", "0.1"])
    ipmi_devices = [str(path) for path in (Path("/dev/ipmi0"), Path("/dev/ipmi/0"), Path("/dev/ipmidev/0")) if path.exists()]
    audit = {
        "status": "PASS" if counters and all(item["readable"] for item in counters) else "BLOCKED_NO_AUTHENTICATED_CPU_ENERGY_COUNTER",
        "host": platform.node(), "kernel": platform.release(), "account_uid": os.getuid(),
        "rapl_counters": counters, "perf_probe": perf, "ipmi_device_paths": ipmi_devices,
        "nvidia_smi_present": shutil.which("nvidia-smi") is not None,
        "gpu_scope_note": "The evaluated models are CPU-only; NVIDIA board-power telemetry cannot measure their CPU/DRAM energy.",
        "scientific_boundary": "Runtime multiplied by a catalog TDP or software-estimated power is not reported as measured energy.",
        "required_access": "Read access to package and DRAM energy_uj/max_energy_range_uj/name files under /sys/class/powercap/intel-rapl:* is required for randomized sequential measurements with idle-baseline subtraction.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__": main()

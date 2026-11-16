"""Shared protocol, provenance and arithmetic. No GPU or ML imports."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CORE = REPO / "core"
BUILD = HERE / "build"
FORMATS = ("bf16", "mxfp6w_wkv", "ssnf_wkv")
MODEL = "NousResearch/Meta-Llama-3.1-8B"
REVISION = "1f47e50cdbe801ad8a5174156ec3a0655108fb9f"
SCHEMA = 1


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    files = list((CORE / "kernel").rglob("*.cu"))
    files += list((CORE / "kernel").rglob("*.cuh"))
    files += list(CORE.rglob("*.py"))
    files += list(HERE.glob("*.py"))
    files += [CORE / "setup.py", CORE / "kernel/pybind.cpp",
              CORE / "ssnf_vllm/__init__.py", CORE / "ssnf_lib/__init__.py",
              CORE / "calib/H_w_llama31.pt", CORE / "calib/H_K_metallama318b.pt"]
    return {str(p.relative_to(REPO)): file_hash(p) for p in sorted(set(files))}


def fixed_environment(gpu_uuid):
    return {
        "CUDA_VISIBLE_DEVICES": gpu_uuid, "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "SSNF_PROFILE": "gs8",
        "SSNF_KV_ASYM": "3,16,4,32", "SSNF_TRITON": "1",
        "SSNF_KV_SPLIT_MULT": "32", "SSNF_APPEND_FUSED": "1",
        "SSNF_W_SCALE_OFFS": "0", "SSNF_KV_RADIUS": "0", "SSNF_RADIUS": "0",
        "SSNF_KV_WSHARE_RADIUS": "0",
        "SSNF_HFILE": str(CORE / "calib/H_w_llama31.pt"),
        "SSNF_KV_IMPORTANCE": str(CORE / "calib/H_K_metallama318b.pt"),
        "SSNF_SPLIT_CACHE": str(BUILD / ("split_cache_"+gpu_uuid+".json")),
        "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4",
    }


@dataclass(frozen=True)
class Spec:
    stage: str
    format: str
    gpu_uuid: str
    input_tokens: int
    output_tokens: int
    requests: int
    concurrency: int
    batches: int
    round: int = 1
    model: str = MODEL
    revision: str = REVISION
    seed: int = 20261025
    arrival_rate: float = 0.0
    interval_s: float = 0.1
    idle_s: float = 30.0
    warmup_passes: int = 2
    warmup_requests: int = 8
    measurement_timeout_s: float = 180.0
    total_timeout_s: float = 1200.0
    minimum_seconds: float = 30.0
    max_trace_disagreement: float = 0.10
    max_sample_gap_s: float = 1.0
    sample_power: bool = True
    require_full_concurrency: bool = False

    def validate(self):
        if self.stage not in ("smoke", "pilot", "measure") or self.format not in FORMATS:
            raise ValueError("Unsupported stage or format")
        if not self.gpu_uuid.startswith("GPU-") or not self.revision:
            raise ValueError("An explicit GPU UUID and model revision are required")
        if self.model != MODEL:
            raise ValueError("Calibration/model mapping is implemented for Llama-3.1-8B only")
        counts = (self.input_tokens, self.requests, self.concurrency, self.batches,
                  self.round, self.warmup_passes, self.warmup_requests)
        if min(counts) < 1 or self.output_tokens < 2:
            raise ValueError("Counts must be positive, with at least two output tokens")
        vals = (self.arrival_rate, self.interval_s, self.idle_s, self.measurement_timeout_s,
                self.total_timeout_s, self.minimum_seconds, self.max_trace_disagreement,
                self.max_sample_gap_s)
        if not all(math.isfinite(v) for v in vals):
            raise ValueError("All time/rate/tolerance values must be finite")
        if self.arrival_rate < 0 or self.idle_s < 0 or self.minimum_seconds < 0:
            raise ValueError("Negative arrival rate, idle interval or duration")
        if min(self.interval_s, self.measurement_timeout_s, self.total_timeout_s,
               self.max_trace_disagreement, self.max_sample_gap_s) <= 0:
            raise ValueError("Intervals, budgets and tolerances must be positive")
        if self.measurement_timeout_s >= self.total_timeout_s:
            raise ValueError("Total budget must also leave time for initialization/warmup")
        if self.minimum_seconds >= self.measurement_timeout_s:
            raise ValueError("Measurement budget must exceed the minimum duration")
        if self.stage != "smoke" and (not self.sample_power or self.minimum_seconds < 30):
            raise ValueError("Pilot/measure require power sampling and >=30 s minimum duration")
        if self.stage == "measure" and self.minimum_seconds < 60:
            raise ValueError("This protocol requires >=60 s for final measurements")


def identity(spec, provenance):
    return dict(format=spec.format, gpu_uuid=spec.gpu_uuid, model=spec.model,
                revision=spec.revision, python=sys.version, executable=sys.executable,
                env=fixed_environment(spec.gpu_uuid), sources=provenance)


def check_prerequisite(spec, provenance, directory):
    if spec.stage == "smoke":
        return None
    if directory is None:
        raise ValueError("Use --after with a passing smoke/pilot run directory")
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text())
    status = json.loads((directory / "status.json").read_text())
    config = json.loads((directory / "config.json").read_text())
    required = "smoke" if spec.stage == "pilot" else "pilot"
    if status.get("state") != "complete" or not summary.get("stage_passed"):
        raise ValueError("Prerequisite did not pass; inspect its validation/errors")
    if config["spec"]["stage"] != required:
        raise ValueError(f"{spec.stage} requires a {required} run")
    if config["identity"] != identity(spec, provenance):
        raise ValueError("Prerequisite uses a different GPU, format, environment, source or model")
    if config["spec"]["sample_power"] != spec.sample_power:
        raise ValueError("A sampler-overhead control cannot qualify a sampled experiment")
    if spec.stage == "measure":
        for key in ("input_tokens", "output_tokens", "concurrency", "arrival_rate", "seed",
                    "interval_s", "idle_s", "warmup_passes", "warmup_requests", "require_full_concurrency"):
            if config["spec"][key] != asdict(spec)[key]:
                raise ValueError(f"Pilot differs in {key}; validate this operating condition first")
    return dict(directory=str(directory.resolve()), summary_sha256=file_hash(directory / "summary.json"))


def arrival_offsets(count, rate, seed):
    if count < 1 or not math.isfinite(rate) or rate < 0:
        raise ValueError("Invalid arrival trace parameters")
    rng = random.Random(seed)
    result = [0.0]
    for _ in range(1, count):
        result.append(result[-1] + rng.expovariate(rate) if rate else 0.0)
    return result


def percentile(values, q):
    vals = sorted(values)
    if not vals or not 0 <= q <= 100 or not all(math.isfinite(v) for v in vals):
        raise ValueError("Invalid percentile input")
    idx = (len(vals) - 1) * q / 100
    lo = int(idx)
    return vals[lo] + (vals[min(lo + 1, len(vals) - 1)] - vals[lo]) * (idx - lo)


def request_latencies(record, expected_tokens):
    if not record["finished"] or record["output_tokens"] != expected_tokens:
        raise ValueError("Missing request or incomplete output")
    offered, enqueued, first, last = (record[k] for k in ("offered_t", "enqueued_t", "first_t", "last_t"))
    if not all(v is not None and math.isfinite(v) for v in (offered, enqueued, first, last)):
        raise ValueError("Missing or invalid request timestamps")
    if not offered <= enqueued <= first <= last:
        raise ValueError("Request timestamps are out of order")
    return dict(e2e_s=last-offered, ttft_s=first-offered,
                tpot_s=(last-first)/(expected_tokens-1), enqueue_delay_s=enqueued-offered)


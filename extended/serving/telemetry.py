"""NVML board telemetry, with timestamped counters and bounded sampler lifecycle."""
from __future__ import annotations

import csv
import math
import os
from pathlib import Path
import threading
import time


def integrate(samples, start, end):
    """Integrate W over seconds; interpolate both boundaries, never extrapolate."""
    points = [(s["t"], s["power_w"]) for s in samples]
    if not all(math.isfinite(t) and math.isfinite(p) and p >= 0 for t,p in points):
        raise ValueError("Invalid timestamp or power sample")
    if end <= start or len(points) < 2 or points[0][0] > start or points[-1][0] < end:
        raise ValueError("Power trace does not bracket the measurement window")
    energy = 0.0
    for (ta, pa), (tb, pb) in zip(points, points[1:]):
        if tb <= ta:
            raise ValueError("Non-monotonic power timestamps")
        left, right = max(ta, start), min(tb, end)
        if right > left:
            pleft = pa + (pb-pa)*(left-ta)/(tb-ta)
            pright = pa + (pb-pa)*(right-ta)/(tb-ta)
            energy += (pleft+pright)*(right-left)/2
    return energy


class Telemetry:
    def __init__(self, uuid, interval_s=0.1, nvml=None):
        if nvml is None:
            import pynvml as nvml
        self.nv = nvml
        nvml.nvmlInit()
        self.handle = nvml.nvmlDeviceGetHandleByUUID(uuid)
        self.uuid = nvml.nvmlDeviceGetUUID(self.handle)
        self.interval_s = interval_s
        self.initial_limit_w = nvml.nvmlDeviceGetPowerManagementLimit(self.handle)/1000
        self.samples, self.errors = [], []
        self.foreign_pids = set()
        self.phase = "initialization"
        self.event = threading.Event()
        self.thread = None

    def pids(self):
        n, h = self.nv, self.handle
        result = {p.pid for p in n.nvmlDeviceGetComputeRunningProcesses(h)}
        try:
            result.update(p.pid for p in n.nvmlDeviceGetGraphicsRunningProcesses(h))
        except n.NVMLError_NotSupported:
            pass
        return sorted(result)

    def assert_exclusive(self):
        others = set(self.pids()) - {os.getpid()}
        if others:
            raise RuntimeError(f"Other compute/graphics processes on target GPU: {sorted(others)}")

    def metadata(self):
        n, h = self.nv, self.handle
        return dict(uuid=self.uuid, name=n.nvmlDeviceGetName(h), driver=n.nvmlSystemGetDriverVersion(),
                    power_limit_w=n.nvmlDeviceGetPowerManagementLimit(h)/1000,
                    memory_bytes=n.nvmlDeviceGetMemoryInfo(h).total,
                    power_field="nvmlDeviceGetPowerUsage (mW -> W); trailing 1 s average on Blackwell",
                    counter_field="nvmlDeviceGetTotalEnergyConsumption (mJ)",
                    interval_s=self.interval_s, process_pid=os.getpid())

    def counter(self):
        start = time.perf_counter()
        try:
            value = self.nv.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        except self.nv.NVMLError_NotSupported:
            value = None
        end = time.perf_counter()
        return dict(mj=value, before=start, after=end, midpoint=(start+end)/2,
                    query_seconds=end-start)

    def sample(self, auxiliary=True):
        n, h = self.nv, self.handle
        before = time.perf_counter()
        power_w = n.nvmlDeviceGetPowerUsage(h)/1000
        if not math.isfinite(power_w) or power_w < 0:
            raise ValueError("Invalid NVML power value")
        after = time.perf_counter()
        row = dict(t=(before+after)/2, power_w=power_w, phase=self.phase,
                   query_seconds=after-before)
        if auxiliary:
            row.update(temperature_c=n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU),
                       sm_clock_mhz=n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM),
                       memory_clock_mhz=n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_MEM),
                       memory_used_bytes=n.nvmlDeviceGetMemoryInfo(h).used,
                       power_limit_w=n.nvmlDeviceGetPowerManagementLimit(h)/1000)
            if row["power_limit_w"] != self.initial_limit_w:
                self.errors.append("GPU power limit changed during the run")
            try:
                row["throttle_reasons"] = n.nvmlDeviceGetCurrentClocksThrottleReasons(h)
            except n.NVMLError_NotSupported:
                row["throttle_reasons"] = None
            pids = self.pids()
            row["pids"] = ",".join(map(str, pids))
            self.foreign_pids.update(set(pids)-{os.getpid()})
        self.samples.append(row)

    def start(self):
        self.sample()
        def poll():
            tick = 0
            while not self.event.wait(self.interval_s):
                try:
                    self.sample(auxiliary=(tick % max(1, round(1/self.interval_s)) == 0))
                except Exception as exc:
                    self.errors.append(repr(exc))
                tick += 1
        self.thread = threading.Thread(target=poll, name="energy-nvml", daemon=True)
        self.thread.start()

    def close(self, path):
        self.event.set()
        if self.thread:
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                self.errors.append("NVML sampler did not stop within 5 seconds")
                self.save(path)
                return
        try:
            self.sample()
        except Exception as exc:
            self.errors.append(repr(exc))
        self.save(path)

    def save(self, path):
        rows = list(self.samples)
        if rows:
            with Path(path).open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=sorted(set().union(*(s.keys() for s in rows))))
                writer.writeheader()
                writer.writerows(rows)


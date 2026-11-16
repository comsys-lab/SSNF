"""GPU activity guard; no power or energy measurements."""
import atexit
import json
import os
from pathlib import Path
import threading
import time
import pynvml


class Telemetry:
    def __init__(self, uuid):
        self.nv = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
        self.uuid = uuid
        self.pid = os.getpid()
        self.allowed = {p.pid: p.usedGpuMemory for p in self.nv.nvmlDeviceGetComputeRunningProcesses(self.handle) if p.pid != self.pid}
        self.initial_allowed = dict(self.allowed)
        self.started = time.time()
        self.cursor = int(self.started * 1e6)
        self.events = []
        self.own_active_samples = 0
        self.error = None
        self.output = Path(os.environ['D1_GUARD_LOG'])
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.check()
        self.thread.start()
        atexit.register(self.flush)

    def check(self):
        live = {p.pid: p.usedGpuMemory for p in self.nv.nvmlDeviceGetComputeRunningProcesses(self.handle) if p.pid != self.pid}
        if set(live) - set(self.allowed):
            raise RuntimeError('New background GPU process: ' + str(live))
        for pid, memory in live.items():
            if abs(memory - self.allowed[pid]) > 16 * 1024 * 1024:
                raise RuntimeError('Background GPU allocation changed: ' + str(pid))
        # A PID may finish, but it may not allocate a new CUDA context.
        try:
            samples = self.nv.nvmlDeviceGetProcessUtilization(self.handle, self.cursor)
        except self.nv.NVMLError_NotFound:
            samples = []
        for s in samples:
            self.cursor = max(self.cursor, s.timeStamp)
            if s.pid == self.pid and (s.smUtil or s.memUtil):
                self.own_active_samples += 1
            if s.pid != self.pid and (s.smUtil or s.memUtil or s.encUtil or s.decUtil):
                event = dict(pid=s.pid, sm=s.smUtil, mem=s.memUtil, timestamp=s.timeStamp)
                self.events.append(event)
                raise RuntimeError('Background GPU computation detected: ' + str(event))

    def _watch(self):
        while self.error is None:
            try:
                self.check()
            except Exception as exc:
                self.error = repr(exc)
            self.flush()
            time.sleep(.5)

    def flush(self):
        record = dict(gpu_uuid=self.uuid, worker_pid=self.pid, initial_background_memory=self.initial_allowed,
                      own_active_samples=self.own_active_samples, interference_events=self.events,
                      error=self.error, elapsed_s=time.time()-self.started,
                      scope='Idle existing CUDA contexts retained; reject new contexts, changed background allocations, or any sampled background GPU activity; not an empty-GPU experiment')
        tmp = self.output.with_suffix('.tmp')
        tmp.write_text(json.dumps(record, indent=2)+'\n')
        tmp.replace(self.output)

    def assert_exclusive(self):
        if self.error:
            raise RuntimeError(self.error)
        self.check()

    def metadata(self):
        n = self.nv
        return dict(uuid=self.uuid, name=n.nvmlDeviceGetName(self.handle), driver=n.nvmlSystemGetDriverVersion(),
                    memory_bytes=n.nvmlDeviceGetMemoryInfo(self.handle).total, process_pid=self.pid,
                    initial_idle_background_memory=self.initial_allowed,
                    isolation='No concurrent background GPU work; stable pre-existing CUDA contexts remain resident')

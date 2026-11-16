"""Persistent XRT controller and shared-card occupancy check."""
import os,json,select,subprocess,time
from pathlib import Path

class Session:
    def __init__(self,args,output):
        if os.environ.get('XCL_EMULATION_MODE'):raise RuntimeError('Actual-card measurements require XCL_EMULATION_MODE to be unset')
        self.stderr=(output/'host.stderr.log').open('w')
        self.protocol=(output/'controller.jsonl').open('w')
        self.buffer=b''
        self.process=subprocess.Popen([str(args.host),args.device,str(args.image),str(args.vectors)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,bufsize=0)
        try:
            self.ready=self.receive(30)
            if self.ready.get('action')!='ready':raise RuntimeError('Controller did not report ready')
        except BaseException:
            if self.process.poll() is None:self.process.terminate();self.process.wait(timeout=10)
            self.protocol.close();self.stderr.close();raise
    def receive(self,timeout=30):
        end_time=time.monotonic()+timeout
        while True:
            if b'\n' in self.buffer:
                line,self.buffer=self.buffer.split(b'\n',1)
                text=line.decode('utf-8',errors='replace')
                self.protocol.write(text+'\n');self.protocol.flush()
                try:return json.loads(text)
                except json.JSONDecodeError:continue
            remaining=end_time-time.monotonic()
            if remaining<=0 or not select.select([self.process.stdout],[],[],remaining)[0]:
                raise TimeoutError('Controller response timeout')
            chunk=os.read(self.process.stdout.fileno(),65536)
            if not chunk:raise RuntimeError('Controller exited; inspect host.stderr.log')
            self.buffer+=chunk
    def command(self,line,timeout=30):
        self.process.stdin.write((line+'\n').encode());self.process.stdin.flush()
        return self.receive(timeout)
    def close(self):
        try:
            if self.process.poll() is None:
                self.command('stop');self.command('quit');self.process.wait(timeout=10)
        finally:
            if self.process.poll() is None:
                self.process.terminate();self.process.wait(timeout=10)
            self.protocol.close();self.stderr.close()


def assert_no_other_device_users(device):
    drm=Path('/sys/bus/pci/devices')/device/'drm'
    nodes=[Path('/dev/dri')/p.name for p in drm.glob('renderD*')]
    if not nodes:raise RuntimeError('Cannot resolve device render node; inspect server occupancy before programming')
    result=subprocess.run(['fuser',*[str(p) for p in nodes]],text=True,capture_output=True)
    if result.returncode not in (0,1):raise RuntimeError('Device process query failed: '+result.stderr)
    if result.stdout.strip():raise RuntimeError('Device already has users: '+result.stdout)
    if result.returncode==0:raise RuntimeError('Device is in use')
    if result.stderr.strip():raise RuntimeError('Device process visibility is incomplete: '+result.stderr)



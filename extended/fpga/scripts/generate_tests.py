import sys,json,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from source_reference import seeded,records,advance,pack_states
registered='--registered' in sys.argv[2:]
def observed_record(state,mode):return records(state if registered else advance(state))[0][mode]
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
src=[];dots=[];controllers=[]
for i in range(1016):
    state=seeded(i)
    if i==0:state=[0]*6
    if i==1:state=[(1<<64)-1]*6
    rec,expected,integer=records(state)
    for mode in range(3):
        src.append(f'{mode} {pack_states(state):096x} {observed_record(state,mode):0256x} {pack_states(advance(state)):096x}\n')
        dots.append(f'{mode} {rec[mode]:0256x} {expected[mode]:08x} {integer[mode]&0xffffffff:08x}\n')
for mode in range(3):
    for batches in [1,7,128]:
        for compute in [1,0]:
            for replica in range(16):
                start=seeded(11+batches,replica);state=start
                for _ in range(batches-1):state=advance(state)
                rec,expected,_=records(state)
                controllers.append(f'{mode} {compute} {batches} {replica} {pack_states(start):096x} {pack_states(advance(state)):096x} {observed_record(state,mode):0256x} {expected[mode]:08x}\n')
(out/'source.txt').write_text(''.join(src));(out/'vectors.txt').write_text(''.join(dots));(out/'controller.txt').write_text(''.join(controllers))
(out/'manifest.json').write_text(json.dumps({'source_cases':len(src),'dot_cases':len(dots),'controller_cases':len(controllers),'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}},indent=2))
smoke=[]
for mode in range(3):
 for i in range(128):
  state=seeded(i)
  if i==0:state=[0]*6
  if i==1:state=[(1<<64)-1]*6
  rec,expected,_=records(state)
  smoke.append(f'{mode} 1 1 0 {pack_states(state):096x} {pack_states(advance(state)):096x} {observed_record(state,mode):0256x} {expected[mode]:08x}\n')
(out/'smoke.txt').write_text(''.join(smoke))
(out/'power_seeds.bin').write_bytes(b''.join(pack_states(seeded(10000+seed,replica)).to_bytes(48,'little') for seed in range(5) for replica in range(16)))
(out/'manifest.json').write_text(json.dumps({'source_cases':len(src),'dot_cases':len(dots),'controller_cases':len(controllers),'power_seed_sets':5,'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.name!='manifest.json'}},indent=2))

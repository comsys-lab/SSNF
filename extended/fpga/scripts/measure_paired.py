#!/usr/bin/env python3
"""Actual-card OFF/ON/OFF, with smoke -> pilot -> fixed-seed measurements."""
import argparse,fcntl,hashlib,json,signal,subprocess,sys,time
from pathlib import Path
from device_session import Session,assert_no_other_device_users
from power_reader import sample
from paired_analysis import analyze_window,triplet_summary
from analyze import board_power

def dump(p,x,**kwargs):p.write_text(json.dumps(x,indent=2,**kwargs))
def window(session,args,root,index,on,seed,mhz):
    run=root/f'{index:02d}_{"on" if on else "off"}';run.mkdir()
    dump(run/'config.json',{'format_id':args.format_id,'replicas':args.replicas,'compute':on,'seed_set':seed,'interval':1,
         'validated_clock_mhz':mhz,'device':args.device,'seconds':args.seconds,'warmup_s':args.warmup})
    initial=session.command(f'run {args.format_id} {int(on)} {args.replicas} {seed} 1 0');dump(run/'initial.json',initial)
    time.sleep(args.warmup)
    with (run/'window.jsonl').open('x') as log:
        def record(x):log.write(json.dumps(x)+'\n');log.flush()
        n=0
        def telemetry():
            nonlocal n
            row=sample(args.device,run/'sensor_raw',n);n+=1;board_power(row);return row
        record(telemetry());begin=session.command('snapshot');begin['marker']='counter_start';record(begin)
        end_time=time.monotonic()+args.seconds
        while time.monotonic()<end_time:
            record(telemetry());remaining=end_time-time.monotonic()
            if remaining>0:time.sleep(min(.5,remaining))
        end=session.command('snapshot');end['marker']='counter_end';record(end);record(telemetry())
    drain=session.command('stop');dump(run/'drain.json',drain)
    result=analyze_window(run);print(json.dumps({'window':str(run),**result}),flush=True);return result

def validate_files(directory):
    manifest=json.loads((directory/'manifest.json').read_text())
    for name,digest in manifest['sha256'].items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=digest:raise ValueError('Vector hash mismatch: '+name)
    return hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest()

def main():
    def terminate(*_):raise KeyboardInterrupt('Interrupted; stop own kernel')
    signal.signal(signal.SIGTERM,terminate)
    p=argparse.ArgumentParser()
    for key in ['host','image','vectors','output','build-validation','shared-lock']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',default='0000:01:00.1');p.add_argument('--format-id',type=int,choices=range(3),required=True)
    p.add_argument('--stage',choices=['smoke','pilot','measure'],required=True);p.add_argument('--pilot-validation',type=Path)
    p.add_argument('--replicas',type=int,default=1);p.add_argument('--seed-set',type=int,choices=range(5),default=0)
    p.add_argument('--repetitions',type=int,default=1);p.add_argument('--seconds',type=float,default=10);p.add_argument('--warmup',type=float,default=5)
    args=p.parse_args()
    args.host=args.host.resolve();args.image=args.image.resolve();args.vectors=args.vectors.resolve()
    if args.output.exists():p.error('Refusing to overwrite existing stage')
    if args.warmup<5 or args.seconds<5 or args.repetitions not in [1,3]:p.error('Use >=5s settle/window and 1 or 3 repetitions')
    gate=json.loads(args.build_validation.read_text());ih=hashlib.sha256(args.image.read_bytes()).hexdigest();vh=validate_files(args.vectors)
    if not gate.get('timing_passed') or gate['image_sha256']!=ih or gate['format_id']!=args.format_id:p.error('Image/timing/format gate mismatch')
    if not 1<=args.replicas<=gate['replicas']:p.error('Replica count invalid')
    if args.stage=='measure':
        if not args.pilot_validation:p.error('Pilot validation required')
        pilot=json.loads(args.pilot_validation.read_text())
        if not pilot.get('ready_for_main') or pilot.get('replicas')!=args.replicas or pilot.get('image_sha256')!=ih or pilot.get('vectors_manifest_sha256')!=vh:p.error('Pilot does not validate this experiment')
        if args.seconds<45 or args.repetitions!=1:p.error('Main requires one >=45s triplet per predeclared seed')
    with args.shared_lock.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert_no_other_device_users(args.device)
        args.output.mkdir(parents=True)
        build_manifest=args.build_validation.parent/'source_manifest.json'
        if not build_manifest.is_file():raise ValueError('Missing image build source provenance')
        (args.output/'image_build_source_manifest.json').write_bytes(build_manifest.read_bytes())
        dump(args.output/'request.json',{**vars(args),'image_sha256':ih,'host_binary_sha256':hashlib.sha256(args.host.read_bytes()).hexdigest(),'vectors_manifest_sha256':vh,'image_build_source_manifest_sha256':hashlib.sha256(build_manifest.read_bytes()).hexdigest(),'reservation_note':'User authorized U250 experiment; common lock and fuser check; normal XRT load, no force/reset'},default=str)
        subprocess.run([sys.executable,str(Path(__file__).with_name('write_manifest.py')),str(Path(__file__).resolve().parents[1]),str(args.output/'source_manifest.json')],check=True)
        session=Session(args,args.output)
        try:
            if session.ready['format_id']!=args.format_id or session.ready['max_replicas']!=gate['replicas']:raise ValueError('Actual programmed format/replica mismatch')
            smoke=session.command('smoke',120)
            if not smoke.get('passed') or smoke['outputs_checked']!=128:raise ValueError('Actual hardware smoke failed')
            dump(args.output/'smoke.json',smoke)
            check=session.command(f'check {args.replicas}',120)
            if not check.get('passed') or check['outputs_checked']!=3*args.replicas or check['off_holds_checked']!=3*args.replicas:raise ValueError('Hardware ON/OFF finite trace check failed')
            dump(args.output/'replica_check.json',check)
            results=[]
            if args.stage!='smoke':
                for rep in range(args.repetitions):
                    seed=rep if args.repetitions==3 else args.seed_set
                    root=args.output/f'triplet_seed{seed}';root.mkdir()
                    windows=[window(session,args,root,i,on,seed,gate['runtime_clock_mhz']) for i,on in enumerate([False,True,False])]
                    result=triplet_summary(windows);dump(root/'summary.json',result);results.append(result)
                    print(json.dumps({'triplet':str(root),**result}),flush=True)
            ready=False
            if args.stage=='pilot' and args.repetitions==3:
                import statistics
                values=[x['delta_w'] for x in results];sd=statistics.stdev(values);mean=statistics.mean(values);drift=max(x['off_drift_w'] for x in results)
                ready=mean>max(2*sd,drift,.02)
                dump(args.output/'resolution.json',{'mean_delta_w':mean,'sample_sd_w':sd,'max_off_drift_w':drift,'threshold_w':max(2*sd,drift,.02),'signal_resolved':ready,'rule':'positive activation exceeds max(2 repeat SD, max OFF drift, 0.02 W); no format advantage criterion'})
            dump(args.output/'stage_validation.json',{'stage':args.stage,'image_sha256':ih,'vectors_manifest_sha256':vh,'format_id':args.format_id,
                 'replicas':args.replicas,'interval':1,'actual_board_smoke_passed':True,'triplets':len(results),'all_windows_valid':True,'ready_for_main':ready,'seed_set':args.seed_set})
        finally:session.close()

if __name__=='__main__':main()

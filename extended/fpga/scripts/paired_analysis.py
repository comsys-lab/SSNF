"""Counter-aligned electrical telemetry, with independent end-state verification."""
import json,math,sys
from functools import lru_cache
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from source_reference import advance,seeded,records,pack_states
from analyze import midpoint,board_power,integrate

def apply(columns,x):
    result=0
    while x:
        low=x&-x;result^=columns[low.bit_length()-1];x^=low
    return result

@lru_cache(maxsize=64)
def power(k):
    if not k:return tuple(advance([1<<b])[0] for b in range(64))
    a=power(k-1)
    return tuple(apply(a,x) for x in a)

def jump(states,n):
    if n<0:raise ValueError('Negative source step count')
    out=list(states);k=0
    while n:
        if n&1:out=[apply(power(k),x) for x in out]
        n>>=1;k+=1
    return out

def words(x,n):return [(x>>(32*i))&0xffffffff for i in range(n)]

def audit_drain(config,drain,initial):
    r=config['replicas'];n=drain['batches'];on=config['compute'];mode=config['format_id']
    if drain['status']!=2 or drain['accepted']!=drain['events'] or drain['accepted']!=n*r or not n:
        raise ValueError('Drain/count/status mismatch')
    if drain['completed']!=(drain['events'] if on else 0) or drain['baseline_events']!=(0 if on else drain['events']):
        raise ValueError('OFF events mislabeled as completed work')
    for i in range(r):
        seed=seeded(10000+config['seed_set'],i)
        last=jump(seed,n-1);final=advance(last)
        if words(pack_states(final),12)!=drain['source_states'][i]:raise ValueError('Long-run source state mismatch')
        rec,result,_=records(last)
        if words(rec[mode],32)!=drain['input_records'][i]:raise ValueError('Long-run encoded input mismatch')
        expected=result[mode] if on else initial['initial_results'][i]
        if drain['results'][i]!=expected:raise ValueError('Long-run result/OFF hold mismatch')
    return {'independent_jump_ahead_verified':True,'replicas_verified':r,'source_batches_verified':n,'last_result_or_off_hold_verified':True}

def analyze_window(run):
    rows=[json.loads(x) for x in (run/'window.jsonl').read_text().splitlines()]
    a=[x for x in rows if x.get('marker')=='counter_start'];b=[x for x in rows if x.get('marker')=='counter_end']
    if len(a)!=1 or len(b)!=1:raise ValueError('Counter markers missing/duplicated')
    a,b=a[0],b[0];c=json.loads((run/'config.json').read_text());on=c['compute'];r=c['replicas'];interval=c['interval']
    if any(x['status']!=1 or x['compute']!=on for x in [a,b]):raise ValueError('Wrong running configuration')
    if any(x['completed']!=(x['events'] if on else 0) or x['baseline_events']!=(0 if on else x['events']) for x in [a,b]):raise ValueError('Work counters mislabeled')
    cycles=b['cycles']-a['cycles'];events=b['events']-a['events'];t0,t1=midpoint(a),midpoint(b);elapsed=t1-t0
    if cycles<=0 or events<=0 or elapsed<=0:raise ValueError('Non-increasing work/time')
    if abs(events-cycles*r/interval)>2*r:raise ValueError('Event rate does not match actual cycle count')
    hz=cycles/elapsed
    if not .98*c['validated_clock_mhz']*1e6<=hz<=1.02*c['validated_clock_mhz']*1e6:raise ValueError('Clock/counter timestamp mismatch')
    samples=[x for x in rows if x.get('action')=='telemetry'];points=[(midpoint(x),board_power(x)) for x in samples]
    if len(points)<6:raise ValueError('Too few actual telemetry queries')
    if any(x['device']!=c['device'] for x in samples):raise ValueError('Device sensor mismatch')
    energy=integrate(points,t0,t1)
    temps=[float(t['temp_C']) for row in samples for t in row.get('thermal',[]) if t.get('location_id')=='fpga0' and str(t.get('is_present')).lower()=='true']
    if not temps or any(not math.isfinite(t) or t<0 or t>100 for t in temps):raise ValueError('Missing or invalid FPGA temperature')
    if max(y[0]-x[0] for x,y in zip(points,points[1:]))>2.5:raise ValueError('Telemetry gap exceeds 2.5 seconds')
    audit=audit_drain(c,json.loads((run/'drain.json').read_text()),json.loads((run/'initial.json').read_text()))
    out={'status':'valid_actual_board_window','format_id':c['format_id'],'compute':on,'replicas':r,'seed_set':c['seed_set'],
         'elapsed_s':elapsed,'active_cycles':cycles,'events':events,'completed_dots':events if on else 0,'baseline_events':0 if on else events,
         'observed_clock_hz':hz,'board_power_w':energy/elapsed,'board_energy_j':energy,'dot_per_s':events/elapsed if on else None,
         'board_nj_per_dot':energy/events*1e9 if on else None,'telemetry_queries':len(points),'distinct_power_values':len(set(p for _,p in points)),
         'max_counter_query_s':max((x['after_ns']-x['before_ns'])/1e9 for x in [a,b]),'fpga_temperature_min_C':min(temps),'fpga_temperature_max_C':max(temps),'fpga_temperature_first_C':temps[0],'fpga_temperature_last_C':temps[-1],**audit}
    (run/'summary.json').write_text(json.dumps(out,indent=2));return out

def triplet_summary(windows):
    before,on,after=windows
    if [x['compute'] for x in windows]!=[False,True,False]:raise ValueError('Expected OFF/ON/OFF')
    keys=['format_id','replicas','seed_set']
    if any(any(x[k]!=on[k] for k in keys) for x in windows):raise ValueError('Mismatched paired conditions')
    baseline=(before['board_power_w']+after['board_power_w'])/2
    delta=on['board_power_w']-baseline;drift=abs(after['board_power_w']-before['board_power_w'])
    return {**{k:on[k] for k in keys},'scope':'same-image arithmetic activation above running input source, including isolation and monitor residuals; common clock tree and static power stay in OFF baseline',
            'off_before_w':before['board_power_w'],'on_w':on['board_power_w'],'off_after_w':after['board_power_w'],'off_mean_w':baseline,
            'delta_w':delta,'off_drift_w':drift,'dot_per_s':on['dot_per_s'],'activation_nj_per_dot':delta/on['dot_per_s']*1e9,
            'board_nj_per_dot':on['board_nj_per_dot'],'fpga_temperature_min_C':min(x.get('fpga_temperature_min_C',float('nan')) for x in windows),'fpga_temperature_max_C':max(x.get('fpga_temperature_max_C',float('nan')) for x in windows),'signal_exceeds_off_drift':abs(delta)>drift,'windows_valid':True}

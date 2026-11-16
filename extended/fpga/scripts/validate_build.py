#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('build',type=Path)
p.add_argument('--replicas',type=int,required=True)
p.add_argument('--format-id',type=int,choices=[0,1,2],required=True)
args=p.parse_args()
review=args.build/'review'
route=json.loads((review/'route_check.json').read_text())
topology=json.loads((review/'clock_topology.json').read_text())
clocks=[r for r in topology['clock_freq_topology']['m_clock_freq'] if r['m_type']=='DATA']
if len(clocks)!=1:
    raise SystemExit('Expected one xclbin DATA clock')
period_difference=1000/float(clocks[0]['m_freq_Mhz'])-1000/route['reviewed_clock_mhz']
if not route.get('clock_constraints_preserved') or not -1e-9<=period_difference<=0.002:
    raise SystemExit('Timing clock does not match xclbin DATA clock')
text=(review/'timing_routed.rpt').read_text()
if 'VIOLATED' in (review/'bus_skew_routed.rpt').read_text():
    raise SystemExit('Post-route bus skew constraints are violated')
if not (route['setup_slack_ns']>=0 and route['hold_slack_ns']>=0 and route['drc_error_count']==0):
    raise SystemExit('Post-route setup/hold/DRC checks failed')
if route.get('drc_critical_warning_count',-1)!=0 or route.get('hdpr5_violation_count',-1)!=0:
    raise SystemExit('Critical or HDPR-5 DRC violations require review before board load')
if 'All user specified timing constraints are met.' not in text:
    raise SystemExit('Timing summary does not confirm all constraints are met')
for check in ['no_clock','unconstrained_internal_endpoints']:
    match=re.search(r'checking\s+'+check+r'\s*\((\d+)\)',text)
    if not match or int(match.group(1))!=0:
        raise SystemExit('Timing coverage requires review: '+check)
inventory=(review/'core_inventory.txt').read_text()
for name in ['ssnf_dot32','mxint8_dot32','bf16_dot32','input_source']:
    expected=args.replicas if name=='input_source' or name==['ssnf_dot32','mxint8_dot32','bf16_dot32'][args.format_id] else 0
    if f'{name} {expected}\n' not in inventory:
        raise SystemExit('Unexpected implementation core inventory: '+name)
data={**route,'timing_passed':True,'replicas':args.replicas,'format_id':args.format_id,
      'image_sha256':hashlib.sha256((args.build/'ssnf_bench.xclbin').read_bytes()).hexdigest(),
      'scope':'post-route timing, DRC and retained core inventory; not board correctness'}
(args.build/'build_validation.json').write_text(json.dumps(data,indent=2))
print(json.dumps(data,indent=2))

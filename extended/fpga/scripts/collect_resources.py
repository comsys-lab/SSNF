#!/usr/bin/env python3
"""Keep directly reported hierarchy values; never subtract packed LUT totals."""
import argparse,json,re,statistics
from pathlib import Path

def parse(path):
    rows=[];header=None
    mapping={'LUT':'Total LUTs','logic_LUT':'Logic LUTs','LUTRAM':'LUTRAMs','SRL':'SRLs','FF':'FFs','RAMB36':'RAMB36','RAMB18':'RAMB18','URAM':'URAM','DSP':'DSP Blocks'}
    for line in path.read_text().splitlines():
        cells=[s.strip() for s in line.split('|')][1:-1]
        if cells and cells[0]=='Instance' and 'Total LUTs' in cells:
            header={name:i for i,name in enumerate(cells)};continue
        if not header or len(cells)<=max(header.values()):continue
        try:values={key:int(cells[header[name]].replace(',','')) for key,name in mapping.items()}
        except (ValueError,KeyError):continue
        rows.append({'instance':cells[header['Instance']],'module':cells[header['Module']],**values})
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('report',type=Path);p.add_argument('output',type=Path);p.add_argument('--format-id',type=int,required=True);p.add_argument('--replicas',type=int,default=16);a=p.parse_args()
    rows=parse(a.report);selected=['ssnf_dot32','mxint8_dot32','bf16_dot32'][a.format_id]
    groups={}
    for name in ['input_source',selected]:
        found=[r for r in rows if re.search(r'(?:^|_)'+re.escape(name)+r'(?:_.*)?$',r['module']) and not r['instance'].startswith('(')]
        if len(found)!=a.replicas:raise ValueError(f'Expected {a.replicas} {name}, found {len(found)}')
        if any(r['RAMB36'] or r['RAMB18'] or r['URAM'] for r in found):raise ValueError('Unexpected kernel memory allocation')
        groups[name]={'instances':found,'per_instance_range':{key:[min(r[key] for r in found),max(r[key] for r in found)] for key in ['LUT','FF','DSP','RAMB36','RAMB18']}}
    data={'report':str(a.report),'format_id':a.format_id,'replicas':a.replicas,'groups':groups,
          'warning':'Hierarchy resource attribution can move across boundaries; values are not additive and are not transistor-level area.'}
    a.output.write_text(json.dumps(data,indent=2));print(json.dumps({k:v['per_instance_range'] for k,v in groups.items()}))
if __name__=='__main__':main()

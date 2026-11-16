#!/usr/bin/env python3
import argparse,json,math,statistics
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();groups={}
    plan=json.loads((a.root/'measurement_plan.json').read_text())
    if plan['source_seed_sets']!=list(range(5)) or plan['main_repetitions_per_format']!=5:raise ValueError('Unexpected predeclared seed plan')
    for f,name in enumerate(['ssnf','mxint8','bf16']):
        rows=[];ih=set();vh=set()
        for seed in range(5):
            d=a.root/f'{name}_main_seed{seed}';gate=json.loads((d/'stage_validation.json').read_text())
            if gate['stage']!='measure' or not gate['all_windows_valid'] or gate['format_id']!=f or gate['seed_set']!=seed:raise ValueError('Invalid stage')
            request=json.loads((d/'request.json').read_text())
            expected={'stage':'measure','format_id':f,'seed_set':seed,'replicas':plan['main_replicas'],'seconds':plan['main_seconds'],'warmup':plan['settle_seconds'],'repetitions':1}
            if any(request.get(k)!=v for k,v in expected.items()):raise ValueError('Measurement differs from declared common conditions')
            ih.add(gate['image_sha256']);vh.add(gate['vectors_manifest_sha256'])
            row=json.loads((d/f'triplet_seed{seed}/summary.json').read_text())
            if not row['windows_valid'] or row['replicas']!=plan['main_replicas'] or row['format_id']!=f or row['seed_set']!=seed:raise ValueError('Invalid or mismatched triplet')
            rows.append(row)
        if len(ih)!=1 or len(vh)!=1:raise ValueError('Image/inputs changed across repeats')
        values=[r['activation_nj_per_dot'] for r in rows]
        fields=['activation_nj_per_dot','delta_w','off_mean_w','on_w','off_drift_w','board_nj_per_dot','dot_per_s']
        stats={key:{'mean':statistics.mean(r[key] for r in rows),'sample_sd':statistics.stdev(r[key] for r in rows)} for key in fields}
        groups[name]={'n_seed_triplets':5,'image_sha256':next(iter(ih)),'vectors_manifest_sha256':next(iter(vh)),'replicas':rows[0]['replicas'],'rows':rows,'statistics':stats,
                      'activation_mean_ci95_t4':[statistics.mean(values)+s*2.7764451052*statistics.stdev(values)/math.sqrt(5) for s in [-1,1]]}
    if len({g['vectors_manifest_sha256'] for g in groups.values()})!=1:raise ValueError('Different vector inputs across formats')
    comparisons={}
    for other in ['mxint8','bf16']:
        diff=[b['activation_nj_per_dot']-s['activation_nj_per_dot'] for s,b in zip(groups['ssnf']['rows'],groups[other]['rows'])]
        mean,sd=statistics.mean(diff),statistics.stdev(diff);ci=[mean+s*2.7764451052*sd/math.sqrt(5) for s in [-1,1]]
        comparisons[f'{other}_minus_ssnf_nj_per_dot']={'paired_by_seed_mean':mean,'sample_sd':sd,'ci95_t4':ci,'ssnf_advantage_resolved':ci[0]>0}
    out={'scope':'Actual U250 same-image OFF/ON/OFF arithmetic activation on common exactly representable synthetic inputs; different static images per format',
         'formats':groups,'comparisons':comparisons,'uncertainty_scope':'Student-t intervals summarize five seed-triplet observations under an independence assumption; they do not include absolute sensor calibration error, workload representativeness, or all thermal/control systematic effects',
         'temperature_and_residual_control_effects_not_removed':True,'all_predeclared_15_main_triplets_included':True}
    (a.root/'aggregate.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()

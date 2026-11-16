#!/usr/bin/env python3
"""Fixed-seed workload activity/correlation audit; no seed selection."""
import hashlib,json,math,statistics,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from source_reference import seeded,advance,operand,records

def corr(a,b):
    ma,mb=statistics.mean(a),statistics.mean(b)
    va=sum((x-ma)**2 for x in a);vb=sum((y-mb)**2 for y in b)
    return sum((x-ma)*(y-mb) for x,y in zip(a,b))/math.sqrt(va*vb) if va*vb else None

def main():
    out=Path(sys.argv[1]);rows=[]
    for seed in range(5):
        states=[seeded(10000+seed,r) for r in range(2)]
        a0=[];b0=[];a1=[];sign=[0,0,0];shared=set();exponents=set();hamming=[[],[],[]];output_hamming=[[],[],[]];prev=None;prev_y=None
        for i in range(4096):
            op=[operand(states[0][:3]),operand(states[0][3:]),operand(states[1][:3])]
            q=[8*x[1][0]+x[2][0] for x in op];a0.append(q[0]);b0.append(q[1]);a1.append(q[2])
            for e,h,low in op:
                shared.update(low);exponents.add(e)
                for j,v in enumerate(h):
                    z=8*v+low[j//8];sign[(z>0)-(z<0)+1]+=1
            rec,result,_=records(states[0])
            if prev:
                for f in range(3):hamming[f].append(bin(prev[f]^rec[f]).count('1'));output_hamming[f].append(bin(prev_y[f]^result[f]).count('1'))
            prev,prev_y=rec,result
            states=[advance(s) for s in states]
        rows.append({'seed_set':seed,'steps_per_representative_replica':4096,'replicas_sampled':[0,1],
                     'q_lane0_A_B_pearson':corr(a0,b0),'q_lane0_replica0_replica1_pearson':corr(a0,a1),'q_lane0_lag1_pearson':corr(a0[:-1],a0[1:]),
                     'observed_shared_values':sorted(shared),'observed_block_exponents':sorted(exponents),'q_sign_counts_negative_zero_positive':sign,
                     'encoded_record_mean_changed_bits_per_step':dict(zip(['ssnf','mxint8','bf16'],map(statistics.mean,hamming))),
                     'fp32_result_mean_changed_bits_per_step':dict(zip(['ssnf','mxint8','bf16'],map(statistics.mean,output_hamming)))})
    out.write_text(json.dumps({'scope':'Representative fixed-seed audit; common LFSR sequences are correlated across time, no independence claim. All five predeclared seeds retained.',
                               'upper_minus16_maps_to_minus15':True,'actual_model_distribution':False,'seed_sets':rows},indent=2))
if __name__=='__main__':main()

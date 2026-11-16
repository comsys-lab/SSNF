"""Independent exact scalar AMS paper-core definitions; no torch/model/GPU use."""
from fractions import Fraction as Q
import math
import random
import struct


def pow2(e):return Q(1<<e) if e>=0 else Q(1,1<<(-e))


def positive_book(mantissa,bias=1):
    result=[]
    for code in range(1<<(2+mantissa)):
        exponent,mant=code>>mantissa,code&((1<<mantissa)-1)
        result.append(pow2(1-bias)*Q(mant,1<<mantissa) if exponent==0 else
            pow2(exponent-bias)*(1+Q(mant,1<<mantissa)))
    return result


def decode(code,mantissa,bias=1):
    n=1<<(2+mantissa)
    return (-1 if code&n else 1)*positive_book(mantissa,bias)[code&(n-1)]


def nearest_code(x,scale,mantissa,bias=1,tie='even'):
    x,scale=Q(x),Q(scale);book=positive_book(mantissa,bias)
    value=abs(x)/scale
    code=min(range(len(book)),key=lambda i:(abs(book[i]-value),i&1 if tie=='even' else 0,i))
    return code|((1<<(2+mantissa)) if x<0 else 0)


def fixed_parent(xs,scales,mantissa,group,bias=1,weights=None,tie='even'):
    xs=list(map(Q,xs));scales=list(map(Q,scales))
    weights=[Q(1)]*len(xs) if weights is None else list(map(Q,weights))
    codes=[nearest_code(x,s,mantissa,bias,tie) for x,s in zip(xs,scales)]
    if group==1:return {'codes':codes,'shared':None,'reconstructed':[decode(c,mantissa,bias)*s for c,s in zip(codes,scales)]}
    output=[];shared=[];result_codes=[]
    for start in range(0,len(xs),group):
        indices=range(start,min(start+group,len(xs)))
        candidates=[]
        for bit in (0,1):
            recon=[decode((codes[i]&~1)|bit,mantissa,bias)*scales[i] for i in indices]
            loss=sum(weights[i]*(value-xs[i])**2 for i,value in zip(indices,recon))
            candidates.append((loss,recon))
        bit=int(candidates[1][0]<candidates[0][0]);shared.append(bit);output+=candidates[bit][1]
        result_codes +=[(codes[i]&~1)|bit for i in indices]
    return {'parent_codes':codes,'codes':result_codes,'shared':shared,'reconstructed':output}


def channel(xs,mantissa,group,bias=1,weights=None,tie='even',scale_round=None):
    xs=list(map(Q,xs));amax=max(map(abs,xs));maximum=max(positive_book(mantissa,bias))
    scale=amax/maximum if amax else Q(1)
    if scale_round:
        scale=Q(struct.unpack('<'+scale_round,struct.pack('<'+scale_round,float(scale)))[0])
        if scale==0:raise ArithmeticError('rounded scale underflow')
    result=fixed_parent(xs,[scale]*len(xs),mantissa,group,bias,weights,tie)
    return dict(result,scale=scale)


def matrix(xs,mantissa,group,axis):
    rows,cols=len(xs),len(xs[0]);maximum=max(positive_book(mantissa))
    if axis=='row':scales=[[max(map(abs,row))/maximum or Q(1)]*cols for row in xs]
    else:
        percol=[max(abs(xs[r][c]) for r in range(rows))/maximum or Q(1) for c in range(cols)]
        scales=[percol]*rows
    return [fixed_parent(row,s,mantissa,group)['reconstructed'] for row,s in zip(xs,scales)]


def selftest():
    rng=random.Random(20261025);channels=0
    for m in (2,3):
        base=positive_book(m,1)
        for b in (2,3):assert positive_book(m,b)==[x*pow2(1-b) for x in base]
        for g in (1,3,4):
            for _ in range(12):
                xs=[Q(rng.randrange(-1024,1025),1024) for i in range(17)]
                weights=[Q(i+1,7) for i in range(len(xs))]
                for h in (None,weights):
                    reference=channel(xs,m,g,1,h)
                    for b in (2,3):
                        candidate=channel(xs,m,g,b,h)
                        assert candidate['codes']==reference['codes']
                        assert candidate['reconstructed']==reference['reconstructed']
                        assert candidate['shared']==reference['shared']
                    channels+=1
        # Exact midpoint cases are included, not dismissed as measure-zero.
        xs=[base[-1]]+[(a+b)/2 for a,b in zip(base,base[1:])]
        for b in (2,3):assert channel(xs,m,3,b)['codes']==channel(xs,m,3,1)['codes']
    xs=[Q(15,8),Q(26,100),Q(26,100),Q(7)]
    even=channel(xs,2,3,tie='even');low=channel(xs,2,3,tie='low')
    assert sum(a!=b for a,b in zip(even['reconstructed'],low['reconstructed']))==3
    # Short-tail policies cannot affect preceding, unchanged full groups.
    for length in (7,8,10,11):
        xs=[Q(rng.randrange(-100,101),100) for _ in range(length)]
        short=channel(xs,3,3);pad=channel(xs+[Q(0)]*((-length)%3),3,3)
        first=(length//3)*3
        assert short['reconstructed'][:first]==pad['reconstructed'][:first]
        assert sum(a!=b for a,b in zip(short['reconstructed'],pad['reconstructed']))<=length%3
    w=[[Q(7),Q(11,10),Q(3,10),Q(1,100)],[Q(1,2),Q(1,4),Q(1,8),Q(1)]]
    assert matrix(w,2,4,'row')!=matrix(w,2,4,'column')
    # Binary power scaling commutes in normal FP32; FP16 subnormal scales
    # provide a deliberate counterexample to an unconditional finite-dtype claim.
    for m in (2,3):
        for amax in (pow2(-24),pow2(-12),Q(1),Q(65504)):
            xs=[amax,amax/3,-amax/7]
            a=channel(xs,m,3,1,scale_round='f');b=channel(xs,m,3,3,scale_round='f')
            assert a['codes']==b['codes'] and a['reconstructed']==b['reconstructed']
    xs=[pow2(-22),pow2(-23)]
    assert channel(xs,3,3,1,scale_round='e')['reconstructed']!=channel(xs,3,3,3,scale_round='e')['reconstructed']
    # Figure 1's four FP5 bit/value pairs identify bias2; changing bias only
    # changes normalization units, not an amax/maxfinite-scaled channel result.
    assert [decode(c,2,2) for c in (4,31,10,13)]==[Q(1,2),Q(-7,2),Q(3,2),Q(5,2)]
    return {'status':'passed','exact_bias_invariance_channel_objectives_tested':channels,
        'normal_fp32_scale_invariance':True,'fp16_subnormal_scale_counterexample':True,
        'rtn_tie_can_change_entire_group':True,'row_column_axis_changes_reconstruction':True,
        'short_tail_difference_bound_verified':True,'figure1_fp5_bias2_bit_values_verified':True}


if __name__=='__main__':
    import json
    print(json.dumps(selftest(),indent=2))

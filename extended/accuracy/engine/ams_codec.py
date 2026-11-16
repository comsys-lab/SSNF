"""AMS paper-core reconstruction with declared conventions, not author code.

Finite E2Mx signed codes, amax/maxfinite scale per output row in FP32,
nearest-even parent RTN, then only shared LSB0/1 selection in physical units.
Parent sign/exponent/upper-mantissa codes are never re-rounded for sharing.
"""
from __future__ import annotations

import math
import struct
import numpy as np
import torch

MAGIC=b'AMSREF01'
HEADER=struct.Struct('<8sQQBBBBI')  # 32 bytes: magic, rows, cols, bits, group, axis, bias, flags
BOOKS={}


def positive_book(bits,bias=1,device='cpu'):
    device=torch.device(device);key=(bits,bias,str(device))
    if key not in BOOKS:
        m=bits-3;values=[]
        for code in range(1<<(bits-1)):
            e,mant=code>>m,code&((1<<m)-1)
            values.append(2.**(1-bias)*mant/(1<<m) if e==0 else 2.**(e-bias)*(1+mant/(1<<m)))
        BOOKS[key]=torch.tensor(values,dtype=torch.float32,device=device)
    return BOOKS[key]


def decode(codes,scales,bits,bias=1,axis='row'):
    codes=codes.long();book=positive_book(bits,bias,codes.device)
    sign=(codes>>(bits-1)).bool();magnitude=codes&((1<<(bits-1))-1)
    normalized=torch.where(sign,-book[magnitude],book[magnitude])
    return normalized*(scales[:,None] if axis=='row' else scales[None,:])


def nearest_codes(x,scales,bits,bias=1,axis='row'):
    book=positive_book(bits,bias,x.device)
    normalized=x/(scales[:,None] if axis=='row' else scales[None,:])
    magnitude=normalized.abs().contiguous()
    upper=torch.bucketize(magnitude,book).clamp_max(len(book)-1)
    lower=(upper-1).clamp_min(0)
    # At amax rounding above maxfinite both adjacent candidates must saturate.
    lower=torch.where(magnitude>=book[-1],upper,lower)
    dl=(magnitude-book[lower]).abs();du=(magnitude-book[upper]).abs()
    tie=(du==dl)&(upper!=lower)
    choose=(du<dl)|(tie&((upper&1)==0))
    code=torch.where(choose,upper,lower).to(torch.uint8)
    code=code|(torch.signbit(x).to(torch.uint8)<<(bits-1))
    return code,tie


@torch.no_grad()
def encode(x,bits,group=1,importance=None,*,axis='row',bias=1,scale_dtype=torch.float32):
    if x.ndim!=2 or bits not in (5,6) or group not in (1,3,4) or axis not in ('row','column'):
        raise ValueError('AMS reference expects a 2-D matrix, FP5/FP6 and row/column scales')
    z=x.float();rows,cols=z.shape
    if not torch.isfinite(z).all():raise ValueError('nonfinite source weight')
    maximum=positive_book(bits,bias,z.device)[-1]
    amax=z.abs().amax(1 if axis=='row' else 0)
    scales=torch.where(amax>0,amax/maximum,torch.ones_like(amax)).to(scale_dtype).float()
    if not torch.isfinite(scales).all() or not (scales>0).all():raise ValueError('nonfinite/underflowed channel scale')
    parent,ties=nearest_codes(z,scales,bits,bias,axis)
    shared=None;group_ties=torch.tensor(0,device=z.device)
    if group==1:
        codes=parent
    else:
        upper=parent&254
        q0=decode(upper,scales,bits,bias,axis);q1=decode(upper|1,scales,bits,bias,axis)
        h=torch.ones(1,cols,device=z.device) if importance is None else torch.broadcast_to(importance,z.shape).float()
        if not torch.isfinite(h).all() or (h<0).any():raise ValueError('importance must be finite and nonnegative')
        e0=(q0-z).square()*h;e1=(q1-z).square()*h
        padding=(-cols)%group
        if padding:
            # Padded slots have zero objective weight; only real weights select the bit.
            e0=torch.nn.functional.pad(e0,(0,padding));e1=torch.nn.functional.pad(e1,(0,padding))
        l0=e0.reshape(rows,-1,group).sum(-1);l1=e1.reshape(rows,-1,group).sum(-1)
        if not torch.isfinite(l0).all() or not torch.isfinite(l1).all():raise ValueError('nonfinite shared-field objective')
        shared=l1<l0;group_ties=(l0==l1).sum()
        codes=upper|shared.repeat_interleave(group,-1)[:,:cols].to(torch.uint8)
        if not torch.equal(codes>>1,parent>>1):raise AssertionError('parent upper code was modified')
    q=decode(codes,scales,bits,bias,axis)
    return {'reconstructed':q,'codes':codes,'parent_codes':parent,'shared':shared,'scales':scales,
        'bits':bits,'group':group,'axis':axis,'bias':bias,
        'diagnostics':{'parent_midpoint_ties':int(ties.sum()),'shared_objective_ties':int(group_ties),
            'zero_channels':int((amax==0).sum()),'real_tail_weights_per_row':cols%group if group>1 else 0}}


def sizes(rows,cols,bits,group,axis='row'):
    if rows<1 or cols<1 or (bits,group) not in [(5,1),(6,1),(6,3),(5,4)]:raise ValueError('unsupported encoded layout')
    scale_bytes=4*(rows if axis=='row' else cols);groups=(cols+group-1)//group
    if group==1:private=(rows*cols*bits+7)//8;shared=0;tail_slots=0
    elif group==3:private=rows*groups*2;shared=0;tail_slots=rows*(groups*3-cols)
    else:private=(rows*cols*4+7)//8;shared=(rows*groups+7)//8;tail_slots=0
    payload=private+shared;total=HEADER.size+scale_bytes+payload
    return {'header_bytes':HEADER.size,'channel_scale_bytes':scale_bytes,'private_or_word_stream_bytes':private,
        'shared_bitstream_bytes':shared,'payload_bytes':payload,'total_bytes':total,
        'physical_zero_padded_private_slots':tail_slots,
        'nominal_full_group_bpe':bits if group==1 else bits-1+1/group,
        'payload_bpe':payload*8/(rows*cols),'total_bpe':total*8/(rows*cols),
        'layout':'full parent bitstream' if group==1 else 'three private5 fields + shared1 in uint16 per group' if group==3
            else 'contiguous private4 stream + contiguous one-bit-per-group stream; no per-group byte padding'}


def pack_stream(values,bits):
    values=np.asarray(values,dtype=np.uint8).reshape(-1)
    stream=((values[:,None]>>np.arange(bits,dtype=np.uint8))&1).reshape(-1)
    return np.packbits(stream,bitorder='little').tobytes()


def unpack_stream(blob,bits,count):
    binary=np.unpackbits(np.frombuffer(blob,dtype=np.uint8),bitorder='little')[:count*bits].reshape(count,bits)
    return (binary.astype(np.uint16)*(1<<np.arange(bits))).sum(-1).astype(np.uint8)


def pack(encoded):
    codes=encoded['codes'].detach().cpu().numpy();rows,cols=codes.shape
    bits,group,axis,bias=(encoded[k] for k in ('bits','group','axis','bias'))
    layout=sizes(rows,cols,bits,group,axis)
    scales=encoded['scales'].detach().cpu().numpy().astype('<f4')
    if len(scales)!=(rows if axis=='row' else cols):raise ValueError('scale count mismatch')
    header=HEADER.pack(MAGIC,rows,cols,bits,group,int(axis=='column'),bias,0)
    if group==1:payload=pack_stream(codes,bits)
    elif group==3:
        n=(cols+2)//3;padded=np.zeros((rows,n*3),dtype=np.uint16);padded[:,:cols]=(codes>>1).astype(np.uint16)
        private=padded.reshape(rows,n,3);shared=encoded['shared'].detach().cpu().numpy().astype(np.uint16)
        require_shared=np.repeat(shared,3,axis=1)[:,:cols]
        if not np.array_equal(codes&1,require_shared):raise ValueError('inconsistent shared LSB codes')
        words=private[:,:,0]|(private[:,:,1]<<5)|(private[:,:,2]<<10)|(shared<<15)
        payload=words.astype('<u2').tobytes()
    else:
        shared=encoded['shared'].detach().cpu().numpy().astype(np.uint8)
        if not np.array_equal(codes&1,np.repeat(shared,4,axis=1)[:,:cols]):raise ValueError('inconsistent shared LSB codes')
        payload=pack_stream(codes>>1,4)+pack_stream(shared,1)
    blob=header+scales.tobytes()+payload
    if len(blob)!=layout['total_bytes']:raise AssertionError('serialized byte count mismatch')
    return blob


def unpack(blob):
    magic,rows,cols,bits,group,axis,bias,flags=HEADER.unpack(blob[:HEADER.size])
    if magic!=MAGIC or flags or axis not in (0,1):raise ValueError('invalid AMS reference header')
    axis='column' if axis else 'row';layout=sizes(rows,cols,bits,group,axis)
    if len(blob)!=layout['total_bytes']:raise ValueError('encoded tensor length mismatch')
    offset=HEADER.size;end=offset+layout['channel_scale_bytes']
    scales=torch.from_numpy(np.frombuffer(blob[offset:end],dtype='<f4').copy());data=blob[end:]
    shared=None
    if group==1:codes=unpack_stream(data,bits,rows*cols).reshape(rows,cols)
    elif group==3:
        n=(cols+2)//3;words=np.frombuffer(data,dtype='<u2').reshape(rows,n)
        shared=(words>>15).astype(np.uint8)
        private=np.stack([words&31,(words>>5)&31,(words>>10)&31],-1).reshape(rows,n*3)
        codes=((private<<1)|np.repeat(shared,3,axis=1))[:,:cols].astype(np.uint8)
    else:
        private_bytes=layout['private_or_word_stream_bytes'];n=(cols+3)//4
        private=unpack_stream(data[:private_bytes],4,rows*cols).reshape(rows,cols)
        shared=unpack_stream(data[private_bytes:],1,rows*n).reshape(rows,n)
        codes=(private<<1)|np.repeat(shared,4,axis=1)[:,:cols]
    codes=torch.from_numpy(codes.copy());shared=torch.from_numpy(shared.copy()) if shared is not None else None
    return {'codes':codes,'shared':shared,'scales':scales,'reconstructed':decode(codes,scales,bits,bias,axis),
        'bits':bits,'group':group,'axis':axis,'bias':bias,'layout':layout}


def selftest():
    from fractions import Fraction as Q
    from . import ams_math_reference as reference
    reference.selftest();torch.manual_seed(831)
    results=[]
    for bits,group in [(6,1),(5,1),(6,3),(5,4)]:
        m=bits-3;maximum=float(max(reference.positive_book(m)))
        x=torch.randint(-96,97,(3,17)).float()/16;x[:,0]=maximum
        h=torch.arange(1,18).float()[None]
        for weighted in (False,True):
            importance=h if weighted and group>1 else None
            encoded=encode(x,bits,group,importance)
            for row in range(len(x)):
                scalar=reference.channel([Q(float(v)) for v in x[row]],m,group,
                    weights=[Q(float(v)) for v in h[0]] if importance is not None else None)
                assert encoded['codes'][row].tolist()==scalar['codes']
                assert encoded['reconstructed'][row].tolist()==[float(v) for v in scalar['reconstructed']]
            restored=unpack(pack(encoded))
            assert torch.equal(encoded['codes'],restored['codes'])
            assert torch.equal(encoded['scales'],restored['scales'])
            assert torch.equal(encoded['reconstructed'],restored['reconstructed'])
            assert torch.equal(encoded['codes']>>1,encoded['parent_codes']>>1) or group==1
        # Bias rescaling leaves exact parent/shared code choices unchanged.
        a=encode(x,bits,group,h if group>1 else None,bias=1)
        b=encode(x,bits,group,h if group>1 else None,bias=3)
        assert torch.equal(a['codes'],b['codes']) and torch.equal(a['reconstructed'],b['reconstructed'])
        results.append({'bits':bits,'group':group,'actual_bytes':len(pack(a)),**sizes(*x.shape,bits,group)})
    assert sizes(2,64,5,4)['payload_bpe']==4.25
    assert sizes(1,4096,6,3)['physical_zero_padded_private_slots']==2
    z=encode(torch.zeros(2,12),6,3);assert not z['codes'].any() and not z['reconstructed'].any()
    for bits in (5,6):
        for bias in (1,2,3):
            # Every signed normal/subnormal/zero code independently matches Fraction.
            codes=torch.arange(1<<bits,dtype=torch.uint8)[None]
            values=decode(codes,torch.ones(1),bits,bias).tolist()[0]
            assert values==[float(reference.decode(i,bits-3,bias)) for i in range(1<<bits)]
    # Parent sign (including the negative-zero code for tiny negative values) is kept.
    x=torch.tensor([[-.01,7.,0.,-0.]])
    e=encode(x,5,1);assert int(e['codes'][0,0])==16 and int(e['codes'][0,3])==16
    probe=torch.tensor([[.071,-.028,.011,.001,.023,-.019,.003],[.033,.007,-.021,.015,-.002,.009,.017]])
    sensitivity={}
    for bits,group in [(6,3),(5,4)]:
        row=encode(probe,bits,group);column=encode(probe,bits,group,axis='column')
        half=encode(probe,bits,group,scale_dtype=torch.float16)
        assert torch.equal(unpack(pack(column))['reconstructed'],column['reconstructed'])
        difference=(row['reconstructed']-column['reconstructed']).abs().max().item()
        assert difference>0
        sensitivity[f'FP{bits}g{group}']={'row_vs_column_max_abs':difference,
            'fp32_vs_fp16_scale_max_abs':(row['reconstructed']-half['reconstructed']).abs().max().item(),
            'fp32_vs_fp16_scale_parent_codes_changed':int((row['parent_codes']!=half['parent_codes']).sum()),
            'fp32_vs_fp16_scale_shared_bits_changed':int((row['shared']!=half['shared']).sum()),
            'fp32_vs_fp16_scale_after_bf16_cast_changed':int((row['reconstructed'].bfloat16()!=half['reconstructed'].bfloat16()).sum())}
    print('PASS AMS scalar/Fraction agreement, exhaustive finite codebooks, fixed parent upper fields, H/MSE, short tails, exact numerical pack/unpack, 4.25b unpadded payload, FP32 scale/header bytes, bias invariance')
    print('CPU synthetic convention sensitivity (no model/PPL):',sensitivity)
    return {'layouts':results,'synthetic_convention_sensitivity':sensitivity}


if __name__=='__main__':
    import json
    print(json.dumps(selftest(),indent=2))

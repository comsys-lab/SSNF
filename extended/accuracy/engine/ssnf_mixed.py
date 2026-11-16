"""CPU reference payloads and exact byte budgets for the prespecified SSNF menu.

This module does not select a geometry menu from evaluation data. Codes follow
the unchanged SSNF quantizer, including its ascending shared-field tie order.
"""
from __future__ import annotations
import math
import struct
from fractions import Fraction
import numpy as np
import torch

MENU={'w':((3,16),(3,4)), 'k':((4,4),(3,8)), 'v':((5,8),(4,8))}
HEADER=struct.Struct('<8sQBBBBIII')
MAGIC=b'ACCSSNF1'

def block_bytes(geometry):
    lb,gs=geometry
    if not 1<=lb<=7 or gs not in (2,4,8,16,32):raise ValueError('invalid geometry')
    return 1+4*(8-lb)+math.ceil((32//gs)*lb/8)

def sizes(blocks,low,high,count):
    if not 0<=count<=blocks:raise ValueError('invalid high count')
    payload=(blocks-count)*block_bytes(low)+count*block_bytes(high)
    tags=(blocks+7)//8;rank=((blocks+255)//256)*8
    return {'payload_bytes':payload,'tag_bytes':tags,'rank_bytes':rank,'header_bytes':32,
            'total_bytes':payload+tags+rank+32}

def high_count(blocks,low,high,budget,charge_map=True):
    lowbytes,highbytes=block_bytes(low),block_bytes(high)
    if highbytes<=lowbytes:raise ValueError('non-increasing geometry costs')
    quota=math.floor(Fraction(str(budget))*blocks*4)
    overhead=((blocks+7)//8+((blocks+255)//256)*8+32) if charge_map else 0
    count=(quota-blocks*lowbytes-overhead)//(highbytes-lowbytes)
    if count<0:raise ValueError('budget below low geometry plus metadata')
    return min(blocks,int(count))

def _pack(codes,width):
    c=np.asarray(codes,dtype=np.int64)
    bits=((c[...,None]&((1<<width)-1))>>np.arange(width))&1
    return np.packbits(bits.reshape(len(c),-1).astype(np.uint8),axis=1,bitorder='little')

def _unpack(raw,width,count):
    bits=np.unpackbits(raw,axis=1,bitorder='little')
    if bits.shape[1]>width*count and bits[:,width*count:].any():raise ValueError('nonzero field padding')
    codes=(bits[:,:width*count].reshape(len(raw),count,width)*(1<<np.arange(width))).sum(-1)
    return np.where(codes>=(1<<(width-1)),codes-(1<<width),codes)

@torch.no_grad()
def encode_geometry(x,geometry,importance=None,search=False):
    """Encode actual integer fields on x.device, then move only codes to CPU."""
    lb,gs=geometry;block_bytes(geometry)
    z=x.reshape(-1,32).float();n=len(z);ng=32//gs
    if not n:return np.empty((0,block_bytes(geometry)),np.uint8)
    h=torch.ones_like(z) if importance is None else torch.broadcast_to(importance,x.shape).reshape(-1,32).float()
    e=torch.floor(torch.log2(z.abs().amax(-1,keepdim=True).clamp_min(1e-30)))-6
    if not ((e>=-127)&(e<=127)).all():raise ValueError('E8M0 exponent outside finite range')
    scale=torch.exp2(e);upscale=scale*(1<<lb);qmax=(1<<(7-lb))-1
    lo,hi=-(1<<(lb-1)),(1<<(lb-1))-1
    if not search:
        upper=torch.round(z/upscale).clamp(-qmax,qmax)
        residual=(z-upper*upscale).reshape(-1,ng,gs);hg=h.reshape(-1,ng,gs)
        shared=torch.round((residual*hg).sum(-1)/hg.sum(-1).clamp_min(1e-30)/scale).clamp(lo,hi)
    else:
        loss=torch.full((n,ng),float('inf'),device=z.device)
        upper=torch.zeros_like(z);shared=torch.zeros((n,ng),device=z.device)
        for code in range(lo,hi+1):
            shifted=code*scale;candidate_upper=torch.round((z-shifted)/upscale).clamp(-qmax,qmax)
            candidate=candidate_upper*upscale+shifted
            error=(h*(candidate-z).square()).reshape(-1,ng,gs).sum(-1)
            take=error<loss
            upper=torch.where(take.repeat_interleave(gs,-1),candidate_upper,upper)
            shared=torch.where(take,torch.full_like(shared,code),shared)
            loss=torch.where(take,error,loss)
    exponent=(e.cpu().numpy().astype(np.int16)+127).astype(np.uint8)
    return np.concatenate([exponent,_pack(upper.cpu().numpy(),8-lb),_pack(shared.cpu().numpy(),lb)],axis=1)

def decode_geometry(raw,geometry):
    lb,gs=geometry;raw=np.asarray(raw,dtype=np.uint8).reshape(-1,block_bytes(geometry))
    n=len(raw)
    if not n:return torch.empty((0,32),dtype=torch.float32)
    upper=_unpack(raw[:,1:1+4*(8-lb)],8-lb,32)
    shared=_unpack(raw[:,1+4*(8-lb):],lb,32//gs)
    scale=np.exp2(raw[:,0].astype(np.int16)-127).astype(np.float32)[:,None]
    value=upper.astype(np.float32)*(scale*(1<<lb))+np.repeat(shared.astype(np.float32),gs,axis=1)*scale
    return torch.from_numpy(value.astype(np.float32))

def pack(x,mask,low,high,importance=None,search=False):
    z=x.reshape(-1,32);mask=mask.reshape(-1).bool().cpu();n=len(z);count=int(mask.sum())
    if len(mask)!=n:raise ValueError('mask/block mismatch')
    h=None if importance is None else torch.broadcast_to(importance,x.shape).reshape(-1,32)
    device_mask=mask.to(z.device)
    a=encode_geometry(z[~device_mask],low,None if h is None else h[~device_mask],search)
    b=encode_geometry(z[device_mask],high,None if h is None else h[device_mask],search)
    header=HEADER.pack(MAGIC,n,*low,*high,count,256,0)
    tags=np.packbits(mask.numpy(),bitorder='little').tobytes()
    ranks=np.asarray([int(mask[:i].sum()) for i in range(0,n,256)],dtype='<u8').tobytes()
    blob=header+tags+ranks+a.tobytes()+b.tobytes()
    if len(blob)!=sizes(n,low,high,count)['total_bytes']:raise AssertionError('serialized byte count mismatch')
    return blob

def unpack(blob):
    if len(blob)<32:raise ValueError('short mixed header')
    magic,n,ll,lg,hl,hg,count,interval,flags=HEADER.unpack(blob[:32])
    if magic!=MAGIC or interval!=256 or flags or count>n:raise ValueError('invalid mixed header')
    low,high=(ll,lg),(hl,hg)
    if len(blob)!=sizes(n,low,high,count)['total_bytes']:raise ValueError('payload byte length mismatch')
    end=32+(n+7)//8;alltags=np.unpackbits(np.frombuffer(blob[32:end],np.uint8),bitorder='little')
    if alltags[n:].any():raise ValueError('nonzero tag padding')
    mask=alltags[:n].astype(bool)
    if int(mask.sum())!=count:raise ValueError('tag count mismatch')
    rankend=end+((n+255)//256)*8;ranks=np.frombuffer(blob[end:rankend],dtype='<u8')
    if any(int(ranks[j])!=int(mask[:i].sum()) for j,i in enumerate(range(0,n,256))):raise ValueError('rank mismatch')
    split=rankend+(n-count)*block_bytes(low)
    a=decode_geometry(np.frombuffer(blob[rankend:split],np.uint8),low)
    b=decode_geometry(np.frombuffer(blob[split:],np.uint8),high)
    value=torch.empty((n,32));m=torch.from_numpy(mask);value[~m]=a;value[m]=b
    return value,m,{'low':low,'high':high,**sizes(n,low,high,count)}

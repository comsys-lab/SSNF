"""Integer-only source oracle. Existing arithmetic oracle is reused unchanged."""
import random
from reference import ssnf_operand,mxint8_operand,bf16_operand,ssnf_dot,mxint8_dot,fp32_exact,bf16_product,fp32_add
MASK=(1<<64)-1
POLY=0xd800000000000000
def advance(states):
    return [(x>>1)^(POLY if x&1 else 0) for x in states]
def seeded(index,replica=0):
    r=random.Random(20261029+index*65537+replica*257)
    return [r.getrandbits(64) or 1 for _ in range(6)]
def pack_states(states):
    return sum(x<<(64*i) for i,x in enumerate(states))
def unpack_states(bits):
    return [(bits>>(64*i))&MASK for i in range(6)]
def signed(x,width):
    return x-(1<<width) if x&(1<<(width-1)) else x
def operand(states):
    bits=pack_states(states)
    h=[signed((bits>>(5*i))&31,5) for i in range(32)]
    h=[-15 if x==-16 else x for x in h]
    low=[signed((bits>>(160+3*g))&7,3) for g in range(4)]
    return signed((bits>>172)&7,3),h,low
def records(states):
    ab=[operand(states[:3]),operand(states[3:])]
    q=[[8*h+o[2][i//8] for i,h in enumerate(o[1])] for o in ab]
    bf=[[fp32_exact(v,o[0])>>16 for v in qq] for qq,o in zip(q,ab)]
    payloads=[ssnf_operand(*ab[0])+ssnf_operand(*ab[1]),
              mxint8_operand(ab[0][0]+6,q[0])+mxint8_operand(ab[1][0]+6,q[1]),
              bf16_operand(bf[0])+bf16_operand(bf[1])]
    sy,si=ssnf_dot(*ab)
    my,mi=mxint8_dot((ab[0][0]+6,q[0]),(ab[1][0]+6,q[1]))
    products=[bf16_product(a,b) for a,b in zip(*bf)]
    while len(products)>1:products=[fp32_add(a,b) for a,b in zip(products[::2],products[1::2])]
    assert sy==my and (products[0]==sy or products[0]&0x7fffffff==sy&0x7fffffff==0)
    return [int.from_bytes(x,'little') for x in payloads],[sy,my,products[0]],[si,mi,0]

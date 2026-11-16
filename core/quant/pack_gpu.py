"""GPU (torch) SSNF-INT8 weight packer — byte-identical to ssnf_lib.pack.pack_weight, but on-device.

Notation: `l_b` = shared field bitwidth, `g_size` = mantissa group size. Builds the
scale_exp / upper_cm / shared_cm planes in float64 (matches numpy rounding bit-for-bit).
Verified byte-exact by check_pack_gpu.py.
"""
import math
import torch

BLOCK = 32
E_MAX = 6  # E8M0 headroom: scale = 2^(floor(log2 max_abs) - 6), matches ssnf_lib.pack._e8m0_scale


def _pack_codes_lsb(codes: torch.Tensor, width: int) -> torch.Tensor:
    """codes int[..., n] (two's-complement width-bit) -> uint8[..., ceil(n*width/8)], LSB-first.
    Mirrors numpy _pack_codes_lsb (np.packbits bitorder='little'). int16 keeps memory small."""
    n = codes.shape[-1]
    mask = (1 << width) - 1
    c = (codes & mask).to(torch.int16)
    w = torch.arange(width, device=codes.device, dtype=torch.int16)
    bits = ((c.unsqueeze(-1) >> w) & 1).to(torch.int16)             # [..., n, width]
    bits = bits.reshape(*codes.shape[:-1], n * width)
    pad = (-(n * width)) % 8
    if pad:
        bits = torch.nn.functional.pad(bits, (0, pad))
    bits = bits.reshape(*bits.shape[:-1], -1, 8)                    # [..., nbytes, 8]
    wj = (1 << torch.arange(8, device=codes.device, dtype=torch.int16))  # LSB-first byte weights
    return (bits * wj).sum(-1).to(torch.uint8)                      # [..., nbytes] in [0,255]


def _pack_chunk(Wc: torch.Tensor, l_b: int, g_size: int, nb: int, n_group: int,
                wbits: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """pack one OUT-chunk Wc [C, K] -> (scale_exp[nb,C] int8, upper_cm[nb,C,UB], shared_cm[nb,C,SB])."""
    C = Wc.shape[0]
    xf = Wc.reshape(C * nb, BLOCK).to(torch.float64)
    max_abs = xf.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    exp = (torch.floor(torch.log2(max_abs)) - E_MAX).clamp(-127, 127)   # (B,1)
    s_base = torch.exp2(exp)
    s_unshared = s_base * float(1 << l_b)
    q_max = (1 << (7 - l_b)) - 1
    h_word = torch.round(xf / s_unshared).clamp(-q_max, q_max).to(torch.int16)  # [B,32]
    residual = xf - h_word.to(torch.float64) * s_unshared
    res_avg = residual.reshape(-1, n_group, g_size).mean(-1)
    s_min, s_max = -(1 << (l_b - 1)), (1 << (l_b - 1)) - 1
    l_g_word = torch.round(res_avg / s_base).clamp(s_min, s_max).to(torch.int16)
    del xf, residual, max_abs, s_base
    scale_exp = exp.reshape(C, nb).round().to(torch.int8).t().contiguous()      # [nb, C]
    up = _pack_codes_lsb(h_word.reshape(C, nb, BLOCK), wbits)                  # [C, nb, UB]
    sh = _pack_codes_lsb(l_g_word.reshape(C, nb, n_group), l_b)                   # [C, nb, SB]
    return scale_exp, up.permute(1, 0, 2).contiguous(), sh.permute(1, 0, 2).contiguous()


def _pack_chunk_gh(Wc: torch.Tensor, l_b: int, g_size: int, nb: int, n_group: int, wbits: int,
                   Hrow: torch.Tensor | None,
                   radius: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """greedy+H encoder for one OUT-chunk: H-weighted shared mean + radius search re-rounding
    upper per candidate; planes byte-format identical to _pack_chunk (Hrow=None -> uniform)."""
    C = Wc.shape[0]
    xb = Wc.reshape(C * nb, BLOCK).to(torch.float64)                     # [B,32]  row = c*nb + bidx
    B = xb.shape[0]
    exp = (torch.floor(torch.log2(xb.abs().amax(-1, keepdim=True).clamp(min=1e-30))) - E_MAX).clamp(-127, 127)
    s = torch.exp2(exp); s_up = s * float(1 << l_b)
    q_max = (1 << (7 - l_b)) - 1
    s_min, s_max = -(1 << (l_b - 1)), (1 << (l_b - 1)) - 1
    if Hrow is None:
        hw = torch.ones_like(xb)
    else:
        hw = Hrow.reshape(nb, BLOCK).to(torch.float64).repeat(C, 1)      # row c*nb+bidx -> H[bidx] (input-chan)
    h_word0 = torch.round(xb / s_up).clamp(-q_max, q_max)                    # seed upper (plain closed-form)
    rg = ((xb - h_word0 * s_up) / s).reshape(B, n_group, g_size)                 # residual in scale units
    hwg = hw.reshape(B, n_group, g_size)
    rm = torch.round((rg * hwg).sum(-1, keepdim=True) / hwg.sum(-1, keepdim=True).clamp(min=1e-30))  # [B,ng,1]
    best_err = best_h_word = best_l_g_word = None
    for d in range(-radius, radius + 1):
        sh = (rm + d).clamp(s_min, s_max)                               # [B,ng,1]
        shb = sh.expand(B, n_group, g_size).reshape(B, BLOCK)
        up = torch.round((xb - shb * s) / s_up).clamp(-q_max, q_max)    # re-round upper for this shared
        err = (hw * (up * s_up + shb * s - xb).pow(2)).reshape(B, n_group, g_size).sum(-1)  # [B,ng]
        if best_err is None:
            best_err, best_h_word, best_l_g_word = err, up, sh.squeeze(-1)
        else:
            take = err < best_err                                      # [B,ng]
            best_err = torch.where(take, err, best_err)
            te = take.unsqueeze(-1).expand(B, n_group, g_size).reshape(B, BLOCK)
            best_h_word = torch.where(te, up, best_h_word)
            best_l_g_word = torch.where(take, sh.squeeze(-1), best_l_g_word)
    h_word = best_h_word.to(torch.int16)
    l_g_word = best_l_g_word.to(torch.int16)
    scale_exp = exp.reshape(C, nb).round().to(torch.int8).t().contiguous()
    up_p = _pack_codes_lsb(h_word.reshape(C, nb, BLOCK), wbits)
    sh_p = _pack_codes_lsb(l_g_word.reshape(C, nb, n_group), l_b)
    return scale_exp, up_p.permute(1, 0, 2).contiguous(), sh_p.permute(1, 0, 2).contiguous()


def pack_weight_ssnf_gpu(W: torch.Tensor, l_b: int, g_size: int, H: torch.Tensor | None = None,
                         radius: int = 0, out_chunk: int = 2048) -> dict:
    """Pack W [OUT, K] (K%32==0) to SSNF planes {scale_exp, upper_cm, shared_cm, nb, ...} on-GPU.
    radius=0/H=None -> plain encoder (byte-identical to ssnf_lib.pack.pack_weight); otherwise the
    DEPLOYED greedy+H encoder. Stored planes / kernel / decode are format-identical either way."""
    OUT, K = int(W.shape[0]), int(W.shape[1])
    assert K % BLOCK == 0
    nb = K // BLOCK
    wbits = 8 - l_b          # unshared bitwidth, the paper's h_b
    n_group = BLOCK // g_size
    UB = (BLOCK * wbits) // 8
    SB = math.ceil(n_group * l_b / 8)
    gh = (H is not None) or (radius > 0)
    chunk = min(out_chunk, 1024) if gh else out_chunk           # greedy holds ~5 temporaries/candidate
    se, uc, sc = [], [], []
    for i in range(0, OUT, chunk):
        if gh:
            a, b, c = _pack_chunk_gh(W[i:i + chunk], l_b, g_size, nb, n_group, wbits, H, radius)
        else:
            a, b, c = _pack_chunk(W[i:i + chunk], l_b, g_size, nb, n_group, wbits)
        se.append(a); uc.append(b); sc.append(c)
    return dict(scale_exp=torch.cat(se, 1), upper_cm=torch.cat(uc, 1), shared_cm=torch.cat(sc, 1),
                nb=nb, OUT=OUT, K=K, l_b=l_b, g_size=g_size, UB=UB, SB=SB, n_group=n_group)


def pack_weight_mxint8_gpu(W: torch.Tensor, out_chunk: int = 4096) -> dict:
    """W [OUT,K] bf16/float cuda (K%32==0) -> {scale_exp[nb,OUT] int8, qweight_cm[nb,OUT,32] int8, nb}.
    On-GPU, float64, byte-identical to ssnf_lib.pack.pack_weight_mxint8 (the fair MXINT8 twin of SSNF)."""
    OUT, K = int(W.shape[0]), int(W.shape[1])
    assert K % BLOCK == 0
    nb = K // BLOCK
    se, qc = [], []
    for i in range(0, OUT, out_chunk):
        Wc = W[i:i + out_chunk]
        C = Wc.shape[0]
        xf = Wc.reshape(C * nb, BLOCK).to(torch.float64)
        exp = (torch.floor(torch.log2(xf.abs().amax(-1, keepdim=True).clamp(min=1e-30))) - E_MAX).clamp(-127, 127)
        q = torch.round(xf / torch.exp2(exp)).clamp(-127, 127).to(torch.int8)   # [C*nb,32]
        se.append(exp.reshape(C, nb).to(torch.int8).t().contiguous())           # [nb,C]
        qc.append(q.reshape(C, nb, BLOCK).permute(1, 0, 2).contiguous())        # [nb,C,32]
    return dict(scale_exp=torch.cat(se, 1), qweight_cm=torch.cat(qc, 1), nb=nb, OUT=OUT, K=K)


def pack_weight_mxfp8_gpu(W: torch.Tensor, out_chunk: int = 4096) -> dict:
    """MXFP8-E4M3 twin of pack_weight_mxint8_gpu: per-32 E8M0 scale (E_MAX=8) + e4m3 element
    bytes stored int8, same [nb,OUT,32] layout as MXINT8."""
    OUT, K = int(W.shape[0]), int(W.shape[1])
    assert K % BLOCK == 0
    nb = K // BLOCK
    se, qc = [], []
    for i in range(0, OUT, out_chunk):
        Wc = W[i:i + out_chunk]
        C = Wc.shape[0]
        xf = Wc.reshape(C * nb, BLOCK).to(torch.float32)
        exp = (torch.floor(torch.log2(xf.abs().amax(-1, keepdim=True).clamp(min=1e-30))) - 8).clamp(-127, 127)
        # clamp to e4m3 max (448) BEFORE the cast — matches the CUDA write kernel's __NV_SATFINITE.
        # Without this, block maxima in (448,512) become NaN (torch e4m3fn casts overflow -> NaN).
        q = (xf / torch.exp2(exp)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.int8)  # e4m3 bytes as int8
        se.append(exp.reshape(C, nb).to(torch.int8).t().contiguous())           # [nb,C]
        qc.append(q.reshape(C, nb, BLOCK).permute(1, 0, 2).contiguous())        # [nb,C,32]
    return dict(scale_exp=torch.cat(se, 1), qweight_cm=torch.cat(qc, 1), nb=nb, OUT=OUT, K=K)


def _f_to_e2m3_torch(x: torch.Tensor) -> torch.Tensor:
    """float -> 6-bit E2M3 code (0..63), mirrors kernel/mxfp.cu::f_to_e2m3 (round-nearest, sat 7.5)."""
    sign = torch.where(x < 0, 0x20, 0).int()
    a = x.abs().clamp(max=7.5)
    is_sub = a < 1.0
    e = (torch.floor(torch.log2(a.clamp(min=1e-30))).int() + 1).clamp(1, 3)     # normal exp 1..3
    base = torch.exp2((e - 1).float())
    step = torch.exp2((e - 4).float())
    m = torch.round((a - base) / step).int()                                    # 0..8
    carry = m >= 8
    e = torch.where(carry, (e + 1).clamp(max=3), e)
    m = torch.where(carry, torch.zeros_like(m), m)
    ms = torch.round(a * 8.0).int()                                             # subnormal m 0..8
    subc = ms >= 8
    e = torch.where(is_sub, torch.where(subc, torch.ones_like(ms), torch.zeros_like(ms)), e)
    m = torch.where(is_sub, torch.where(subc, torch.zeros_like(ms), ms), m)
    return (sign | (e << 3) | m).to(torch.uint8)


def pack_weight_mxfp6_gpu(W: torch.Tensor, out_chunk: int = 4096) -> dict:
    """MXFP6-E2M3 twin of pack_weight_mxint8_gpu: per-32 E8M0 scale (E_MAX=2) + 6-bit e2m3
    codes packed 4-per-3-byte-word ([nb,OUT,24]; matches the CUDA e2m3 decode). 6.25 b/elem."""
    OUT, K = int(W.shape[0]), int(W.shape[1])
    assert K % BLOCK == 0
    nb = K // BLOCK
    se, qc = [], []
    for i in range(0, OUT, out_chunk):
        Wc = W[i:i + out_chunk]
        C = Wc.shape[0]
        xf = Wc.reshape(C * nb, BLOCK).to(torch.float32)
        exp = (torch.floor(torch.log2(xf.abs().amax(-1, keepdim=True).clamp(min=1e-30))) - 2).clamp(-127, 127)
        codes = _f_to_e2m3_torch(xf / torch.exp2(exp)).int()                    # [C*nb,32] 0..63
        c = codes.reshape(C * nb, 8, 4)                                         # 8 groups of 4 codes
        word = c[..., 0] | (c[..., 1] << 6) | (c[..., 2] << 12) | (c[..., 3] << 18)   # [C*nb,8]
        b = torch.stack([word & 0xff, (word >> 8) & 0xff, (word >> 16) & 0xff], -1)   # [C*nb,8,3]
        q = b.reshape(C * nb, 24).to(torch.uint8).view(torch.int8)             # 24 B/block
        se.append(exp.reshape(C, nb).to(torch.int8).t().contiguous())          # [nb,C]
        qc.append(q.reshape(C, nb, 24).permute(1, 0, 2).contiguous())          # [nb,C,24]
    return dict(scale_exp=torch.cat(se, 1), qweight_cm=torch.cat(qc, 1), nb=nb, OUT=OUT, K=K)

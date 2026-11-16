# ssnf_lib/pack.py — SSNF-signed numerics + packing layer (offline / preprocessing).
# Notation: `l_b` = shared field bitwidth, `g_size` = mantissa group size; reconstruction word =
# h_word*2^l_b + l_g_word, scaled by the E8M0 2^e. Bit-exact certified ground truth — the CUDA
# kernels (kernel/) and the oracle (reference.py) verify against this exact math; DO NOT "improve".

import math
import numpy as np

# --- OCP MX constants (shared across the whole project) ----------------------
BLOCK = 32        # OCP MX block size
E_MAX = 6         # MXINT8 (elem_bitwidth=8): E_max = n - 2 = 6


# =============================================================================
#  NUMERICS  (matches SSNF_signed)
# =============================================================================
def _e8m0_scale(max_abs: np.ndarray | float) -> np.ndarray:
    """E8M0 shared scale for an MXINT8 block: 2^(floor(log2(max_abs)) - 6)."""
    max_abs = np.maximum(np.asarray(max_abs, dtype=np.float64), 1e-30)
    exp = np.floor(np.log2(max_abs)) - E_MAX
    exp = np.clip(exp, -127, 127)
    return np.exp2(exp)


def decompose(x_blocked: np.ndarray, l_b: int, g_size: int, H: np.ndarray | None = None,
              radius: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize float blocks (B,32) -> (scale (B,1), h_word (B,32) int, l_g_word (B,n_group) int),
    single FP-domain rounding; H = per-channel weights for a weighted shared mean (None = uniform);
    radius>0 = trunc search re-rounding h_word per candidate (radius=0 == weighted mean == hmean)."""
    xf = np.asarray(x_blocked, dtype=np.float64)
    B = xf.shape[0]
    n_group = BLOCK // g_size
    max_abs = np.abs(xf).max(axis=1, keepdims=True)
    s_base = _e8m0_scale(max_abs)                                   # (B,1)
    s_unshared = s_base * (1 << l_b)
    q_max = (1 << (7 - l_b)) - 1
    s_min, s_max = -(1 << (l_b - 1)), (1 << (l_b - 1)) - 1
    Hb = (np.ones_like(xf) if H is None
          else np.broadcast_to(np.asarray(H, np.float64), xf.shape))   # (B,32)
    Hg = Hb.reshape(B, n_group, g_size)

    h_word = np.clip(np.round(xf / s_unshared), -q_max, q_max).astype(np.int64)
    residual = (xf - h_word * s_unshared).reshape(B, n_group, g_size)
    res_avg = (Hg * residual).sum(2) / Hg.sum(2)                    # (B,n_group) weighted mean
    rm = np.clip(np.round(res_avg / s_base), s_min, s_max)          # rounded shared (seed)
    if radius == 0:
        return s_base, h_word, rm.astype(np.int64)

    best_err = np.full((B, n_group), np.inf)
    best_h_word = h_word.copy()
    best_l_g_word = rm.copy()
    for d in range(-radius, radius + 1):
        sh = np.clip(rm + d, s_min, s_max)                         # (B,n_group)
        sh_e = np.repeat(sh, g_size, axis=1)                           # (B,32)
        up = np.clip(np.round((xf - sh_e * s_base) / s_unshared), -q_max, q_max)
        recon = up * s_unshared + sh_e * s_base
        err = (Hb * (recon - xf) ** 2).reshape(B, n_group, g_size).sum(2)   # (B,n_group)
        take = err < best_err
        best_err = np.where(take, err, best_err)
        best_l_g_word = np.where(take, sh, best_l_g_word)
        take_e = np.repeat(take, g_size, axis=1)
        best_h_word = np.where(take_e, up, best_h_word)
    return s_base, best_h_word.astype(np.int64), best_l_g_word.astype(np.int64)


def decompose_lightms(x_blocked: np.ndarray, l_b: int, g_size: int,
                      H: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """INT-domain oracle for csrc decompose_lightms_block (integer upper/low-bit residual, shared =
    round-mean of low bits, H-weighted if H); replicates the device integer rounding exactly."""
    xf = np.asarray(x_blocked, dtype=np.float64)
    B = xf.shape[0]; ng = BLOCK // g_size
    max_abs = np.abs(xf).max(axis=1, keepdims=True)
    exp = np.clip(np.floor(np.log2(np.maximum(max_abs, 1e-30))) - E_MAX, -127, 127)   # E8M0 ea
    s_base = np.exp2(exp)                                                              # (B,1)
    q_max = (1 << (7 - l_b)) - 1
    s_min, s_max = -(1 << (l_b - 1)), (1 << (l_b - 1)) - 1
    half_u = 1 << (l_b - 1)
    q8 = np.rint(xf / s_base).astype(np.int64)                                         # (B,32)
    up = np.sign(q8).astype(np.int64) * ((np.abs(q8) + half_u) >> l_b)                   # signed round-half-up
    up = np.clip(up, -q_max, q_max)
    res = np.clip(q8 - (up << l_b), s_min, s_max)                                        # int low bits
    resg = res.reshape(B, ng, g_size)
    if H is None:
        ssum = resg.sum(2)                                                             # int round-mean
        half_g, lg = g_size >> 1, int(g_size).bit_length() - 1
        m = np.sign(ssum).astype(np.int64) * ((np.abs(ssum) + half_g) >> lg)
    else:
        Hb = np.broadcast_to(np.asarray(H, np.float64), xf.shape).reshape(B, ng, g_size)
        m = np.rint((Hb * resg).sum(2) / Hb.sum(2))                                    # weighted round-mean
    l_g_word = np.clip(m, s_min, s_max).astype(np.int64)
    return s_base, up.astype(np.int64), l_g_word


def reconstruct(scale: np.ndarray, h_word: np.ndarray, l_g_word: np.ndarray, l_b: int,
                g_size: int) -> np.ndarray:
    """Inverse of decompose -> float blocks (B,32)."""
    r_exp = np.repeat(l_g_word, g_size, axis=1)                         # (B,32)
    return (h_word * (1 << l_b) + r_exp).astype(np.float64) * np.asarray(scale)


def per_block_bits(l_b: int, g_size: int) -> int:
    """Bits to store one 32-block: E8M0 scale + per-element upper + per-group shared."""
    n_group = BLOCK // g_size
    return 8 + (8 - l_b) * BLOCK + l_b * n_group


def per_elem_bits(l_b: int, g_size: int) -> float:
    return per_block_bits(l_b, g_size) / BLOCK


# =============================================================================
#  PACKING  (out-innermost SoA, dense LSB-first)
# =============================================================================
def _pack_codes_lsb(codes: np.ndarray, width: int) -> np.ndarray:
    """codes int[..., n] (two's-complement width-bit) -> uint8[..., ceil(n*width/8)]."""
    n = codes.shape[-1]
    c = (codes & ((width and (1 << width) - 1) or 0)).astype(np.uint64)
    bit = ((c[..., :, None] >> np.arange(width, dtype=np.uint64)) & 1).astype(np.uint8)
    bit = bit.reshape(*codes.shape[:-1], n * width)
    return np.packbits(bit, axis=-1, bitorder="little")


def _unpack_codes_lsb(buf: np.ndarray, n: int, width: int) -> np.ndarray:
    """Inverse of _pack_codes_lsb -> signed int[..., n]."""
    bits = np.unpackbits(buf, axis=-1, bitorder="little")
    bits = bits[..., : n * width].reshape(*buf.shape[:-1], n, width)
    shifts = np.arange(width, dtype=np.int64)
    vals = (bits.astype(np.int64) << shifts).sum(axis=-1)
    half = 1 << (width - 1)
    return np.where(vals >= half, vals - (1 << width), vals)


def pack_weight(W: np.ndarray, l_b: int, g_size: int) -> dict:
    """W float[OUT, K] (K%32==0) -> dict of out-innermost packed planes + meta.

    Reused for KV: pack one head's K or V as [Lk, D] (OUT=Lk, K=D) so the 32-blocks
    run along head_dim and the token axis is innermost (coalesced)."""
    OUT, K = W.shape
    assert K % BLOCK == 0, "K (in_features / head_dim) must be a multiple of 32"
    nb = K // BLOCK
    wbits = 8 - l_b
    n_group = BLOCK // g_size

    blocks = W.reshape(OUT, nb, BLOCK).reshape(OUT * nb, BLOCK)
    scale, h_word, l_g_word = decompose(blocks, l_b, g_size)
    exp = np.rint(np.log2(scale.reshape(-1))).astype(np.int64).reshape(OUT, nb)
    qu = h_word.reshape(OUT, nb, BLOCK)
    rs = l_g_word.reshape(OUT, nb, n_group)

    UB = (BLOCK * wbits) // 8
    SB = math.ceil(n_group * l_b / 8)
    up = _pack_codes_lsb(qu, wbits)                # [OUT, nb, UB]
    sh = _pack_codes_lsb(rs, l_b)                    # [OUT, nb, SB]

    d = dict(
        scale_exp=np.ascontiguousarray(exp.T.astype(np.int8)),     # [nb, OUT]
        upper=np.ascontiguousarray(up.transpose(1, 2, 0)),         # [nb, UB, OUT]
        shared=np.ascontiguousarray(sh.transpose(1, 2, 0)),        # [nb, SB, OUT]
        OUT=OUT, K=K, nb=nb, l_b=l_b, g_size=g_size, wbits=wbits,
        UB=UB, SB=SB, n_group=n_group,
    )
    # Wide-load GEMV plane: COLUMN-MAJOR [nb, OUT, UB] so a thread reads its
    # column's whole UB-byte block in ONE (u4: int4) or a few (u2/u3: 4-aligned
    # uint32) wide, coalesced loads instead of UB byte-strided reads. Just a
    # transpose of the dense plane (no re-packing); bytes/codes identical ->
    # oracle unaffected. Built for all u (u4 = int4 width; u2/u3 word-loaded).
    d["upper_cm"] = np.ascontiguousarray(d["upper"].transpose(0, 2, 1))   # [nb, OUT, UB]
    d["shared_cm"] = np.ascontiguousarray(d["shared"].transpose(0, 2, 1)) # [nb, OUT, SB]
    return d
def pack_weight_ra(W: np.ndarray, l_b: int, g_size: int) -> dict:
    """REGISTER-ALIGNED pack: SAME codes as pack_weight, upper field padded so each (8-u)-bit
    code lies wholly inside one 32-bit word (straddle-free single-bfe extract); +~20% upper
    bytes buys ~4->1 ALU ops/elem. Decode == pack_weight."""
    OUT, K = W.shape
    assert K % BLOCK == 0
    nb = K // BLOCK; wbits = 8 - l_b; n_group = BLOCK // g_size
    CPW = 32 // wbits; NW = (BLOCK + CPW - 1) // CPW
    blocks = W.reshape(OUT, nb, BLOCK).reshape(OUT * nb, BLOCK)
    scale, h_word, l_g_word = decompose(blocks, l_b, g_size)
    exp = np.rint(np.log2(scale.reshape(-1))).astype(np.int64).reshape(OUT, nb)
    qu = h_word.reshape(OUT, nb, BLOCK); rs = l_g_word.reshape(OUT, nb, n_group)
    # register-aligned upper: words[..,wi] |= (code & mask) << ((k%CPW)*wbits)
    mask = (1 << wbits) - 1
    words = np.zeros((OUT, nb, NW), dtype=np.uint32)
    for k in range(BLOCK):
        wi = k // CPW; bp = (k % CPW) * wbits
        words[..., wi] |= ((qu[..., k].astype(np.uint64) & mask).astype(np.uint32) << bp)
    upper_ra = words.view(np.uint8).reshape(OUT, nb, NW * 4)        # little-endian bytes
    sh = _pack_codes_lsb(rs, l_b)                                     # [OUT,nb,SB] dense (1 byte)
    SB = sh.shape[-1]
    return dict(
        scale_exp=np.ascontiguousarray(exp.T.astype(np.int8)),     # [nb, OUT]
        upper_ra_cm=np.ascontiguousarray(upper_ra.transpose(1, 0, 2)),  # [nb, OUT, NW*4]
        shared_cm=np.ascontiguousarray(sh.transpose(1, 0, 2)),     # [nb, OUT, SB]
        OUT=OUT, K=K, nb=nb, l_b=l_b, g_size=g_size, wbits=wbits, CPW=CPW, NW=NW, SB=SB, n_group=n_group,
    )


def dequant_weight_ra(p: dict) -> np.ndarray:
    """register-aligned packed -> dense float [OUT,K] (oracle)."""
    OUT, nb, l_b, g_size, wbits, CPW, NW, n_group = (p["OUT"], p["nb"], p["l_b"], p["g_size"],
                                               p["wbits"], p["CPW"], p["NW"], p["n_group"])
    exp = p["scale_exp"].T.astype(np.int64)                        # [OUT,nb]
    words = np.ascontiguousarray(p["upper_ra_cm"].transpose(1, 0, 2)).reshape(OUT, nb, NW, 4).view(np.uint32).reshape(OUT, nb, NW)
    mask = (1 << wbits) - 1; half = 1 << (wbits - 1)
    h_word = np.empty((OUT, nb, BLOCK), np.int64)
    for k in range(BLOCK):
        wi = k // CPW; bp = (k % CPW) * wbits
        v = ((words[..., wi] >> bp) & mask).astype(np.int64)
        h_word[..., k] = np.where(v >= half, v - (1 << wbits), v)   # sign-extend
    sh = np.ascontiguousarray(p["shared_cm"].transpose(1, 0, 2))    # [OUT,nb,SB]
    l_g_word = _unpack_codes_lsb(sh, n_group, l_b)                    # signed
    r_exp = np.repeat(l_g_word, g_size, axis=2)
    qfull = h_word * (1 << l_b) + r_exp
    return (qfull.astype(np.float64) * (2.0 ** exp)[:, :, None]).reshape(OUT, p["K"])


def decompose_naive(x_blocked: np.ndarray, l_b: int,
                    g_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NAIVE mantissa-sharing: quantize to MXINT8 int8, split into per-element high part +
    group-AVERAGED unsigned low u bits (simplest sharing, for bottleneck isolation; same stored
    plane layout as SSNF, unsigned shared -> reuses the SSNF-unsigned unpack)."""
    xf = np.asarray(x_blocked, dtype=np.float64)
    B = xf.shape[0]; n_group = BLOCK // g_size
    s_base = _e8m0_scale(np.abs(xf).max(axis=1, keepdims=True))
    q8 = np.clip(np.round(xf / s_base), -127, 127).astype(np.int64)
    upper = q8 >> l_b                                   # arithmetic shift (Python int >> is arithmetic)
    low = q8 - (upper << l_b)                            # == q8 & (2^l_b-1), unsigned [0,2^l_b)
    low_avg = low.reshape(B, n_group, g_size).mean(axis=2)
    l_g_word = np.clip(np.round(low_avg), 0, (1 << l_b) - 1).astype(np.int64)
    return s_base, upper, l_g_word


def pack_weight_naive(W: np.ndarray, l_b: int, g_size: int) -> dict:
    """naive SSNF-unsigned pack: SAME plane layout/keys as pack_weight_unsigned (upper_cm/
    shared_cm/scale_exp, shared UNSIGNED). Read by the SSNF-unsigned unpack kernels."""
    OUT, K = W.shape
    assert K % BLOCK == 0
    nb = K // BLOCK; wbits = 8 - l_b; n_group = BLOCK // g_size
    blocks = W.reshape(OUT, nb, BLOCK).reshape(OUT * nb, BLOCK)
    scale, upper, l_g_word = decompose_naive(blocks, l_b, g_size)
    exp = np.rint(np.log2(scale.reshape(-1))).astype(np.int64).reshape(OUT, nb)
    qu = upper.reshape(OUT, nb, BLOCK); rs = l_g_word.reshape(OUT, nb, n_group)
    UB = (BLOCK * wbits) // 8; SB = math.ceil(n_group * l_b / 8)
    up = _pack_codes_lsb(qu, wbits); sh = _pack_codes_lsb(rs, l_b)
    d = dict(
        scale_exp=np.ascontiguousarray(exp.T.astype(np.int8)),
        upper=np.ascontiguousarray(up.transpose(1, 2, 0)),
        shared=np.ascontiguousarray(sh.transpose(1, 2, 0)),
        OUT=OUT, K=K, nb=nb, l_b=l_b, g_size=g_size, wbits=wbits, UB=UB, SB=SB, n_group=n_group,
    )
    d["upper_cm"] = np.ascontiguousarray(d["upper"].transpose(0, 2, 1))
    d["shared_cm"] = np.ascontiguousarray(d["shared"].transpose(0, 2, 1))
    return d


def dequant_weight_naive(p: dict) -> np.ndarray:
    """naive SSNF-unsigned packed -> dense float [OUT,K] (oracle). Identical decode to UNSIGNED."""
    return dequant_weight_unsigned(p)


def decompose_unsigned(x_blocked: np.ndarray, l_b: int,
                       g_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SSNF-unsigned variant: FLOOR the upper code (residual >= 0) so the shared code is UNSIGNED
    and reconstruction is a pure bit-concat (h_word<<l_b | l_g_word); same E8M0 scale & sharing."""
    xf = np.asarray(x_blocked, dtype=np.float64)
    B = xf.shape[0]; n_group = BLOCK // g_size
    s_base = _e8m0_scale(np.abs(xf).max(axis=1, keepdims=True))
    s_unshared = s_base * (1 << l_b)
    q_lo, q_hi = -(1 << (7 - l_b)), (1 << (7 - l_b)) - 1
    h_word = np.clip(np.floor(xf / s_unshared), q_lo, q_hi).astype(np.int64)   # FLOOR -> residual>=0
    residual = xf - h_word * s_unshared                                        # in [0, s_unshared)
    res_avg = residual.reshape(B, n_group, g_size).mean(axis=2)
    l_g_word = np.clip(np.round(res_avg / s_base), 0, (1 << l_b) - 1).astype(np.int64)  # UNSIGNED l_b-bit
    return s_base, h_word, l_g_word


def pack_weight_unsigned(W: np.ndarray, l_b: int, g_size: int) -> dict:
    """SSNF-unsigned pack: SAME plane layout/keys as pack_weight but UNSIGNED shared codes
    (decompose_unsigned); dequant_weight_unsigned is the oracle."""
    OUT, K = W.shape
    assert K % BLOCK == 0
    nb = K // BLOCK; wbits = 8 - l_b; n_group = BLOCK // g_size
    blocks = W.reshape(OUT, nb, BLOCK).reshape(OUT * nb, BLOCK)
    scale, h_word, l_g_word = decompose_unsigned(blocks, l_b, g_size)
    exp = np.rint(np.log2(scale.reshape(-1))).astype(np.int64).reshape(OUT, nb)
    qu = h_word.reshape(OUT, nb, BLOCK); rs = l_g_word.reshape(OUT, nb, n_group)
    UB = (BLOCK * wbits) // 8; SB = math.ceil(n_group * l_b / 8)
    up = _pack_codes_lsb(qu, wbits); sh = _pack_codes_lsb(rs, l_b)   # rs unsigned -> mask keeps bits
    d = dict(
        scale_exp=np.ascontiguousarray(exp.T.astype(np.int8)),
        upper=np.ascontiguousarray(up.transpose(1, 2, 0)),
        shared=np.ascontiguousarray(sh.transpose(1, 2, 0)),
        OUT=OUT, K=K, nb=nb, l_b=l_b, g_size=g_size, wbits=wbits, UB=UB, SB=SB, n_group=n_group,
    )
    d["upper_cm"] = np.ascontiguousarray(d["upper"].transpose(0, 2, 1))
    d["shared_cm"] = np.ascontiguousarray(d["shared"].transpose(0, 2, 1))
    return d


def dequant_weight_unsigned(p: dict) -> np.ndarray:
    """SSNF-unsigned packed -> dense float [OUT,K] (oracle). shared is UNSIGNED."""
    OUT, nb, l_b, g_size, wbits, n_group = (p["OUT"], p["nb"], p["l_b"], p["g_size"], p["wbits"], p["n_group"])
    exp = p["scale_exp"].T.astype(np.int64)
    up = np.ascontiguousarray(p["upper"].transpose(2, 0, 1))
    sh = np.ascontiguousarray(p["shared"].transpose(2, 0, 1))
    h_word = _unpack_codes_lsb(up, BLOCK, wbits)                  # signed (8-u)-bit
    # shared UNSIGNED: unpack bits without sign extension
    bits = np.unpackbits(sh, axis=-1, bitorder="little")[..., : n_group * l_b].reshape(*sh.shape[:-1], n_group, l_b)
    l_g_word = (bits.astype(np.int64) << np.arange(l_b, dtype=np.int64)).sum(axis=-1)   # [OUT,nb,ng], 0..2^l_b-1
    r_exp = np.repeat(l_g_word, g_size, axis=2)
    qfull = h_word * (1 << l_b) + r_exp
    return (qfull.astype(np.float64) * (2.0 ** exp)[:, :, None]).reshape(OUT, p["K"])


def pack_weight_relayout(W: np.ndarray, l_b: int, g_size: int) -> dict:
    """NIBBLE-ALIGNED re-layout of pack_weight (SAME weights, bit-exact, byte-neutral):
    upper split into a SIMD-friendly high nibble + dense unsigned low part, shared unchanged;
    reconstruct w = upper4*16 + low_un*2^l_b + shared. Planes column-major [nb, OUT, .]."""
    OUT, K = W.shape
    assert K % BLOCK == 0
    nb = K // BLOCK; n_group = BLOCK // g_size; lu_bits = 4 - l_b
    blocks = W.reshape(OUT, nb, BLOCK).reshape(OUT * nb, BLOCK)
    scale, h_word, l_g_word = decompose(blocks, l_b, g_size)
    exp = np.rint(np.log2(scale.reshape(-1))).astype(np.int64).reshape(OUT, nb)
    qu = h_word.reshape(OUT, nb, BLOCK)
    upper4 = (qu >> lu_bits).astype(np.int64)                      # signed 4-bit
    low_un = (qu & ((1 << lu_bits) - 1)).astype(np.int64)          # unsigned (4-u)-bit
    rs = l_g_word.reshape(OUT, nb, n_group)
    # high-nibble plane: store BIASED (upper4+8 in [0,15]) so the SIMD dequant gets
    # the signed value via a uniform (nibble-8) with no per-nibble two's-complement
    # branch. 32 nibbles -> 16 bytes (k even=low nibble, odd=high).
    hu = ((upper4 + 8) & 0xF).astype(np.uint8)                     # [OUT,nb,32]
    hi4 = (hu[..., 0::2] | (hu[..., 1::2] << 4)).astype(np.uint8)  # [OUT,nb,16]
    lu = _pack_codes_lsb(low_un, lu_bits)                          # [OUT,nb, ceil(32*lu_bits/8)]
    sh = _pack_codes_lsb(rs, l_b)                                    # [OUT,nb,SB]
    return dict(
        scale_exp=np.ascontiguousarray(exp.T.astype(np.int8)),     # [nb, OUT]
        hi4_cm=np.ascontiguousarray(hi4.transpose(1, 0, 2)),       # [nb, OUT, 16]
        lowun_cm=np.ascontiguousarray(lu.transpose(1, 0, 2)),      # [nb, OUT, lu_B]
        shared_cm=np.ascontiguousarray(sh.transpose(1, 0, 2)),     # [nb, OUT, SB]
        OUT=OUT, K=K, nb=nb, l_b=l_b, g_size=g_size, lu_bits=lu_bits, n_group=n_group,
        HB=16, LUB=lu.shape[-1], SB=sh.shape[-1],
    )


def dequant_weight_relayout(p: dict) -> np.ndarray:
    """Re-layout planes -> dense float [OUT,K]; oracle for the CUDA nibble kernel."""
    OUT, nb, l_b, g_size, lu_bits, n_group = (p["OUT"], p["nb"], p["l_b"], p["g_size"],
                                        p["lu_bits"], p["n_group"])
    exp = p["scale_exp"].T.astype(np.int64)                        # [OUT,nb]
    hi4 = np.ascontiguousarray(p["hi4_cm"].transpose(1, 0, 2))     # [OUT,nb,16]
    lo = np.ascontiguousarray(p["lowun_cm"].transpose(1, 0, 2))    # [OUT,nb,LUB]
    sh = np.ascontiguousarray(p["shared_cm"].transpose(1, 0, 2))   # [OUT,nb,SB]
    # unpack high nibbles (signed 4-bit)
    lown = (hi4 & 0xF).astype(np.int64); highn = (hi4 >> 4).astype(np.int64)
    upper4 = np.empty((OUT, nb, BLOCK), np.int64)
    upper4[..., 0::2] = lown; upper4[..., 1::2] = highn
    upper4 = upper4 - 8                                            # un-bias (stored as upper4+8)
    low_un = _unpack_codes_lsb(lo, BLOCK, lu_bits) if lu_bits else np.zeros((OUT, nb, BLOCK), np.int64)
    low_un &= (1 << lu_bits) - 1                                   # unsigned
    shared = _unpack_codes_lsb(sh, n_group, l_b)                    # [OUT,nb,n_group]
    sh_exp = np.repeat(shared, g_size, axis=2)
    w_int = upper4 * 16 + low_un * (1 << l_b) + sh_exp
    return (w_int.astype(np.float64) * (2.0 ** exp)[:, :, None]).reshape(OUT, p["K"])


def unpack_weight(p: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Packed planes -> (scale [OUT,nb], q_upper [OUT,nb,32], r_shared [OUT,nb,n_group])."""
    OUT, nb, l_b, g_size, wbits, n_group = (p["OUT"], p["nb"], p["l_b"], p["g_size"],
                                      p["wbits"], p["n_group"])
    exp = p["scale_exp"].T.astype(np.int64)                        # [OUT, nb]
    up = np.ascontiguousarray(p["upper"].transpose(2, 0, 1))       # [OUT, nb, UB]
    sh = np.ascontiguousarray(p["shared"].transpose(2, 0, 1))      # [OUT, nb, SB]
    h_word = _unpack_codes_lsb(up, BLOCK, wbits)                  # [OUT, nb, 32]
    l_g_word = _unpack_codes_lsb(sh, n_group, l_b)                  # [OUT, nb, n_group]
    scale = (2.0 ** exp)                                           # [OUT, nb]
    return scale, h_word, l_g_word


def dequant_weight(p: dict) -> np.ndarray:
    """Packed -> dense float weight [OUT, K]  (W-only reconstruction)."""
    scale, h_word, l_g_word = unpack_weight(p)
    g_size, l_b = p["g_size"], p["l_b"]
    r_exp = np.repeat(l_g_word, g_size, axis=2)                        # [OUT,nb,32]
    qfull = h_word * (1 << l_b) + r_exp                            # [OUT,nb,32]
    return (qfull.astype(np.float64) * scale[:, :, None]).reshape(p["OUT"], p["K"])


def dequant_weight_nores(p: dict) -> np.ndarray:
    """No-residual packed (pack_weight_nores) -> dense float [OUT,K]. word = q_upper<<u."""
    exp = p["scale_exp"].T.astype(np.int64)                        # [OUT, nb]
    up = np.ascontiguousarray(p["upper"].transpose(2, 0, 1))       # [OUT, nb, UB]
    h_word = _unpack_codes_lsb(up, BLOCK, p["wbits"])             # [OUT, nb, 32]
    word = h_word.astype(np.int64) << p["l_b"]
    return (word.astype(np.float64) * (2.0 ** exp)[:, :, None]).reshape(p["OUT"], p["K"])


def weight_int8(p: dict) -> tuple[np.ndarray, np.ndarray]:
    """Packed -> (qW int [OUT,nb,32], scale_w [OUT,nb]) (W+A reconstruction);
    qW = h_word*2^l_b + shared_expanded is the reusable MXINT8 integer word."""
    scale, h_word, l_g_word = unpack_weight(p)
    g_size, l_b = p["g_size"], p["l_b"]
    r_exp = np.repeat(l_g_word, g_size, axis=2)
    qfull = h_word * (1 << l_b) + r_exp                            # [OUT,nb,32], int8 range
    return qfull, scale


def pack_kv(KV: np.ndarray, l_b: int, g_size: int) -> dict:
    """KV float[H, L, D] (D%32==0) -> stacked head-major planes for the attn kernel; per head =
    pack_weight on [L, D], byte planes TOKEN-MAJOR (bytes innermost) for coalesced loads.
    (Register-aligned KV packing was tried and REVERTED: +26% bytes, no win — KV decode is
    load-latency-bound, not extraction-bound.)"""
    H, L, D = KV.shape
    per = [pack_weight(KV[h], l_b, g_size) for h in range(H)]
    # per-head upper/shared are [nb,UB,L]/[nb,SB,L]; move BYTES innermost -> [nb,L,UB]
    up = np.stack([q["upper"] for q in per]).transpose(0, 1, 3, 2)             # [H,nb,L,UB]
    sh = np.stack([q["shared"] for q in per]).transpose(0, 1, 3, 2)            # [H,nb,L,SB]
    return dict(
        scale_exp=np.ascontiguousarray(np.stack([q["scale_exp"] for q in per])),  # [H,nb,L]
        upper=np.ascontiguousarray(up),                                        # [H,nb,L,UB]
        shared=np.ascontiguousarray(sh),                                       # [H,nb,L,SB]
        H=H, L=L, D=D, nb=per[0]["nb"], l_b=l_b, g_size=g_size, wbits=per[0]["wbits"],
        UB=per[0]["UB"], SB=per[0]["SB"], n_group=per[0]["n_group"],
        _per=per,
    )


def pack_weight_nores(W: np.ndarray, l_b: int) -> dict:
    """SSNF upper-only pack (shared residual DROPPED); recon = (h_word << l_b) * scale. Under
    block-Had rotation this equals plain MXINT(8-l_b) staged to int8 and is iso-bit >= SSNF
    (measured). Reused for KV per head like pack_weight."""
    OUT, K = W.shape
    assert K % BLOCK == 0, "K (in_features / head_dim) must be a multiple of 32"
    nb = K // BLOCK; wbits = 8 - l_b
    blocks = W.reshape(OUT, nb, BLOCK).reshape(OUT * nb, BLOCK)
    scale, h_word, _ = decompose(blocks, l_b, BLOCK)              # gs irrelevant (shared dropped)
    exp = np.rint(np.log2(scale.reshape(-1))).astype(np.int64).reshape(OUT, nb)
    qu = h_word.reshape(OUT, nb, BLOCK)
    UB = (BLOCK * wbits) // 8
    up = _pack_codes_lsb(qu, wbits)                             # [OUT, nb, UB]
    d = dict(
        scale_exp=np.ascontiguousarray(exp.T.astype(np.int8)),  # [nb, OUT]
        upper=np.ascontiguousarray(up.transpose(1, 2, 0)),      # [nb, UB, OUT]
        OUT=OUT, K=K, nb=nb, l_b=l_b, g_size=BLOCK, wbits=wbits, UB=UB, SB=0, n_group=0,
    )
    d["upper_cm"] = np.ascontiguousarray(d["upper"].transpose(0, 2, 1))  # [nb, OUT, UB]
    return d


def reconstruct_nores(scale: np.ndarray, h_word: np.ndarray, l_b: int) -> np.ndarray:
    """Inverse of pack_weight_nores' decompose -> float blocks (B,32). int8 word = q_upper<<u."""
    return (h_word.astype(np.int64) << l_b).astype(np.float64) * np.asarray(scale)


def pack_kv_nores(KV: np.ndarray, l_b: int) -> dict:
    """KV float[H,L,D] -> token-major upper-only planes (no shared), mirror of pack_kv.
    Per head = pack_weight_nores on [L,D]; BYTES innermost ([H,nb,L,UB]) for coalesced reads."""
    H, L, D = KV.shape
    per = [pack_weight_nores(KV[h], l_b) for h in range(H)]
    up = np.stack([q["upper"] for q in per]).transpose(0, 1, 3, 2)             # [H,nb,L,UB]
    return dict(
        scale_exp=np.ascontiguousarray(np.stack([q["scale_exp"] for q in per])),  # [H,nb,L]
        upper=np.ascontiguousarray(up),                                        # [H,nb,L,UB]
        H=H, L=L, D=D, nb=per[0]["nb"], l_b=l_b, g_size=BLOCK, wbits=per[0]["wbits"],
        UB=per[0]["UB"], SB=0, n_group=0, _per=per,
    )
def pack_kv_relayout(KV: np.ndarray, l_b: int, g_size: int) -> dict:
    """KV float[H,L,D] -> nibble-relayout token-major planes (BYTES innermost) for the
    u2/u3 relayout KV-decode kernel. Per head = pack_weight_relayout on [L,D]; the _cm
    planes are already [nb, L, BYTES] (token-major) so just stack over H."""
    H, L, D = KV.shape
    per = [pack_weight_relayout(KV[h], l_b, g_size) for h in range(H)]
    return dict(
        scale_exp=np.ascontiguousarray(np.stack([q["scale_exp"] for q in per])),  # [H,nb,L]
        hi4=np.ascontiguousarray(np.stack([q["hi4_cm"] for q in per])),            # [H,nb,L,16]
        lowun=np.ascontiguousarray(np.stack([q["lowun_cm"] for q in per])),        # [H,nb,L,LUB]
        shared=np.ascontiguousarray(np.stack([q["shared_cm"] for q in per])),      # [H,nb,L,SB]
        H=H, L=L, D=D, nb=per[0]["nb"], l_b=l_b, g_size=g_size, lu_bits=per[0]["lu_bits"],
        HB=16, LUB=per[0]["LUB"], SB=per[0]["SB"], n_group=per[0]["n_group"],
    )


def effective_bits_per_elem(p: dict) -> float:
    total = p["scale_exp"].size + p["upper"].size + p["shared"].size
    denom = (p["OUT"] * p["K"]) if "OUT" in p else (p["H"] * p["L"] * p["D"])
    return total * 8 / denom


# =============================================================================
#  PLAIN MXINT8  (baseline format: int8 mantissa + E8M0 scale, NO sharing)
# =============================================================================
#  The matched-optimization baseline for the kernels. Same out-innermost SoA
#  plane layout as the SSNF pack, but the mantissa is a full int8 stored
#  directly — so the MXINT8 kernels are byte-for-byte identical to the SSNF
#  kernels EXCEPT they skip the sub-byte unpack (a direct int8 load replaces
#  ssnf::unpack_ssnf_weight_elem). This isolates the unpack overhead. Per element:
#  8 bits + 8/32 scale = 8.25 b/elem (vs e.g. SSNF u3gs8 = 5.625).
def _quant_mxint8_blocks(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """float blocks (B,32) -> (scale (B,1), q (B,32) int)."""
    xf = np.asarray(blocks, np.float64)
    s = _e8m0_scale(np.abs(xf).max(axis=1, keepdims=True))
    q = np.clip(np.round(xf / s), -127, 127).astype(np.int64)
    return s, q


def pack_weight_mxint8(W: np.ndarray) -> dict:
    """W float[OUT,K] (K%32==0) -> out-innermost MXINT8 planes:
       scale_exp [nb,OUT] int8,  qweight [nb,32,OUT] int8."""
    OUT, K = W.shape
    assert K % BLOCK == 0, "K must be a multiple of 32"
    nb = K // BLOCK
    blocks = W.reshape(OUT, nb, BLOCK).reshape(OUT * nb, BLOCK)
    s, q = _quant_mxint8_blocks(blocks)
    exp = np.rint(np.log2(s.reshape(-1))).astype(np.int64).reshape(OUT, nb)
    q = q.reshape(OUT, nb, BLOCK)
    qw = np.ascontiguousarray(q.transpose(1, 2, 0).astype(np.int8))      # [nb, 32, OUT]
    return dict(
        scale_exp=np.ascontiguousarray(exp.T.astype(np.int8)),          # [nb, OUT]
        qweight=qw,                                                     # [nb, 32, OUT] row-major
        # column-major twin [nb, OUT, 32]: a column's 32 int8 bytes are CONTIGUOUS, so the
        # decode/GEMM kernels wide-load them (1 int4x2) instead of 32 strided 1-byte reads.
        qweight_cm=np.ascontiguousarray(q.transpose(1, 0, 2).astype(np.int8)),  # [nb, OUT, 32]
        OUT=OUT, K=K, nb=nb,
    )


def dequant_weight_mxint8(p: dict) -> np.ndarray:
    """MXINT8 packed -> dense float [OUT,K]."""
    exp = p["scale_exp"].T.astype(np.int64)                              # [OUT, nb]
    q = np.ascontiguousarray(p["qweight"].transpose(2, 0, 1)).astype(np.float64)  # [OUT,nb,32]
    return (q * (2.0 ** exp)[:, :, None]).reshape(p["OUT"], p["K"])


def weight_int8_mxint8(p: dict) -> tuple[np.ndarray, np.ndarray]:
    """MXINT8 packed -> (qW int [OUT,nb,32], scale_w [OUT,nb])  (W+A path)."""
    exp = p["scale_exp"].T.astype(np.int64)                              # [OUT, nb]
    qW = np.ascontiguousarray(p["qweight"].transpose(2, 0, 1)).astype(np.int64)   # [OUT,nb,32]
    return qW, (2.0 ** exp)


def _e8m0_exp_from_amax(amax: np.ndarray, emax_elem: int) -> np.ndarray:
    """E8M0 block exponent so the element format's max maps near the block amax."""
    e = np.floor(np.log2(np.maximum(amax, 1e-30))).astype(np.int32) - emax_elem
    return np.clip(e, -127, 127)


def _e2m3_pos_grid() -> np.ndarray:
    """32 non-negative E2M3 levels (sign=0), monotonic increasing; index = e*8+m."""
    g = []
    for e in range(4):
        for m in range(8):
            g.append(m * 0.125 if e == 0 else (1.0 + m * 0.125) * (2.0 ** (e - 1)))
    return np.asarray(g, dtype=np.float64)


def pack_kv_mxfp8(KV: np.ndarray) -> dict:
    """KV float[H,L,D] -> native MXFP8-E4M3 planes (mirror of pack_kv_mxint8 layout).
    scale_exp int8[H,nb,L] (E8M0, emax=8) + qcodes uint8[H,nb,L,32] (E4M3 byte/elem). 8.25 b/elem."""
    import torch
    H, L, D = KV.shape
    nb = D // 32
    x = np.ascontiguousarray(KV).reshape(H, L, nb, 32)
    se = _e8m0_exp_from_amax(np.abs(x).max(-1), 8)                 # [H,L,nb]
    # saturate to E4M3 max (448); torch casts out-of-range to NaN (0x7F), not saturate.
    xn = np.clip(x / (2.0 ** se)[..., None], -448.0, 448.0).astype(np.float32)
    codes = torch.from_numpy(xn).to(torch.float8_e4m3fn).view(torch.uint8).numpy()  # [H,L,nb,32]
    return dict(
        scale_exp=np.ascontiguousarray(se.transpose(0, 2, 1).astype(np.int8)),      # [H,nb,L]
        qcodes=np.ascontiguousarray(codes.transpose(0, 2, 1, 3)),                    # [H,nb,L,32]
        H=H, L=L, D=D, nb=nb,
    )


def pack_kv_mxfp6(KV: np.ndarray) -> dict:
    """KV float[H,L,D] -> native MXFP6-E2M3 planes. scale_exp int8[H,nb,L] (E8M0, emax=2) +
    qcodes uint8[H,nb,L,24] (32x 6-bit codes packed 4-into-3 bytes). 6.25 b/elem."""
    H, L, D = KV.shape
    nb = D // 32
    x = np.ascontiguousarray(KV).reshape(H, L, nb, 32)
    se = _e8m0_exp_from_amax(np.abs(x).max(-1), 2)                 # [H,L,nb]
    xn = x / (2.0 ** se)[..., None]
    grid = _e2m3_pos_grid()
    a = np.abs(xn)
    idx = np.clip(np.searchsorted(grid, a), 0, 31)
    lo = np.clip(idx - 1, 0, 31)
    pick_lo = np.abs(a - grid[lo]) <= np.abs(a - grid[idx])
    mag = np.where(pick_lo, lo, idx).astype(np.uint32)            # 0..31 = e*8+m
    codes = ((xn < 0).astype(np.uint32) << 5) | mag               # 6-bit [H,L,nb,32]
    c = codes.reshape(H, L, nb, 8, 4)
    b0 = (c[..., 0] | (c[..., 1] << 6)) & 0xFF
    b1 = ((c[..., 1] >> 2) | (c[..., 2] << 4)) & 0xFF
    b2 = ((c[..., 2] >> 4) | (c[..., 3] << 2)) & 0xFF
    packed = np.stack([b0, b1, b2], -1).reshape(H, L, nb, 24).astype(np.uint8)       # [H,L,nb,24]
    return dict(
        scale_exp=np.ascontiguousarray(se.transpose(0, 2, 1).astype(np.int8)),      # [H,nb,L]
        qcodes=np.ascontiguousarray(packed.transpose(0, 2, 1, 3)),                   # [H,nb,L,24]
        H=H, L=L, D=D, nb=nb,
    )


def pack_kv_mxint8(KV: np.ndarray) -> dict:
    """KV float[H,L,D] -> stacked MXINT8 head-major planes (mirror of pack_kv)."""
    H, L, D = KV.shape
    per = [pack_weight_mxint8(KV[h]) for h in range(H)]
    # token-major (Stage 4a): mantissa axis innermost -> [H,nb,L,32] for coalescing
    qw = np.stack([q["qweight"] for q in per]).transpose(0, 1, 3, 2)           # [H,nb,L,32]
    return dict(
        scale_exp=np.ascontiguousarray(np.stack([q["scale_exp"] for q in per])),  # [H,nb,L]
        qweight=np.ascontiguousarray(qw),                                      # [H,nb,L,32]
        H=H, L=L, D=D, nb=per[0]["nb"], _per=per,
    )

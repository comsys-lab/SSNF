"""Chunked-prefill helper (pure torch, no vLLM dep so it is unit-testable in the accuracy environment).

Synthesizes per-token (causal seq_len, block_table row) so kv_decode_paged reads past
quantized blocks + the current chunk from the paged cache, and dequantizes past K/V per format.
"""
import torch


def synth_causal(pf_qsl: torch.Tensor, pf_seq_lens: torch.Tensor, pf_bt: torch.Tensor,
                 Npf: int) -> tuple[torch.Tensor, torch.Tensor]:
    """pf_qsl [npf+1] (chunk cu-seqlens, starts at 0), pf_seq_lens [npf] (full ctx incl chunk),
    pf_bt [npf, max_blocks]. -> (synth_bt [Npf, max_blocks], synth_L [Npf] int32)."""
    dev = pf_seq_lens.device
    qlen = (pf_qsl[1:] - pf_qsl[:-1]).to(torch.int64)          # chunk len per seq
    ctx = pf_seq_lens.to(torch.int64) - qlen                   # past (cached) len per seq
    npf = pf_seq_lens.shape[0]
    seq_idx = torch.repeat_interleave(torch.arange(npf, device=dev), qlen)   # [Npf] seq of each token
    local = torch.arange(Npf, device=dev) - pf_qsl[:-1].to(torch.int64)[seq_idx]  # local pos in chunk
    synth_L = (ctx[seq_idx] + local + 1).to(torch.int32)       # causal length per token
    synth_bt = pf_bt[seq_idx]                                  # [Npf, max_blocks]
    return synth_bt, synth_L


def dequant_past(cache: torch.Tensor, bt_row: torch.Tensor, ctx: int, nb: int, UB: int, SB: int,
                 per_head: int, l_b: int, g_size: int, Hk: int, D: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct one sequence's PAST quantized K/V from the paged cache -> (K,V) bf16 [Hk,ctx,D]
    (reuses the validated wonly_dequant_bf16; KV per-head planes == weight-plane layout)."""
    bs = cache.shape[1]
    npb = (ctx + bs - 1) // bs
    phys = cache[bt_row[:npb].long()]                          # [npb, bs, Hk, 2*ph]
    flat = phys.reshape(npb * bs, Hk, cache.shape[-1])[:ctx]   # [ctx, Hk, 2*ph]

    def deq(blob: torch.Tensor) -> torch.Tensor:               # blob [ctx, Hk, per_head]
        blk = blob.reshape(ctx, Hk, nb, 1 + UB + SB)
        sc = blk[..., 0].permute(1, 2, 0).contiguous().view(torch.int8)   # [Hk, nb, ctx]
        up = blk[..., 1:1 + UB].permute(1, 2, 0, 3).contiguous()          # [Hk, nb, ctx, UB]
        sh = blk[..., 1 + UB:].permute(1, 2, 0, 3).contiguous()           # [Hk, nb, ctx, SB]
        outs = [torch.ops.ssnf.wonly_dequant_bf16(sc[h], up[h], sh[h], ctx, D, nb, l_b, g_size)
                for h in range(Hk)]                                       # each [ctx, D]
        return torch.stack(outs)                                         # [Hk, ctx, D]
    return deq(flat[..., :per_head]), deq(flat[..., per_head:])


def dequant_past_asym(cache: torch.Tensor, bt_row: torch.Tensor, ctx: int, nb: int, KUB: int,
                      KSB: int, VUB: int, VSB: int, phK: int, ku: int, kgs: int, vu: int,
                      vgs: int, Hk: int, D: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Asymmetric K/V twin of dequant_past: K plane [:phK] (ku,kgs), V plane [phK:] (vu,vgs)."""
    bs = cache.shape[1]
    npb = (ctx + bs - 1) // bs
    phys = cache[bt_row[:npb].long()]
    flat = phys.reshape(npb * bs, Hk, cache.shape[-1])[:ctx]

    def deq(blob: torch.Tensor, UB: int, SB: int, l_b: int, g_size: int) -> torch.Tensor:
        blk = blob.reshape(ctx, Hk, nb, 1 + UB + SB)
        sc = blk[..., 0].permute(1, 2, 0).contiguous().view(torch.int8)   # [Hk,nb,ctx]
        up = blk[..., 1:1 + UB].permute(1, 2, 0, 3).contiguous()          # [Hk,nb,ctx,UB]
        sh = blk[..., 1 + UB:].permute(1, 2, 0, 3).contiguous()           # [Hk,nb,ctx,SB]
        outs = [torch.ops.ssnf.wonly_dequant_bf16(sc[h], up[h], sh[h], ctx, D, nb, l_b, g_size)
                for h in range(Hk)]
        return torch.stack(outs)                                          # [Hk,ctx,D]
    return deq(flat[..., :phK], KUB, KSB, ku, kgs), deq(flat[..., phK:], VUB, VSB, vu, vgs)


def dequant_past_mxint8(cache: torch.Tensor, bt_row: torch.Tensor, ctx: int, nb: int,
                        per_head: int, Hk: int, D: int) -> tuple[torch.Tensor, torch.Tensor]:
    """MXINT8 twin: per block [e8m0 exp(1) | int8(32)] -> value = int8 * 2^exp. (torch, no kernel.)"""
    bs = cache.shape[1]
    npb = (ctx + bs - 1) // bs
    phys = cache[bt_row[:npb].long()]
    flat = phys.reshape(npb * bs, Hk, cache.shape[-1])[:ctx]              # [ctx,Hk,2*ph] uint8

    def deq(blob: torch.Tensor) -> torch.Tensor:                          # [ctx,Hk,nb*33]
        blk = blob.reshape(ctx, Hk, nb, 33)
        exp = blk[..., 0].contiguous().view(torch.int8).float()          # [ctx,Hk,nb] (signed exp)
        q = blk[..., 1:].contiguous().view(torch.int8).float()           # [ctx,Hk,nb,32]
        val = (q * torch.exp2(exp).unsqueeze(-1)).reshape(ctx, Hk, nb * 32)
        return val.permute(1, 0, 2).to(torch.bfloat16).contiguous()      # [Hk,ctx,D]
    return deq(flat[..., :per_head]), deq(flat[..., per_head:])


def dequant_past_mxfp8(cache: torch.Tensor, bt_row: torch.Tensor, ctx: int, nb: int,
                       per_head: int, Hk: int, D: int) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP8 twin of dequant_past_mxint8: [e8m0 exp(1) | e4m3(32)] -> e4m3_decode * 2^exp."""
    bs = cache.shape[1]
    npb = (ctx + bs - 1) // bs
    phys = cache[bt_row[:npb].long()]
    flat = phys.reshape(npb * bs, Hk, cache.shape[-1])[:ctx]              # [ctx,Hk,2*ph] uint8

    def deq(blob: torch.Tensor) -> torch.Tensor:                          # [ctx,Hk,nb*33]
        blk = blob.reshape(ctx, Hk, nb, 33)
        exp = blk[..., 0].contiguous().view(torch.int8).float()          # [ctx,Hk,nb] (signed exp)
        q = blk[..., 1:].contiguous().view(torch.float8_e4m3fn).float()  # [ctx,Hk,nb,32] e4m3->f
        val = (q * torch.exp2(exp).unsqueeze(-1)).reshape(ctx, Hk, nb * 32)
        return val.permute(1, 0, 2).to(torch.bfloat16).contiguous()      # [Hk,ctx,D]
    return deq(flat[..., :per_head]), deq(flat[..., per_head:])


def dequant_past_mxfp6(cache: torch.Tensor, bt_row: torch.Tensor, ctx: int, nb: int,
                       per_head: int, Hk: int, D: int) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP6-E2M3 twin: [e8m0 exp(1) | 24 B packed 6-bit codes] -> e2m3 decode * 2^exp."""
    bs = cache.shape[1]
    npb = (ctx + bs - 1) // bs
    phys = cache[bt_row[:npb].long()]
    flat = phys.reshape(npb * bs, Hk, cache.shape[-1])[:ctx]              # [ctx,Hk,2*ph] uint8
    _sh = torch.tensor([0, 6, 12, 18], device=cache.device)

    def deq(blob: torch.Tensor) -> torch.Tensor:                          # [ctx,Hk,nb*25]
        blk = blob.reshape(ctx, Hk, nb, 25)
        exp = blk[..., 0].contiguous().view(torch.int8).float()          # [ctx,Hk,nb]
        w = blk[..., 1:].contiguous().reshape(ctx, Hk, nb, 8, 3).int()   # 8 groups x 3 bytes
        word = w[..., 0] | (w[..., 1] << 8) | (w[..., 2] << 16)          # [ctx,Hk,nb,8]
        c = (word.unsqueeze(-1) >> _sh) & 0x3F                           # [ctx,Hk,nb,8,4]
        c = c.reshape(ctx, Hk, nb, 32)
        e = (c >> 3) & 3
        mant = ((c & 7) + (e != 0).int() * 8).float()
        val = mant * torch.exp2((torch.clamp(e, min=1) - 4).float())     # e2m3 magnitude
        val = val * (1.0 - 2.0 * ((c >> 5) & 1).float())                 # apply sign
        val = (val * torch.exp2(exp).unsqueeze(-1)).reshape(ctx, Hk, nb * 32)
        return val.permute(1, 0, 2).to(torch.bfloat16).contiguous()      # [Hk,ctx,D]
    return deq(flat[..., :per_head]), deq(flat[..., per_head:])

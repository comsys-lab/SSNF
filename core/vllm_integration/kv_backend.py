# SSNF paged-KV attention backend for vLLM 0.23 (V1). Selection: register_backend(CUSTOM) +
# VLLM_ATTENTION_BACKEND=SSNF; kv-cache dtype "ssnf_u3gs16" (also mxint8_kv / mxfp8_kv / mxfp6_kv).
# Cache = one uint8 tensor (num_blocks, block_size, Hkv, SLOT), token-major [K|V] slot bytes
# (layout round-trip verified vs kv_layout.py); store = fused append kernels, decode = paged kernel.
import math
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch

import ssnf_cuda  # noqa: F401 — registers torch.ops.ssnf.*
from vllm.config.cache import CacheDType
from vllm.v1.attention.backend import (
    AttentionBackend, AttentionCGSupport, AttentionImpl, AttentionLayer,
    AttentionMetadata, AttentionMetadataBuilder, AttentionType, CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.backends.fa_utils import (
    is_flash_attn_varlen_func_available,
)
if is_flash_attn_varlen_func_available():
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import AttentionSpec

U, GS, BLOCK = 3, 16, 32

# Fused single-kernel KV append (default ON). SSNF_APPEND_FUSED=0 -> old ~14-op sequence (A/B).
_APPEND_FUSED = os.environ.get("SSNF_APPEND_FUSED", "1") not in ("0", "", "false", "False")

# Importance-weighted additive shared: K-only. SSNF_KV_IMPORTANCE=<H.pt> enables
# (dict with H_K [L,Hkv,D]); SSNF_KV_WSHARE_RADIUS>=1 -> trunc search (0 -> hmean weighted-mean).
# Decode UNCHANGED (only which shared code is written differs). Off (empty path) -> bit-exact legacy.
_IMPORTANCE_PATH = os.environ.get("SSNF_KV_IMPORTANCE", "")
_WSHARE_RADIUS = int(os.environ.get("SSNF_KV_WSHARE_RADIUS", "0"))
# Asymmetric K/V (asym kernel): SSNF_KV_ASYM="ku,kgs,vu,vgs" (e.g. "3,32,4,32").
# K plane uses (ku,kgs)+importance, V plane (vu,vgs) plain; smaller slot -> more KV blocks (capacity).
_ASYM = None
if os.environ.get("SSNF_KV_ASYM"):
    _K_LB, _K_G_SIZE, _V_LB, _V_G_SIZE = (int(x) for x in os.environ["SSNF_KV_ASYM"].split(","))
    _ASYM = (_K_LB, _K_G_SIZE, _V_LB, _V_G_SIZE)


def _slot_geom_asym(D: int) -> tuple[int, int, int, int, int, int, int]:
    nb = D // BLOCK
    KUB = BLOCK * (8 - _K_LB) // 8; KSB = math.ceil((BLOCK // _K_G_SIZE) * _K_LB / 8); phK = nb * (1 + KUB + KSB)
    VUB = BLOCK * (8 - _V_LB) // 8; VSB = math.ceil((BLOCK // _V_G_SIZE) * _V_LB / 8); phV = nb * (1 + VUB + VSB)
    return nb, KUB, KSB, VUB, VSB, phK, phV
_HK = {"loaded": False, "t": None, "n": 0}   # t: [L, Hkv*D] fp32 cuda


def _load_importance(device: torch.device) -> None:
    if _HK["loaded"]:
        return
    _HK["loaded"] = True
    if not _IMPORTANCE_PATH or not os.path.exists(_IMPORTANCE_PATH):
        return
    d = torch.load(_IMPORTANCE_PATH, map_location="cpu")
    HK = d["H_K"].to(torch.float32)                       # [L, Hkv, D]
    _HK["n"] = HK.shape[0]
    _HK["t"] = HK.reshape(_HK["n"], -1).contiguous().to(device)   # [L, Hkv*D]
    print(f"[SSNF] loaded K-importance {tuple(HK.shape)} from {_IMPORTANCE_PATH} "
          f"(radius={_WSHARE_RADIUS})", flush=True)


def _hk_layer(layer: AttentionLayer, device: torch.device) -> torch.Tensor:
    """[Hkv*D] fp32 K-importance for this layer, or empty tensor (off/unknown layer)."""
    _load_importance(device)
    empty = torch.empty(0, device=device, dtype=torch.float32)
    if _HK["t"] is None:
        return empty
    nm = getattr(layer, "layer_name", None) or ""
    m = re.search(r"layers?\.(\d+)", nm)
    if not m:
        return empty
    li = int(m.group(1))
    return _HK["t"][li] if 0 <= li < _HK["n"] else empty


def _slot_geom(D: int) -> tuple[int, int, int, int]:
    nb = D // BLOCK
    UB = BLOCK * (8 - U) // 8            # 20 for D-block of 32, u3
    SB = math.ceil((BLOCK // GS) * U / 8)  # 1
    per_head = nb * (1 + UB + SB)        # 88 for D=128
    return nb, UB, SB, per_head


class SSNFAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False
    supported_dtypes: ClassVar[list] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["ssnf_u3gs16", "mxint8_kv", "mxfp8_kv", "mxfp6_kv"]

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"  # registered under AttentionBackendEnum.CUSTOM

    @staticmethod
    def get_impl_cls() -> type["SSNFAttentionImpl"]:
        return SSNFAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["SSNFMetadataBuilder"]:
        return SSNFMetadataBuilder

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: "CacheDType | None") -> bool:
        return kv_cache_dtype is not None and kv_cache_dtype.startswith(("ssnf_", "mxint8_", "mxfp8_", "mxfp6_"))

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size > 0

    @staticmethod
    def get_kv_cache_shape(num_blocks: int, block_size: int, num_kv_heads: int, head_size: int,
                           cache_dtype_str: str = "ssnf_u3gs16") -> tuple[int, ...]:
        if cache_dtype_str.startswith("mxint8") or cache_dtype_str.startswith("mxfp8"):
            per_head = (head_size // BLOCK) * (1 + BLOCK)        # scale(1) + 32 codes per block
            slot = 2 * per_head
        elif cache_dtype_str.startswith("mxfp6"):
            per_head = (head_size // BLOCK) * (1 + 24)           # scale(1) + 24 packed 6-bit bytes
            slot = 2 * per_head
        elif _ASYM is not None:                                 # asym: slot = per_head_K + per_head_V
            _, _, _, _, _, phK, phV = _slot_geom_asym(head_size)
            slot = phK + phV
        else:
            _, _, _, per_head = _slot_geom(head_size)
            slot = 2 * per_head
        return (num_blocks, block_size, num_kv_heads, slot)


@dataclass
class SSNFMetadata(AttentionMetadata):
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    num_actual_tokens: int = 0
    max_query_len: int = 0
    max_seq_len: int = 0
    is_prefill: bool = False
    num_decodes: int = 0
    num_decode_tokens: int = 0
    seq_lens_cpu: torch.Tensor | None = None


class SSNFMetadataBuilder(AttentionMetadataBuilder[SSNFMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(self, kv_cache_spec: "AttentionSpec", layer_names: list[str],
                 vllm_config: "VllmConfig", device: torch.device) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)

    def build(self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> SSNFMetadata:
        cam = common_attn_metadata
        nd, npf, ndt, _ = split_decodes_and_prefills(
            cam, decode_threshold=self.reorder_batch_threshold)
        return SSNFMetadata(
            seq_lens=cam.seq_lens, slot_mapping=cam.slot_mapping,
            block_table=cam.block_table_tensor, query_start_loc=cam.query_start_loc,
            num_actual_tokens=cam.num_actual_tokens, max_query_len=cam.max_query_len,
            max_seq_len=cam.max_seq_len, is_prefill=(cam.max_query_len > 1),
            num_decodes=nd, num_decode_tokens=ndt,
            seq_lens_cpu=cam.seq_lens_cpu_upper_bound,
        )


class SSNFAttentionImpl(AttentionImpl["SSNFMetadata"]):
    def __init__(self, num_heads: int, head_size: int, scale: float,
                 num_kv_heads: int | None = None,
                 alibi_slopes: list[float] | None = None,
                 sliding_window: int | None = None, kv_cache_dtype: str = "auto",
                 logits_soft_cap: float | None = None, attn_type: str = AttentionType.DECODER,
                 kv_sharing_target_layer_name: str | None = None, **kwargs) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads or num_heads
        self.kv_cache_dtype = kv_cache_dtype
        # format dispatch: plain MXINT8 (fair baseline) vs SSNF. mxint8 slot = nb*(1+32); no UB/SB.
        self.mxi = str(kv_cache_dtype).startswith("mxint8")
        self.mxf8 = str(kv_cache_dtype).startswith("mxfp8")   # native MXFP8-E4M3 KV (mxint8 slot twin)
        self.mxf6 = str(kv_cache_dtype).startswith("mxfp6")   # MXFP6-E2M3 KV (6-bit packed, slot nb*25)
        self.asym = False
        if self.mxi or self.mxf8:
            self.nb = head_size // BLOCK
            self.per_head = self.nb * (1 + BLOCK)
            self.UB = self.SB = 0
        elif self.mxf6:
            self.nb = head_size // BLOCK
            self.per_head = self.nb * (1 + 24)               # scale(1) + 24 packed 6-bit code bytes
            self.UB = self.SB = 0
        elif _ASYM is not None:                                 # asymmetric K/V
            self.asym = True
            self.nb, self.KUB, self.KSB, self.VUB, self.VSB, self.phK, self.phV = _slot_geom_asym(head_size)
            self.per_head = self.phK                            # K plane size = V offset
            self.slot = self.phK + self.phV
            self.UB, self.SB = self.KUB, self.KSB               # legacy fields (unused in asym path)
        else:
            self.nb, self.UB, self.SB, self.per_head = _slot_geom(head_size)

    # --- byte surgery: pack-planes <-> token-major blob (torch; mirrors kv_layout.py) ---
    def _planes_to_blob(self, scale: torch.Tensor, upper: torch.Tensor, shared: torch.Tensor,
                        N: int) -> torch.Tensor:
        # scale [Hk,nb,N] int8 ; upper [Hk,nb,N,UB] ; shared [Hk,nb,N,SB]  (kv_write output)
        Hk = self.num_kv_heads
        sc = scale.permute(2, 0, 1).to(torch.uint8).unsqueeze(-1)        # [N,Hk,nb,1]
        up = upper.permute(2, 0, 1, 3)                                   # [N,Hk,nb,UB]
        sh = shared.permute(2, 0, 1, 3)                                  # [N,Hk,nb,SB]
        blk = torch.cat([sc, up, sh], dim=-1)                            # [N,Hk,nb,1+UB+SB]
        return blk.reshape(N, Hk, self.per_head).contiguous()           # [N,Hk,per_head]

    def _planes_to_blob_mxint8(self, scale: torch.Tensor, qweight: torch.Tensor,
                               N: int) -> torch.Tensor:
        # scale [Hk,nb,N] int8, qweight [Hk,nb,N,32] int8 -> [N,Hk,per_head=nb*33] uint8
        Hk = self.num_kv_heads
        sc = scale.permute(2, 0, 1).unsqueeze(-1)                        # [N,Hk,nb,1] int8
        q = qweight.permute(2, 0, 1, 3)                                  # [N,Hk,nb,32] int8
        blk = torch.cat([sc, q], dim=-1)                                 # [N,Hk,nb,33] int8
        return blk.reshape(N, Hk, self.per_head).contiguous().view(torch.uint8)

    def _blob_to_planes(self, blob: torch.Tensor,
                        L: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # blob [L,Hk,per_head] uint8 -> planes for kv_decode_attention
        Hk = self.num_kv_heads
        blk = blob.view(L, Hk, self.nb, 1 + self.UB + self.SB)
        sc = blk[..., 0].permute(1, 2, 0).contiguous().view(torch.int8)        # [Hk,nb,L]
        up = blk[..., 1:1 + self.UB].permute(1, 2, 0, 3).contiguous()          # [Hk,nb,L,UB]
        sh = blk[..., 1 + self.UB:].permute(1, 2, 0, 3).contiguous()           # [Hk,nb,L,SB]
        return sc, up, sh

    def do_kv_cache_update(self, layer: AttentionLayer, key: torch.Tensor, value: torch.Tensor,
                           kv_cache: torch.Tensor, slot_mapping: torch.Tensor) -> None:
        N = slot_mapping.shape[0]
        if N <= 0:
            return
        D, Hk = self.head_size, self.num_kv_heads
        if self.asym:                                          # asymmetric K/V fused append
            torch.ops.ssnf.kv_append_paged_fused_asym(
                key[:N].contiguous(), value[:N].contiguous(), kv_cache,
                slot_mapping.to(torch.int64), N, Hk, D, self.nb, _K_LB, _K_G_SIZE, _V_LB, _V_G_SIZE,
                self.phK, self.slot, _hk_layer(layer, kv_cache.device), _WSHARE_RADIUS)
            return
        _fused = os.environ.get("SSNF_APPEND_FUSED", "1") not in ("0", "", "false", "False")
        if (self.mxi or self.mxf8 or self.mxf6) and _fused:    # fair fused write path (matches SSNF)
            op = (torch.ops.ssnf.mxint8_kv_append_fused if self.mxi
                  else torch.ops.ssnf.mxfp8_kv_append_fused if self.mxf8
                  else torch.ops.ssnf.mxfp6_kv_append_fused)
            op(key[:N].contiguous(), value[:N].contiguous(), kv_cache,
               slot_mapping.to(torch.int64), N, Hk, D, self.nb, self.per_head, 2 * self.per_head)
            return
        if not self.mxi and not self.mxf8 and not self.mxf6 and _fused:
            # FUSED fast path (deploy default): ONE kernel quantizes+packs k/v and writes the blob
            # straight into slots — replaces permute/kv_write x2/planes_to_blob x2/cat/scatter
            # (~14 launches -> 1). Byte-IDENTICAL to the sequence below (verified). ~8x faster append.
            torch.ops.ssnf.kv_append_paged_fused(
                key[:N].contiguous(), value[:N].contiguous(), kv_cache,
                slot_mapping.to(torch.int64), N, Hk, D, self.nb, U, GS,
                self.per_head, 2 * self.per_head,
                _hk_layer(layer, kv_cache.device), _WSHARE_RADIUS)
            return
        k = key[:N].view(N, Hk, D).permute(1, 0, 2).contiguous()   # [Hk,N,D]
        v = value[:N].view(N, Hk, D).permute(1, 0, 2).contiguous()
        if self.mxi:
            ks, kq = torch.ops.ssnf.mxint8_kv_write(k, Hk, N, D, self.nb)
            vs, vq = torch.ops.ssnf.mxint8_kv_write(v, Hk, N, D, self.nb)
            kblob = self._planes_to_blob_mxint8(ks, kq, N)
            vblob = self._planes_to_blob_mxint8(vs, vq, N)
        elif self.mxf8:                                            # MXFP8-E4M3 (same slot bytes)
            ks, kq = torch.ops.ssnf.mxfp8_kv_write(k, Hk, N, D, self.nb)
            vs, vq = torch.ops.ssnf.mxfp8_kv_write(v, Hk, N, D, self.nb)
            kblob = self._planes_to_blob_mxint8(ks, kq, N)
            vblob = self._planes_to_blob_mxint8(vs, vq, N)
        elif self.mxf6:                                            # MXFP6-E2M3 (24B packed codes/block)
            ks, kq = torch.ops.ssnf.mxfp6_kv_write(k, Hk, N, D, self.nb)
            vs, vq = torch.ops.ssnf.mxfp6_kv_write(v, Hk, N, D, self.nb)
            kblob = self._planes_to_blob_mxint8(ks, kq, N)         # generic cat(scale,codes)+reshape
            vblob = self._planes_to_blob_mxint8(vs, vq, N)
        else:
            ks, ku, kh = torch.ops.ssnf.kv_write(k, Hk, N, D, self.nb, U, GS)
            vs, vu, vh = torch.ops.ssnf.kv_write(v, Hk, N, D, self.nb, U, GS)
            kblob = self._planes_to_blob(ks, ku, kh, N)
            vblob = self._planes_to_blob(vs, vu, vh, N)
        slot = torch.cat([kblob, vblob], dim=-1)                   # [N,Hk,2*per_head]
        nblk, bs, _, S = kv_cache.shape
        flat = kv_cache.view(nblk * bs, Hk, S)
        flat[slot_mapping] = slot

    def _paged(self, q: torch.Tensor, kv_cache: torch.Tensor, block_table: torch.Tensor,
               seq_lens: torch.Tensor, Nt: int, Hq: int, Hk: int, D: int) -> torch.Tensor:
        nblk, bs, _, S = kv_cache.shape
        bt = block_table.to(torch.int32).contiguous()
        sl = seq_lens.to(torch.int32).contiguous()
        if self.mxi:
            return torch.ops.ssnf.mxint8_kv_decode_paged(
                q.contiguous(), kv_cache, bt, sl, Nt, Hq, Hk, D, self.nb, bs, S, self.per_head)
        if self.mxf8:
            return torch.ops.ssnf.mxfp8_kv_decode_paged(
                q.contiguous(), kv_cache, bt, sl, Nt, Hq, Hk, D, self.nb, bs, S, self.per_head)
        if self.mxf6:
            return torch.ops.ssnf.mxfp6_kv_decode_paged(
                q.contiguous(), kv_cache, bt, sl, Nt, Hq, Hk, D, self.nb, bs, S, self.per_head)
        if self.asym:
            return torch.ops.ssnf.kv_decode_paged_asym(
                q.contiguous(), kv_cache, bt, sl, Nt, Hq, Hk, D, self.nb,
                _K_LB, _K_G_SIZE, self.KUB, self.KSB, _V_LB, _V_G_SIZE, self.VUB, self.VSB, bs, S, self.phK)
        return torch.ops.ssnf.kv_decode_paged(
            q.contiguous(), kv_cache, bt, sl,
            Nt, Hq, Hk, D, self.nb, U, GS, self.UB, self.SB, bs, S, self.per_head)

    def _prefill_paged(self, q_pf: torch.Tensor, kv_cache: torch.Tensor, pf_qsl: torch.Tensor,
                       pf_seq_lens: torch.Tensor, pf_bt: torch.Tensor, Npf: int, Hq: int,
                       Hk: int, D: int) -> torch.Tensor:
        """Chunked-prefill / continuation: each prefill token attends causally to
        [0, ctx+local] reading past quantized blocks + current chunk from the cache."""
        from ssnf_vllm.paged_prefill import synth_causal
        synth_bt, synth_L = synth_causal(pf_qsl, pf_seq_lens, pf_bt, Npf)
        return self._paged(q_pf, kv_cache, synth_bt, synth_L, Npf, Hq, Hk, D)

    def forward(self, layer: AttentionLayer, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, kv_cache: torch.Tensor,
                attn_metadata: "SSNFMetadata | None",
                output: torch.Tensor | None = None,
                output_scale: torch.Tensor | None = None,
                output_block_scale: torch.Tensor | None = None) -> torch.Tensor:
        Hq, D = self.num_heads, self.head_size
        Hk = self.num_kv_heads
        Nq = query.shape[0]
        if output is None:
            output = torch.zeros(Nq, Hq * D, dtype=query.dtype, device=query.device)
        if attn_metadata is None:
            return output
        N = attn_metadata.num_actual_tokens
        if N <= 0:
            return output
        q = query[:N].view(N, Hq, D)
        nd = attn_metadata.num_decodes
        ndt = attn_metadata.num_decode_tokens
        attn_out = torch.empty(N, Hq, D, dtype=q.dtype, device=q.device)

        # decode portion (reorder_batch puts the nd decode requests first, 1 token each)
        if ndt > 0:
            attn_out[:ndt] = self._paged(
                q[:ndt], kv_cache, attn_metadata.block_table[:nd],
                attn_metadata.seq_lens[:nd], ndt, Hq, Hk, D)

        # prefill portion (remaining tokens)
        if N > ndt:
            # "fresh" = the prefill tokens have NO past quantized context (each prefill seq's query
            # spans its whole seq). Then attend over the RAW k/v with bf16 flash — full precision, no
            # quantized emulation — EVEN in a mixed decode+prefill batch (relaxes the old nd==0 gate,
            # which forced fresh prefill through synth_causal whenever any decode shared the step).
            # Deploy intent: quantize KV cache for decode reads, compute prefill in bf16. The quantized
            # chunk is still written by do_kv_cache_update for future decodes.
            pf_qsl = attn_metadata.query_start_loc[nd:]                 # [num_prefill+1], starts at ndt
            fresh = (nd == 0 and attn_metadata.max_query_len == attn_metadata.max_seq_len)
            if not fresh:
                fresh = bool((attn_metadata.seq_lens[nd:] == (pf_qsl[1:] - pf_qsl[:-1])).all())
            if fresh:
                kp = key[ndt:N].view(N - ndt, Hk, D)
                vp = value[ndt:N].view(N - ndt, Hk, D)
                cu = (pf_qsl - ndt).to(torch.int32)
                mql = int((pf_qsl[1:] - pf_qsl[:-1]).max())
                attn_out[ndt:] = flash_attn_varlen_func(
                    q=q[ndt:], k=kp, v=vp,
                    cu_seqlens_q=cu, cu_seqlens_k=cu,
                    max_seqlen_q=mql, max_seqlen_k=mql,
                    softmax_scale=self.scale, causal=True,
                )
            else:
                # continuation (some prefill seq has PAST quantized context, e.g. chunked prefill):
                # per-seq dequant the past quantized K/V to bf16 + raw current chunk -> flash (causal).
                # (a) full bf16 prefill even under chunked prefill; format-dispatched dequant so asym/
                # mxint8 get it too (was synth_causal). Fresh seqs (ctx==0) use raw k/v directly.
                from ssnf_vllm.paged_prefill import (dequant_past, dequant_past_asym,
                                                     dequant_past_mxint8, dequant_past_mxfp8,
                                                     dequant_past_mxfp6)
                qsl = attn_metadata.query_start_loc.tolist()   # [num_seqs+1] (one sync, prefill only)
                sll = attn_metadata.seq_lens.tolist()          # [num_seqs]
                bt = attn_metadata.block_table
                for i in range(nd, len(sll)):
                    qs, qe = qsl[i], qsl[i + 1]
                    ql = qe - qs
                    if ql <= 0:
                        continue
                    ctx = sll[i] - ql                          # past (cached) tokens for this seq
                    q_seq = q[qs:qe]                           # [ql,Hq,D]
                    k_raw = key[qs:qe].view(ql, Hk, D)
                    v_raw = value[qs:qe].view(ql, Hk, D)
                    if ctx > 0:                                # past (cached) tokens exist
                        if self.mxi:
                            Kp, Vp = dequant_past_mxint8(kv_cache, bt[i], ctx, self.nb,
                                                         self.per_head, Hk, D)
                        elif self.mxf8:
                            Kp, Vp = dequant_past_mxfp8(kv_cache, bt[i], ctx, self.nb,
                                                        self.per_head, Hk, D)
                        elif self.mxf6:
                            Kp, Vp = dequant_past_mxfp6(kv_cache, bt[i], ctx, self.nb,
                                                        self.per_head, Hk, D)
                        elif self.asym:
                            Kp, Vp = dequant_past_asym(kv_cache, bt[i], ctx, self.nb, self.KUB,
                                                       self.KSB, self.VUB, self.VSB, self.phK,
                                                       _K_LB, _K_G_SIZE, _V_LB, _V_G_SIZE, Hk, D)
                        else:
                            Kp, Vp = dequant_past(kv_cache, bt[i], ctx, self.nb, self.UB, self.SB,
                                                  self.per_head, U, GS, Hk, D)   # [Hk,ctx,D]
                        k_full = torch.cat([Kp.transpose(0, 1), k_raw], 0)   # [ctx+ql, Hk, D]
                        v_full = torch.cat([Vp.transpose(0, 1), v_raw], 0)
                    else:
                        k_full, v_full = k_raw, v_raw
                    lk = ctx + ql
                    cu_q = torch.tensor([0, ql], dtype=torch.int32, device=q.device)
                    cu_k = torch.tensor([0, lk], dtype=torch.int32, device=q.device)
                    attn_out[qs:qe] = flash_attn_varlen_func(
                        q=q_seq, k=k_full, v=v_full, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                        max_seqlen_q=ql, max_seqlen_k=lk, softmax_scale=self.scale, causal=True)

        if output.ndim == 3:
            output[:N] = attn_out.to(output.dtype)
        else:
            output[:N] = attn_out.reshape(N, -1).to(output.dtype)
        return output


def register() -> None:
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
    register_backend(AttentionBackendEnum.CUSTOM, __name__ + ".SSNFAttentionBackend")

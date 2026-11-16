"""SSNF wikitext-2 PPL at the deployed config (minus rotation): K act-weighted mean (per-layer H_K) radius=0, V uniform mean radius=0, W greedy offline search radius=1<<u.
Deployed config (paper): W l_b3/g8 (5.63b), K l_b3/g16 (5.44b), V l_b4/g32 (4.38b); PPL protocol = PPL_MAXLEN/PPL_STRIDE/PPL_WINDOWS (2048/1024/20 default).
Llama/Mistral use sdpa; Gemma-2 uses eager+BOS (soft-capping). H_K from calib/H_K_*.pt.
Run: SSNF_MODEL=<id> SSNF_HK=calib/H_K_<model>.pt CUDA_VISIBLE_DEVICES=0 python eval_accuracy/wkv_v2_ppl.py
"""
from __future__ import annotations
import os, sys, torch
from typing import Callable
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEV = "cuda"
MAXLEN = int(os.environ.get("PPL_MAXLEN", "2048")); STRIDE = int(os.environ.get("PPL_STRIDE", "1024"))
MW = int(os.environ.get("PPL_WINDOWS", "20"))   # window cap; set huge for full-dataset PPL
LINEAR_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def ssnf_hw(x: torch.Tensor, l_b: int, g_size: int, Hfull: torch.Tensor | None = None, radius: int = 0) -> torch.Tensor:
    """INT-SSNF encoder: E8M0 block scale, H-weighted shared group mean, optional radius search (radius>0 re-rounds h_word per shared candidate, argmin weighted-L2)."""
    shp = x.shape
    xf = x.reshape(-1, 32).float(); B = xf.shape[0]
    absmax = xf.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    s_base = torch.exp2(torch.floor(torch.log2(absmax)) - 6.0)
    s_unshared = s_base * float(1 << l_b)
    q_max = (1 << (7 - l_b)) - 1
    s_min, s_max = -(1 << (l_b - 1)), (1 << (l_b - 1)) - 1
    n_group = 32 // g_size
    Hb = torch.ones_like(xf) if Hfull is None else Hfull.reshape(-1, 32).float()
    h_word = torch.round(xf / s_unshared).clamp(-q_max, q_max)
    residual = (xf - h_word * s_unshared).reshape(B, n_group, g_size)
    Hg = Hb.reshape(B, n_group, g_size)
    res_avg = (Hg * residual).sum(2) / Hg.sum(2).clamp(min=1e-30)
    l_g_word = torch.round(res_avg / s_base).clamp(s_min, s_max)
    if radius == 0:
        recon = h_word * s_unshared + l_g_word.repeat_interleave(g_size, dim=1) * s_base
        return recon.reshape(shp).to(x.dtype)
    best_err = torch.full((B, n_group), float("inf"), device=xf.device)
    best_h_word = h_word.clone(); best_l_g_word = l_g_word.clone()
    for d in range(-radius, radius + 1):
        sh = (l_g_word + d).clamp(s_min, s_max)
        sh_e = sh.repeat_interleave(g_size, dim=1)
        up = torch.round((xf - sh_e * s_base) / s_unshared).clamp(-q_max, q_max)
        err = (Hb * (up * s_unshared + sh_e * s_base - xf) ** 2).reshape(B, n_group, g_size).sum(2)
        take = err < best_err
        best_err = torch.where(take, err, best_err)
        best_l_g_word = torch.where(take, sh, best_l_g_word)
        best_h_word = torch.where(take.repeat_interleave(g_size, dim=1), up, best_h_word)
    recon = best_h_word * s_unshared + best_l_g_word.repeat_interleave(g_size, dim=1) * s_base
    return recon.reshape(shp).to(x.dtype)


def wq_scale(x: torch.Tensor, l_b: int, g_size: int, Hf: torch.Tensor | None, radius: int = 8, offs: tuple[int, ...] = (-1, 0, 1)) -> torch.Tensor:
    """Weight quantizer: greedy+H shared radius search plus upper reround, nested under per-block E8M0 scale-offset candidates (argmin H-weighted L2; format unchanged)."""
    shp = x.shape
    xf = x.reshape(-1, 32).float(); B = xf.shape[0]
    e0 = torch.floor(torch.log2(xf.abs().amax(-1, keepdim=True).clamp(min=1e-30))) - 6.0
    q_max = (1 << (7 - l_b)) - 1; s_min, s_max = -(1 << (l_b - 1)), (1 << (l_b - 1)) - 1; n_group = 32 // g_size
    Hb = torch.ones_like(xf) if Hf is None else Hf.reshape(-1, 32).float()
    Hg = Hb.reshape(B, n_group, g_size)
    best_recon = None; best_err = None
    for off in offs:
        s_base = torch.exp2(e0 + float(off)); s_unshared = s_base * float(1 << l_b)
        up = torch.round(xf / s_unshared).clamp(-q_max, q_max)
        residual = (xf - up * s_unshared).reshape(B, n_group, g_size)
        l_g_word = torch.round((Hg * residual).sum(2) / Hg.sum(2).clamp(min=1e-30) / s_base).clamp(s_min, s_max)
        be = torch.full((B, n_group), float("inf"), device=xf.device); bu = up.clone(); bs = l_g_word.clone()
        for d in range(-radius, radius + 1):
            sh = (l_g_word + d).clamp(s_min, s_max); sh_e = sh.repeat_interleave(g_size, 1)
            u2 = torch.round((xf - sh_e * s_base) / s_unshared).clamp(-q_max, q_max)
            err = (Hb * (u2 * s_unshared + sh_e * s_base - xf) ** 2).reshape(B, n_group, g_size).sum(2)
            take = err < be; be = torch.where(take, err, be); bs = torch.where(take, sh, bs)
            bu = torch.where(take.repeat_interleave(g_size, 1), u2, bu)
        recon = bu * s_unshared + bs.repeat_interleave(g_size, 1) * s_base
        berr = (Hb * (recon - xf) ** 2).sum(1, keepdim=True)
        if best_recon is None: best_recon, best_err = recon, berr
        else:
            take = berr < best_err
            best_recon = torch.where(take.expand_as(recon), recon, best_recon); best_err = torch.where(take, berr, best_err)
    return best_recon.reshape(shp).to(x.dtype)


_S = {"on": False, "HK": None, "Hkv": None, "L": None, "i": 0}   # HK: [L,Hkv,D]
def _kv_quant(k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize one layer's K (act-weighted H_K) and V (uniform mean) with SSNF."""
    # K: per-layer activation-weighted (H_K), V: uniform. per-32 along head_dim, no rotation.
    layer = _S["i"] % _S["L"]; _S["i"] += 1
    HKl = _S["HK"][layer]                                     # [Hkv, D]
    B, nh, Sq, D = k.shape; Hkv = _S["Hkv"]
    hidx = torch.arange(nh, device=k.device) // (nh // Hkv)   # map (repeated) head -> kv-head
    Hfull = HKl[hidx].reshape(1, nh, 1, D).expand(B, nh, Sq, D)
    kr = _S.get("kr", 0); vr = _S.get("vr", 0)
    kgs = _S.get("kgs", 32); vgs = _S.get("vgs", 32)
    kq = ssnf_hw(k.float(), 3, kgs, Hfull, radius=kr)
    vq = ssnf_hw(v.float(), 4, vgs, None, radius=vr)
    return kq.to(k.dtype), vq.to(v.dtype)


@torch.no_grad()
def ppl(model: torch.nn.Module, ids: torch.Tensor, bos: int | None = None) -> float:
    """Sliding-window wikitext-2 perplexity (MAXLEN/STRIDE windows, capped at MW)."""
    seq = ids.size(1); nll, ntok, prev, n = 0.0, 0, 0, 0
    for b in range(0, seq, STRIDE):
        e = min(b + MAXLEN, seq); trg = e - prev
        win = ids[:, b:e].to(DEV)
        inp = win if bos is None else torch.cat([torch.tensor([[bos]], device=DEV), win], 1)
        t = inp.clone(); t[:, :-trg] = -100
        _S["i"] = 0
        nll += model(inp, labels=t).loss.double().item() * trg; ntok += trg; prev = e; n += 1
        if n >= MW or e == seq: break
    return torch.exp(torch.tensor(nll / ntok)).item()


def main() -> None:
    """Measure BF16 / KV-only / W+KV PPL at the deployed SSNF config."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    MODEL = os.environ.get("SSNF_MODEL", "NousResearch/Meta-Llama-3.1-8B")
    HKF = os.environ["SSNF_HK"]
    is_gemma = "gemma" in MODEL.lower()
    attn = "eager" if is_gemma else "sdpa"
    def is_t(n: str, m: torch.nn.Module) -> bool: return isinstance(m, torch.nn.Linear) and any(p in n for p in LINEAR_KEYS)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation=attn).to(DEV).eval()
    d = torch.load(HKF, map_location=DEV)
    _S["HK"] = d["H_K"].to(torch.float32).to(DEV)                # [L,Hkv,D]
    _S["L"], _S["Hkv"] = _S["HK"].shape[0], _S["HK"].shape[1]
    _S["kr"] = int(os.environ.get("SSNF_KV_RADIUS", "0")); _S["vr"] = _S["kr"]   # 0 = deployed (no search)
    _S["kgs"] = int(os.environ.get("SSNF_K_GS", "16")); _S["vgs"] = int(os.environ.get("SSNF_V_GS", "32"))

    # install the KV hook (sdpa patch, or eager registry for Gemma-2 to keep softcap)
    if is_gemma:
        from transformers.models.gemma2 import modeling_gemma2 as G
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        _oe = G.eager_attention_forward
        def _msq(module: torch.nn.Module, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *a, **kw) -> tuple:
            if _S["on"]: k, v = _kv_quant(k, v)
            return _oe(module, q, k, v, *a, **kw)
        ALL_ATTENTION_FUNCTIONS["eager"] = _msq
        BOS = tok.bos_token_id
    else:
        _real = F.scaled_dot_product_attention
        def _patch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *a, **kw) -> torch.Tensor:
            if _S["on"]: k, v = _kv_quant(k, v)
            return _real(q, k, v, *a, **kw)
        F.scaled_dot_product_attention = _patch
        BOS = None

    try: _wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception: _wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in _wt["text"] if t.strip()), return_tensors="pt").input_ids
    targets = [(n, m) for n, m in model.named_modules() if is_t(n, m)]
    master = {m: m.weight.detach().to("cpu", copy=True) for _, m in targets}
    def restore() -> None:
        for _, m in targets: m.weight.data.copy_(master[m].to(DEV))

    # weight-importance calibration: H_w = E[x^2] per input channel (deployed greedy+H encoder).
    Hw = {m: [torch.zeros(m.weight.shape[1], device=DEV), 0] for _, m in targets}
    def mkhook(m: torch.nn.Linear) -> Callable:
        def h(mod: torch.nn.Module, inp: tuple) -> None:
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            Hw[m][0] += x.pow(2).sum(0); Hw[m][1] += x.shape[0]
        return h
    hs = [m.register_forward_pre_hook(mkhook(m)) for _, m in targets]
    NCAL = int(os.environ.get("SSNF_CALIB_W", "1"))         # calibration windows for weight importance
    with torch.no_grad():
        _S["on"] = False
        for w in range(NCAL):
            b = w * MAXLEN
            if b + 16 >= ids.size(1): break
            model(ids[:, b:b + MAXLEN].to(DEV))
    for hk in hs: hk.remove()
    Hw = {m: (s / max(c, 1)) for m, (s, c) in Hw.items()}   # [K_in] E[x^2]

    WOFFS = tuple(int(x) for x in os.environ.get("SSNF_W_SCALE_OFFS", "0").split(","))   # 0 = deployed (no search)
    WGS = int(os.environ.get("SSNF_W_GS", "8"))   # weight group size (8 -> 5.625b deployed, 16 -> 5.44b)
    def quant_w() -> None:   # greedy + importance + per-block scale-offset search
        for _, m in targets:
            W = master[m].to(DEV); OUT, Kin = W.shape
            Hf = Hw[m].reshape(1, Kin).expand(OUT, Kin)
            m.weight.data.copy_(wq_scale(W, 3, WGS, Hf, radius=8, offs=WOFFS).to(m.weight.dtype))

    _S["on"] = False; bf = ppl(model, ids, BOS)
    print(f"\n# SSNF DEPLOY (−rotation) PPL — {MODEL} ({attn})")
    print(f"# wikitext-2 ({ids.size(1):,} tok) BF16={bf:.4f} | {MW}w {MAXLEN}/{STRIDE}")
    print(f"# W=lb3/g{WGS}+greedy+importance+scale-search{WOFFS} K=act-weighted(H_K,lb3/g{_S['kgs']},r{_S['kr']}) V=uniform(lb4/g{_S['vgs']},r{_S['vr']}) no-rot")
    bpe = lambda uu, g: (8 - uu) + uu / g + 8 / 32
    print(f"# bpe: W={bpe(3,WGS):.2f} K={bpe(3,_S['kgs']):.2f} V={bpe(4,_S['vgs']):.2f} KVavg={(bpe(3,_S['kgs'])+bpe(4,_S['vgs']))/2:.2f}\n")
    def cell(p: float) -> str: return f"{p:.4f} ({(p/bf-1)*100:+.2f}%)"
    restore(); _S["on"] = True; p_kv = ppl(model, ids, BOS); _S["on"] = False
    quant_w(); _S["on"] = True; p_wkv = ppl(model, ids, BOS); _S["on"] = False; restore()
    print(f"SSNF-deploy  KV-only {cell(p_kv)} | W+KV {cell(p_wkv)}", flush=True)


if __name__ == "__main__":
    main()

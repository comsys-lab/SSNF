"""Headline QSNR + wikitext-2 W+KV PPL: SSNF (W greedy+H u3/gs8 measured at offs={0} and {-1,0}, K act-mean u3/gs16 r0, V arith u4/gs32 r0) vs MXINT8..4 baselines (per-32 E8M0, n-bit, bpe = n + 0.25).
K/V QSNR is accumulated over the wikitext-2 KV-only forward pass with BF16 weights; encoders ssnf_hw / wq_scale / ppl imported verbatim from wkv_v2_ppl.py.
Run: SSNF_MODEL=<id> SSNF_HK=calib/H_K_<model>.pt CUDA_VISIBLE_DEVICES=0 python eval_accuracy/measure.py
"""
from __future__ import annotations
import os, sys, torch
from typing import Callable
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling modules (wkv_v2_ppl)
from wkv_v2_ppl import ssnf_hw, wq_scale, ppl, _S, LINEAR_KEYS, MW  # same encoders as deployed driver

DEV = "cuda"
MXN = [8, 7, 6, 5, 4]


def qsnr_db(sig: float, noise: float) -> float:
    """Signal-to-quantization-noise ratio in dB from accumulated energies."""
    return 10.0 * torch.log10(torch.tensor(max(sig, 1e-30) / max(noise, 1e-30))).item()


def mxint_n(x: torch.Tensor, n: int) -> torch.Tensor:   # OCP MXINT-n: per-32 block, E8M0 = 2^(floor(log2 amax) - (n-2)), signed n-bit.
    # OCP-standard floor scale (== hessian/kv_quant.py::mxint_n and SSNF s_base), matches the paper
    # motivation QSNR table. (Earlier revisions used a ceil no-clip scale, ~0.5-1 dB lower at low bits.)
    xf = x.reshape(-1, 32).float(); am = xf.abs().amax(-1, keepdim=True).clamp(min=1e-30)
    qmax = (1 << (n - 1)) - 1
    s = torch.exp2(torch.floor(torch.log2(am)) - float(n - 2))
    return (torch.round(xf / s).clamp(-qmax, qmax) * s).reshape(x.shape).to(x.dtype)


def main() -> None:
    """Report QSNR and W+KV PPL for SSNF (both weight offs) and each MXINTn."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    MODEL = os.environ["SSNF_MODEL"]; HKF = os.environ["SSNF_HK"]

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV).eval()
    d = torch.load(HKF, map_location=DEV)
    _S["HK"] = d["H_K"].to(torch.float32).to(DEV)                 # [L,Hkv,D]
    _S["L"], _S["Hkv"] = _S["HK"].shape[0], _S["HK"].shape[1]
    _S["kgs"] = 16; _S["vgs"] = 32                               # SSNF K 5.44 bpe, V 4.38 bpe
    WGS = 8                                                       # SSNF W 5.63 bpe

    # KV quant hook: mode "ssnf" -> deployed encoders; mode "mxN" -> MXINT-N on both K and V.
    kv = {}   # per-mode running {ks,kn,vs,vn}
    _real = F.scaled_dot_product_attention
    def _kv_quant(k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mode = _S["mode"]
        if mode == "ssnf":
            layer = _S["i"] % _S["L"]; _S["i"] += 1
            HKl = _S["HK"][layer]
            B, nh, Sq, D = k.shape; Hkv = _S["Hkv"]
            hidx = torch.arange(nh, device=k.device) // (nh // Hkv)
            Hfull = HKl[hidx].reshape(1, nh, 1, D).expand(B, nh, Sq, D)
            kq = ssnf_hw(k.float(), 3, _S["kgs"], Hfull, radius=0)
            vq = ssnf_hw(v.float(), 4, _S["vgs"], None, radius=0)
        else:
            n = _S["mxn"]; kq = mxint_n(k.float(), n); vq = mxint_n(v.float(), n)
        if _S.get("qsnr"):
            a = kv[mode]; kf, vf = k.float(), v.float()
            a["ks"] += kf.pow(2).sum().item(); a["kn"] += (kf - kq).pow(2).sum().item()
            a["vs"] += vf.pow(2).sum().item(); a["vn"] += (vf - vq).pow(2).sum().item()
        return kq.to(k.dtype), vq.to(v.dtype)
    def _patch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *a, **kw) -> torch.Tensor:
        if _S["on"]: k, v = _kv_quant(k, v)
        return _real(q, k, v, *a, **kw)
    F.scaled_dot_product_attention = _patch

    try: _wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception: _wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in _wt["text"] if t.strip()), return_tensors="pt").input_ids

    def is_t(n: str, m: torch.nn.Module) -> bool: return isinstance(m, torch.nn.Linear) and any(p in n for p in LINEAR_KEYS)
    targets = [(n, m) for n, m in model.named_modules() if is_t(n, m)]
    master = {m: m.weight.detach().to("cpu", copy=True) for _, m in targets}
    def restore() -> None:
        for _, m in targets: m.weight.data.copy_(master[m].to(DEV))

    # weight-Hessian H_w = E[x^2] per input channel (SSNF greedy+H encoder), one calib window
    Hw = {m: [torch.zeros(m.weight.shape[1], device=DEV), 0] for _, m in targets}
    def mkhook(m: torch.nn.Linear) -> Callable:
        def h(mod: torch.nn.Module, inp: tuple) -> None:
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            Hw[m][0] += x.pow(2).sum(0); Hw[m][1] += x.shape[0]
        return h
    hs = [m.register_forward_pre_hook(mkhook(m)) for _, m in targets]
    with torch.no_grad():
        _S["on"] = False
        model(ids[:, :2048].to(DEV))
    for hk in hs: hk.remove()
    Hw = {m: (s / max(c, 1)) for m, (s, c) in Hw.items()}

    def w_qsnr_ssnf(offs: tuple[int, ...]) -> float:   # quant all weights (SSNF), return aggregate weight QSNR(dB)
        sig = 0.0; noise = 0.0
        for _, m in targets:
            W = master[m].to(DEV); OUT, Kin = W.shape
            Hf = Hw[m].reshape(1, Kin).expand(OUT, Kin)
            Wq = wq_scale(W, 3, WGS, Hf, radius=8, offs=offs)
            sig += W.float().pow(2).sum().item(); noise += (W.float() - Wq.float()).pow(2).sum().item()
            m.weight.data.copy_(Wq.to(m.weight.dtype))
        return qsnr_db(sig, noise)

    def w_qsnr_mxint(n: int) -> float:     # quant all weights (MXINT-n), return aggregate weight QSNR(dB)
        sig = 0.0; noise = 0.0
        for _, m in targets:
            W = master[m].to(DEV); Wq = mxint_n(W, n)
            sig += W.float().pow(2).sum().item(); noise += (W.float() - Wq.float()).pow(2).sum().item()
            m.weight.data.copy_(Wq.to(m.weight.dtype))
        return qsnr_db(sig, noise)

    def kv_qsnr(mode: str) -> tuple[float, float]:       # one KV-only forward pass (BF16 weights); return (K dB, V dB)
        kv[mode] = {"ks": 0.0, "kn": 0.0, "vs": 0.0, "vn": 0.0}
        _S["mode"] = mode; _S["on"] = True; _S["qsnr"] = True
        ppl(model, ids)      # forward only; PPL value unused here
        _S["qsnr"] = False; _S["on"] = False
        a = kv[mode]; return qsnr_db(a["ks"], a["kn"]), qsnr_db(a["vs"], a["vn"])

    bssnf = lambda uu, g: (8 - uu) + uu / g + 8 / 32
    print(f"\n# HEADLINE QSNR+PPL — {MODEL}", flush=True)
    print(f"# wikitext-2 ({ids.size(1):,} tok), {min(MW, (ids.size(1) - 2048) // 1024 + 1)}w 2048/1024, sdpa, no-rot", flush=True)
    print(f"# SSNF: W u3/gs8({bssnf(3,8):.2f}b) K act-mean u3/gs16({bssnf(3,16):.2f}b) V arith u4/gs32({bssnf(4,32):.2f}b) KVavg {(bssnf(3,16)+bssnf(4,32))/2:.2f}b", flush=True)
    print(f"# MXINTn: per-32 E8M0, signed n-bit, bpe = n+0.25\n", flush=True)

    _S["on"] = False; bf = ppl(model, ids)
    cell = lambda p: f"{p:.4f} ({(p/bf-1)*100:+.2f}%)"
    print(f"BF16 PPL = {bf:.4f}\n", flush=True)

    # ---------------- QSNR ----------------
    if os.environ.get("HEADLINE_SKIP_QSNR", "0") != "1":   # QSNR costs 6 extra full passes
        print("## QSNR (dB)   [W | K | V | avg]", flush=True)
        # SSNF K/V (independent of weight scale search); weight QSNR for both offs
        k_s, v_s = kv_qsnr("ssnf")
        for offs, tag in [((0,), "SSNF W offs={0}   "), ((-1, 0), "SSNF W offs={-1,0}")]:
            wq = w_qsnr_ssnf(offs); restore()
            avg = (wq + k_s + v_s) / 3
            print(f"{tag} | W {wq:6.2f} | K {k_s:6.2f} | V {v_s:6.2f} | avg {avg:6.2f}", flush=True)
        for n in MXN:
            wq = w_qsnr_mxint(n); restore()
            _S["mxn"] = n; kq, vq = kv_qsnr(f"mx{n}")
            avg = (wq + kq + vq) / 3
            print(f"MXINT{n} ({n + 0.25:.2f}b)     | W {wq:6.2f} | K {kq:6.2f} | V {vq:6.2f} | avg {avg:6.2f}", flush=True)

    # ---------------- PPL (W+KV) ----------------
    print("\n## PPL — W+KV (weights + KV quantized)", flush=True)
    print(f"BF16                  {bf:.4f}", flush=True)
    for offs, tag in [((0,), "SSNF offs={0}   "), ((-1, 0), "SSNF offs={-1,0}")]:
        for _, m in targets:
            W = master[m].to(DEV); OUT, Kin = W.shape
            Hf = Hw[m].reshape(1, Kin).expand(OUT, Kin)
            m.weight.data.copy_(wq_scale(W, 3, WGS, Hf, radius=8, offs=offs).to(m.weight.dtype))
        _S["mode"] = "ssnf"; _S["on"] = True; p = ppl(model, ids); _S["on"] = False; restore()
        print(f"SSNF W+KV {tag}  {cell(p)}", flush=True)
    for n in MXN:
        for _, m in targets: m.weight.data.copy_(mxint_n(master[m].to(DEV), n).to(m.weight.dtype))
        _S["mode"] = f"mx{n}"; _S["mxn"] = n; _S["on"] = True; p = ppl(model, ids); _S["on"] = False; restore()
        print(f"MXINT{n} ({n + 0.25:.2f}b)     {cell(p)}", flush=True)


if __name__ == "__main__":
    main()

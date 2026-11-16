"""SSNF (l_b x g_size) sweep -- WikiText-2 PPL with exactly one plane of {w, k, v} quantized (configuration search, paper Section 4), same encoders as the paper numbers (wkv_v2_ppl).
Fixed protocol: SSNF_W_SCALE_OFFS=0, weight shared-candidate search radius = 1 << l_b, SSNF_KV_RADIUS=0, calibration = wikitext-2-raw train 128 x 2048, PPL = test 2048 / stride 1024, full window unless PPL_WINDOWS is set.
Run: SSNF_SWEEP_PLANE=k SSNF_SWEEP_LB=1,2,3,4,5,6,7 SSNF_SWEEP_GS=2,4,8,16,32 SSNF_MODEL=NousResearch/Meta-Llama-3.1-8B SSNF_HK=calib/H_K_metallama318b.pt HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python eval_accuracy/sweep_ubgs.py
SSNF_SWEEP_SELFTEST=1 (same command) runs the encoder unit test only, no model load.
"""
from __future__ import annotations

import math
import os
import sys
import time
from typing import Callable

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sibling modules (wkv_v2_ppl)

# Full-window by default: the paper table is full-window, so the sweep must match it.
# (Set PPL_WINDOWS to override, e.g. 20 for the pre-flight check.)
os.environ.setdefault("PPL_WINDOWS", "100000")
os.environ.setdefault("SSNF_W_SCALE_OFFS", "0")
os.environ.setdefault("SSNF_KV_RADIUS", "0")

import wkv_v2_ppl as W  # noqa: E402  -- reads PPL_MAXLEN / PPL_STRIDE / PPL_WINDOWS at import

DEV = "cuda"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


def bpe(l_b: int, g_size: int) -> float:
    """Nominal average bits per element (Algorithm 1) -- the FORMAT cost."""
    return (8 - l_b) + l_b / g_size + 8 / 32


def bpe_aligned(l_b: int, g_size: int) -> float:
    """DEPLOYED bits per element: the shared field rounded up to a whole byte per 32-value block, as the packer and the KV slot geometry store it."""
    upper = (8 - l_b) * 4                      # (8-lb) bits x 32 values / 8
    shared = math.ceil((32 // g_size) * l_b / 8)   # shared fields, rounded up to whole bytes
    return (upper + shared + 1) * 8 / 32      # +1 byte for the E8M0 block scale


# ---------------------------------------------------------------------------
# self-test: ssnf_hw must equal a direct transcription of the format definition
# ---------------------------------------------------------------------------
def _reference(x: torch.Tensor, l_b: int, g_size: int, H: torch.Tensor | None = None) -> torch.Tensor:
    """Straight transcription of Algorithm 1 + Algorithm 3 (arithmetic/weighted mean, radius 0)."""
    shp = x.shape
    xf = x.reshape(-1, 32).float()
    B = xf.shape[0]
    s = torch.exp2(torch.floor(torch.log2(xf.abs().amax(-1, keepdim=True).clamp(min=1e-30))) - 6.0)
    v = xf / s                                            # scaled values v_i
    hmax = (1 << (7 - l_b)) - 1                            # h_b = 8 - l_b, signed symmetric
    lmin, lmax = -(1 << (l_b - 1)), (1 << (l_b - 1)) - 1    # shared field, signed asymmetric
    n_group = 32 // g_size
    h = torch.round(v / (1 << l_b)).clamp(-hmax, hmax)     # per-value upper field
    rho = (v - h * (1 << l_b)).reshape(B, n_group, g_size)          # residual
    Hg = (torch.ones_like(xf) if H is None else H.reshape(-1, 32).float()).reshape(B, n_group, g_size)
    l = torch.round((Hg * rho).sum(2) / Hg.sum(2).clamp(min=1e-30)).clamp(lmin, lmax)
    return ((h * (1 << l_b) + l.repeat_interleave(g_size, 1)) * s).reshape(shp).to(x.dtype)


def selftest() -> bool:
    """Unit-test ssnf_hw against _reference over shapes, l_b, g_size, and H weighting."""
    torch.manual_seed(0)
    bad = 0
    for shape in [(64, 32), (7, 4, 32), (129, 32)]:
        x = torch.randn(*shape, device=DEV) * 3.0
        H = torch.rand(*shape, device=DEV) + 0.05
        for l_b in range(1, 8):
            for g_size in (1, 2, 4, 8, 16, 32):
                for Hf in (None, H):
                    a = W.ssnf_hw(x, l_b, g_size, Hf, radius=0)
                    b = _reference(x, l_b, g_size, Hf)
                    d = (a - b).abs().max().item()
                    if d > 0:
                        bad += 1
                        print(f"  MISMATCH lb={l_b} gs={g_size} H={'w' if Hf is not None else '-'} "
                              f"shape={shape} maxdiff={d:.3e}")
    print("selftest:", "PASS" if bad == 0 else f"FAIL ({bad} mismatches)")
    return bad == 0


# ---------------------------------------------------------------------------
# plane-isolated KV hook
# ---------------------------------------------------------------------------
_P = {"plane": None, "lb": None, "gs": None, "on": False}


def _kv_hook(k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize exactly one of key/value; the other passes through untouched."""
    layer = W._S["i"] % W._S["L"]
    W._S["i"] += 1
    if _P["plane"] == "k":
        B, nh, Sq, D = k.shape
        Hkv = W._S["Hkv"]
        hidx = torch.arange(nh, device=k.device) // (nh // Hkv)     # head -> kv-head
        Hf = W._S["HK"][layer][hidx].reshape(1, nh, 1, D).expand(B, nh, Sq, D)
        return W.ssnf_hw(k.float(), _P["lb"], _P["gs"], Hf, radius=0).to(k.dtype), v
    if _P["plane"] == "v":
        return k, W.ssnf_hw(v.float(), _P["lb"], _P["gs"], None, radius=0).to(v.dtype)
    return k, v


def main() -> None:
    """Run the per-plane (l_b, g_size) PPL sweep and append results to a resumable CSV."""
    if os.environ.get("SSNF_SWEEP_SELFTEST", "0") != "0":
        sys.exit(0 if selftest() else 1)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    plane = os.environ["SSNF_SWEEP_PLANE"].lower()
    assert plane in ("w", "k", "v"), plane
    LBS = [int(x) for x in os.environ.get("SSNF_SWEEP_LB", "1,2,3,4,5,6,7").split(",")]
    GSS = [int(x) for x in os.environ.get("SSNF_SWEEP_GS", "2,4,8,16,32").split(",")]
    MODEL = os.environ.get("SSNF_MODEL", "NousResearch/Meta-Llama-3.1-8B")
    NCAL = int(os.environ.get("SSNF_CALIB_W", "128"))
    CSPLIT = os.environ.get("SSNF_CALIB_SPLIT", "train")
    tag = os.environ.get("SSNF_TAG") or MODEL.split("/")[-1].replace(".", "").lower()
    csv = os.path.join(OUT_DIR, f"{plane}_{tag}.csv")
    os.makedirs(OUT_DIR, exist_ok=True)

    done = set()
    if os.path.exists(csv):                                   # resume: skip finished configs
        for line in open(csv):
            f = line.split(",")
            if len(f) > 2 and f[0] == plane:
                done.add((int(f[1]), int(f[2])))

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV).eval()

    d = torch.load(os.environ["SSNF_HK"], map_location=DEV)
    W._S["HK"] = d["H_K"].to(torch.float32).to(DEV)
    W._S["L"], W._S["Hkv"] = W._S["HK"].shape[0], W._S["HK"].shape[1]

    _real = F.scaled_dot_product_attention

    def _patch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *a, **kw) -> torch.Tensor:
        if _P["on"]:
            k, v = _kv_hook(k, v)
        return _real(q, k, v, *a, **kw)
    F.scaled_dot_product_attention = _patch

    def _load(split: str) -> torch.Tensor:
        try:
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        except Exception:
            ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
        return tok("\n\n".join(t for t in ds["text"] if t.strip()), return_tensors="pt").input_ids

    ids = _load("test")                                       # evaluation stream

    def is_t(n: str, m: torch.nn.Module) -> bool:
        return isinstance(m, torch.nn.Linear) and any(p in n for p in W.LINEAR_KEYS)
    targets = [(n, m) for n, m in model.named_modules() if is_t(n, m)]
    master = {m: m.weight.detach().to("cpu", copy=True) for _, m in targets}

    def restore() -> None:
        for _, m in targets:
            m.weight.data.copy_(master[m].to(DEV))

    # ---- weight importance E[x^2], from the TRAIN split (paper §Methodology), computed once ----
    Hw = None
    if plane == "w":
        cal = _load(CSPLIT)
        acc = {m: [torch.zeros(m.weight.shape[1], device=DEV, dtype=torch.float64), 0] for _, m in targets}

        def mk(m: torch.nn.Linear) -> Callable:
            def h(mod: torch.nn.Module, inp: tuple) -> None:
                x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                acc[m][0] += x.pow(2).sum(0).double()
                acc[m][1] += x.shape[0]
            return h
        hs = [m.register_forward_pre_hook(mk(m)) for _, m in targets]
        nseg = min(NCAL, cal.size(1) // W.MAXLEN)
        t0 = time.time()
        with torch.no_grad():
            _P["on"] = False
            for i in range(nseg):
                model(cal[:, i * W.MAXLEN:(i + 1) * W.MAXLEN].to(DEV))
        for h in hs:
            h.remove()
        Hw = {m: (s / max(c, 1)).float() for m, (s, c) in acc.items()}
        print(f"# weight calib: {CSPLIT} split, {nseg}x{W.MAXLEN} tok, {time.time()-t0:.0f}s", flush=True)

    WOFFS = tuple(int(x) for x in os.environ["SSNF_W_SCALE_OFFS"].split(","))

    _P["on"] = False
    t0 = time.time()
    bf = W.ppl(model, ids, None)
    print(f"# {MODEL} plane={plane} | wikitext-2 test {ids.size(1):,} tok, "
          f"{W.MAXLEN}/{W.STRIDE}, window cap {W.MW} | BF16 PPL = {bf:.4f} ({time.time()-t0:.0f}s)",
          flush=True)
    print(f"# fixed: W_SCALE_OFFS={WOFFS} KV_RADIUS=0 W_search_radius=1<<l_b calib={CSPLIT}/{NCAL}",
          flush=True)

    new = not os.path.exists(csv)
    fh = open(csv, "a")
    if new:
        fh.write("plane,lb,gs,bpe_nominal,bpe_aligned,ppl,delta_pct,bf16,secs\n")
        fh.flush()

    for l_b in LBS:
        for g_size in GSS:
            if (l_b, g_size) in done:
                print(f"  skip {plane} lb={l_b} gs={g_size} (already in csv)", flush=True)
                continue
            t0 = time.time()
            if plane == "w":
                for _, m in targets:
                    Wt = master[m].to(DEV)
                    OUT, Kin = Wt.shape
                    Hf = Hw[m].reshape(1, Kin).expand(OUT, Kin)
                    m.weight.data.copy_(
                        W.wq_scale(Wt, l_b, g_size, Hf, radius=(1 << l_b), offs=WOFFS).to(m.weight.dtype))
                p = W.ppl(model, ids, None)
                restore()
            else:
                _P.update(plane=plane, on=True); _P["lb"] = l_b; _P["gs"] = g_size
                p = W.ppl(model, ids, None)
                _P["on"] = False
            dt = time.time() - t0
            b, ba = bpe(l_b, g_size), bpe_aligned(l_b, g_size)
            fh.write(f"{plane},{l_b},{g_size},{b:.4f},{ba:.4f},{p:.6f},{(p/bf-1)*100:.4f},{bf:.6f},{dt:.1f}\n")
            fh.flush()
            print(f"  {plane} l_b={l_b} g_size={g_size:2d} bpe={b:5.3f}/{ba:5.3f}  PPL {p:.4f} "
                  f"({(p/bf-1)*100:+.2f}%)  [{dt:.0f}s]", flush=True)
    fh.close()
    print(f"# wrote {csv}", flush=True)


if __name__ == "__main__":
    main()

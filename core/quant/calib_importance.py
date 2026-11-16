"""KV importance calibration: per-(layer, kv-head, channel) H_K (post-RoPE E[q^2], GQA-group
mean, via an F.scaled_dot_product_attention patch) and H_V (||W_O column||^2, GQA-group mean)
for the importance-weighted additive shared KV encoder.
Run: HF_HUB_OFFLINE=1 SSNF_MODEL=<hf_id> python quant/calib_importance.py
Out: calib/H_K_<modeltag>.pt = {"H_K":[L,Hkv,D], "H_V":[L,Hkv,D], "meta":{...}}
"""
import os, torch
import torch.nn.functional as F

DEV = "cuda"
MAXLEN = int(os.environ.get("CALIB_MAXLEN", 2048))
NSAMPLES = int(os.environ.get("CALIB_NSAMPLES", 128))
MODEL = os.environ.get("SSNF_MODEL", "NousResearch/Meta-Llama-3.1-8B")

# ---- post-RoPE q capture via SDPA patch (repo-standard hook, comparison_kv_ppl.py) ----
_state = {"buf": None, "ntok": 0, "call": 0, "nL": 0, "on": False}
_real_sdpa = F.scaled_dot_product_attention


def _patched_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *a, **kw) -> torch.Tensor:
    if _state["on"]:
        l = _state["call"] % _state["nL"]
        _state["call"] += 1
        # q: [B, Hq, L, D] post-RoPE. accumulate sum over (B, L) of q^2 -> [Hq, D].
        q2 = (q.float() ** 2).sum(dim=(0, 2))              # [Hq, D]
        _state["buf"][l] += q2
        if l == 0:
            _state["ntok"] += q.shape[0] * q.shape[2]      # B*L, counted once/forward
    return _real_sdpa(q, k, v, *a, **kw)


F.scaled_dot_product_attention = _patched_sdpa


@torch.no_grad()
def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV).eval()
    cfg = model.config
    L_, Hq = cfg.num_hidden_layers, cfg.num_attention_heads
    Hkv = cfg.num_key_value_heads
    D = getattr(cfg, "head_dim", cfg.hidden_size // Hq)
    G = Hq // Hkv
    print(f"# {MODEL}: L={L_} Hq={Hq} Hkv={Hkv} D={D} G={G} | calib {NSAMPLES}x{MAXLEN} wikitext-2", flush=True)

    # ---- calibration corpus: wikitext-2 (raw) train, NSAMPLES chunks of MAXLEN ----
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    except Exception:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    ids = tok("\n\n".join(t for t in ds["text"] if t.strip()), return_tensors="pt").input_ids[0]
    nchunk = min(NSAMPLES, ids.numel() // MAXLEN)
    print(f"# corpus {ids.numel():,} tok -> {nchunk} chunks", flush=True)

    _state["buf"] = torch.zeros(L_, Hq, D, device=DEV, dtype=torch.float32)
    _state["nL"] = L_

    # ---- forward pass: accumulate post-RoPE q^2 per (layer, head, channel) ----
    _state["on"] = True
    for i in range(nchunk):
        _state["call"] = 0                                 # reset layer bucket each forward
        inp = ids[i * MAXLEN:(i + 1) * MAXLEN].unsqueeze(0).to(DEV)
        model(inp)
        if (i + 1) % 16 == 0:
            print(f"  {i+1}/{nchunk}", flush=True)
    _state["on"] = False

    Eq2 = _state["buf"] / max(_state["ntok"], 1)           # [L, Hq, D]  E[q^2]

    # ---- H_K: aggregate the G query heads that share each kv-head (mean over group) ----
    H_K = Eq2.reshape(L_, Hkv, G, D).mean(dim=2)           # [L, Hkv, D]

    # ---- H_V: ||W_O[:, hq*D+c]||^2 aggregated over the same GQA group (static) ----
    H_V = torch.zeros(L_, Hkv, D, device=DEV, dtype=torch.float32)
    for l in range(L_):
        WO = model.model.layers[l].self_attn.o_proj.weight.float()   # [hidden, Hq*D]
        colnorm2 = (WO ** 2).sum(dim=0).reshape(Hq, D)               # [Hq, D]
        H_V[l] = colnorm2.reshape(Hkv, G, D).mean(dim=1)

    tag = MODEL.split("/")[-1].replace(".", "").replace("-", "").lower()
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "calib", f"H_K_{tag}.pt")
    torch.save({"H_K": H_K.cpu(), "H_V": H_V.cpu(),
                "meta": {"model": MODEL, "L": L_, "Hkv": Hkv, "D": D, "G": G,
                         "nchunk": nchunk, "maxlen": MAXLEN, "ntok": _state["ntok"]}}, out)

    # ---- sanity: importance must be non-flat (K should have outlier channels) ----
    def spread(H: torch.Tensor) -> float:
        m = H.mean(-1, keepdim=True)
        return (H.amax(-1) / m.squeeze(-1).clamp(min=1e-30)).mean().item()
    print(f"# saved {out}", flush=True)
    print(f"#   H_K spread(max/mean over channels, layer-head avg) = {spread(H_K):.2f}", flush=True)
    print(f"#   H_V spread = {spread(H_V):.2f}", flush=True)
    print(f"#   H_K range [{H_K.min():.3e}, {H_K.max():.3e}]  H_V range [{H_V.min():.3e}, {H_V.max():.3e}]", flush=True)


if __name__ == "__main__":
    main()

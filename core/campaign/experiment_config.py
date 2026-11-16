"""Experiment configuration for the accuracy-task campaign (models, formats, fixed settings).

Edit here; gen_jobs.py and frontier.py import these constants. The mapping mirrors README §3.4
(model tag -> HF checkpoint id -> calibration files in calib/).
"""

# tag -> (hf id, H_K file for the SSNF KV encoder, H_w file for the SSNF weight encoder)
BASE: dict[str, tuple[str, str, str]] = {
    "llama31": ("NousResearch/Meta-Llama-3.1-8B", "H_K_metallama318b.pt", "H_w_llama31.pt"),
    "mistral": ("mistralai/Mistral-7B-v0.3", "H_K_mistral7bv03.pt", "H_w_mistral.pt"),
    "llama2":  ("NousResearch/Llama-2-7b-hf", "H_K_llama27bhf.pt", "H_w_llama2.pt"),
}

# tag -> (hf id, H_K file, H_w file, LongBench max input length)
CHAT: dict[str, tuple[str, str, str, int]] = {
    "llama31i": ("NousResearch/Meta-Llama-3.1-8B-Instruct", "H_K_metallama318binstruct.pt", "H_w_llama31i.pt", 4096),
    "mistrali": ("mistralai/Mistral-7B-Instruct-v0.3", "H_K_mistral7binstructv03.pt", "H_w_mistrali.pt", 8192),
    "llama2c":  ("NousResearch/Llama-2-7b-chat-hf", "H_K_llama27bchathf.pt", "H_w_llama2c.pt", 3456),
}

# Fixed environment for every job: no encoder search (deployed), offline datasets.
COMMON: str = "HF_HUB_OFFLINE=1 SSNF_W_SCALE_OFFS=0 SSNF_KV_RADIUS=0"
# Deployed KV configuration: K l_b3/g16, V l_b4/g32.
SSNF_KV: str = "SSNF_K_LB=3 SSNF_K_GS=16 SSNF_V_LB=4 SSNF_V_GS=32"

# vLLM CONFIG name -> zero-shot CONFIG name (same 9 formats, different driver vocabulary)
FORMATS: list[tuple[str, str]] = [
    ("bf16", "bf16"), ("mxint8_fq", "mxint8"),
    ("mxfp8_fq", "mxfp8_e4m3"), ("mxfp8_e5m2_fq", "mxfp8_e5m2"),
    ("mxfp6_fq", "mxfp6_e2m3"), ("mxfp6_e3m2_fq", "mxfp6_e3m2"),
    ("mxfp4", "mxfp4"), ("nvfp4", "nvfp4"), ("ssnf_fq", "ssnf"),
]

# E4M3 convention pair (consistency phase): mxfp8_fq quantizes weights through the packed CUDA
# path (saturates at the true E4M3 max 448) while every other format uses the fake-quant codec
# (saturates at 480); mxfp8_fqw reruns E4M3 with fake-quant weights so the E4M3-vs-E5M2 comparison
# holds the convention fixed. Both are kept so the size of the effect can be reported.
CONSISTENCY_GSM8K: list[str] = ["mxfp8_fq", "mxfp8_fqw"]
CONSISTENCY_LB: str = "mxfp8_fqw"       # LongBench: the packed-E4M3 numbers come from the B3 phase

# GSM8K keeps its 12-task-independent baselines from the earlier campaign; only these are re-run:
#   ssnf_fq (W 5.63 / K 5.44 / V 4.38 configuration), mxint8_fq (codec on the OCP floor scale),
#   and the two new formats.
GSM8K_REDO: list[str] = ["ssnf_fq", "mxint8_fq", "mxfp8_e5m2_fq", "mxfp6_e3m2_fq"]

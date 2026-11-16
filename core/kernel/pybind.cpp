// kernel/pybind.cpp  —  Python binding for the SSNF-signed kernels.
//
// Registers the four host launchers (defined in the .cu files) under the `ssnf`
// namespace via TORCH_LIBRARY, so after `import ssnf_cuda` they are reachable as
//   torch.ops.ssnf.wonly_gemm_tc / wa_gemm / kv_decode_paged / kv_append_paged_fused
// exactly as ssnf_lib.ops calls them. (TORCH_LIBRARY infers each schema from the
// C++ signature; inference-only, no autograd registered.)

#include <torch/extension.h>


// defined in wa_gemm.cu
torch::Tensor wonly_gemm_tc_cuda(
    torch::Tensor X, torch::Tensor scale_exp, torch::Tensor upper_cm, torch::Tensor shared_cm,
    int64_t M, int64_t OUT, int64_t K, int64_t NB, int64_t l_b, int64_t g_size);
torch::Tensor wonly_dequant_bf16_cuda(
    torch::Tensor scale_exp, torch::Tensor upper_cm, torch::Tensor shared_cm,
    int64_t OUT, int64_t K, int64_t NB, int64_t l_b, int64_t g_size);
torch::Tensor mxint8_gemm_fused_skinny_cuda(
    torch::Tensor X, torch::Tensor scale_exp, torch::Tensor qweight_cm,
    int64_t M, int64_t OUT, int64_t K, int64_t NB);
torch::Tensor mxfp8_gemm_fused_skinny_cuda(
    torch::Tensor X, torch::Tensor scale_exp, torch::Tensor qweight_cm,
    int64_t M, int64_t OUT, int64_t K, int64_t NB);
torch::Tensor mxfp8_dequant_bf16_cuda(
    torch::Tensor scale_exp, torch::Tensor qweight_cm, int64_t OUT, int64_t K, int64_t NB);
torch::Tensor mxfp6_gemm_fused_skinny_cuda(
    torch::Tensor X, torch::Tensor scale_exp, torch::Tensor qweight_cm,
    int64_t M, int64_t OUT, int64_t K, int64_t NB);
torch::Tensor mxfp6_dequant_bf16_cuda(
    torch::Tensor scale_exp, torch::Tensor qweight_cm, int64_t OUT, int64_t K, int64_t NB);

// defined in kv_attention.cu
torch::Tensor kv_decode_paged_cuda(
    torch::Tensor q, torch::Tensor cache, torch::Tensor block_table,
    torch::Tensor seq_lens, int64_t B, int64_t H, int64_t Hkv, int64_t D,
    int64_t NB, int64_t l_b, int64_t g_size, int64_t UB, int64_t SB,
    int64_t block_size, int64_t slot_bytes, int64_t per_head);
torch::Tensor kv_decode_paged_asym_cuda(
    torch::Tensor q, torch::Tensor cache, torch::Tensor block_table, torch::Tensor seq_lens,
    int64_t B, int64_t H, int64_t Hkv, int64_t D, int64_t NB,
    int64_t ku, int64_t kgs, int64_t KUB, int64_t KSB,
    int64_t vu, int64_t vgs, int64_t VUB, int64_t VSB,
    int64_t block_size, int64_t slot_bytes, int64_t per_head_K);
torch::Tensor mxint8_kv_decode_paged_cuda(
    torch::Tensor q, torch::Tensor cache, torch::Tensor block_table,
    torch::Tensor seq_lens, int64_t B, int64_t H, int64_t Hkv, int64_t D,
    int64_t NB, int64_t block_size, int64_t slot_bytes, int64_t per_head);
torch::Tensor mxfp8_kv_decode_paged_cuda(
    torch::Tensor q, torch::Tensor cache, torch::Tensor block_table,
    torch::Tensor seq_lens, int64_t B, int64_t H, int64_t Hkv, int64_t D,
    int64_t NB, int64_t block_size, int64_t slot_bytes, int64_t per_head);
std::vector<torch::Tensor> mxfp8_kv_write_cuda(
    torch::Tensor X, int64_t H, int64_t L, int64_t D, int64_t NB);
torch::Tensor mxfp6_kv_decode_paged_cuda(
    torch::Tensor q, torch::Tensor cache, torch::Tensor block_table,
    torch::Tensor seq_lens, int64_t B, int64_t H, int64_t Hkv, int64_t D,
    int64_t NB, int64_t block_size, int64_t slot_bytes, int64_t per_head);
std::vector<torch::Tensor> mxfp6_kv_write_cuda(
    torch::Tensor X, int64_t H, int64_t L, int64_t D, int64_t NB);
void mxint8_kv_append_fused_cuda(
    torch::Tensor key, torch::Tensor value, torch::Tensor kv_cache, torch::Tensor slot_mapping,
    int64_t N, int64_t Hk, int64_t D, int64_t NB, int64_t per_head, int64_t S);
void mxfp8_kv_append_fused_cuda(
    torch::Tensor key, torch::Tensor value, torch::Tensor kv_cache, torch::Tensor slot_mapping,
    int64_t N, int64_t Hk, int64_t D, int64_t NB, int64_t per_head, int64_t S);
void mxfp6_kv_append_fused_cuda(
    torch::Tensor key, torch::Tensor value, torch::Tensor kv_cache, torch::Tensor slot_mapping,
    int64_t N, int64_t Hk, int64_t D, int64_t NB, int64_t per_head, int64_t S);
std::vector<torch::Tensor> kv_write_cuda(
    torch::Tensor X, int64_t H, int64_t L, int64_t D, int64_t NB, int64_t l_b, int64_t g_size);
void kv_append_paged_fused_cuda(
    torch::Tensor key, torch::Tensor value, torch::Tensor kv_cache, torch::Tensor slot_mapping,
    int64_t N, int64_t Hk, int64_t D, int64_t NB, int64_t l_b, int64_t g_size, int64_t per_head, int64_t S,
    torch::Tensor H_k, int64_t radius);
void kv_append_paged_fused_asym_cuda(
    torch::Tensor key, torch::Tensor value, torch::Tensor kv_cache, torch::Tensor slot_mapping,
    int64_t N, int64_t Hk, int64_t D, int64_t NB, int64_t ku, int64_t kgs, int64_t vu, int64_t vgs,
    int64_t per_head_K, int64_t S, torch::Tensor H_k, int64_t radius);
// MXFP8-SSNF (E3M4) KV — 1:1 analogs of the INT SSNF KV ops

// defined in mxint8.cu (baseline)
// column-major wide-load variants (matched to the SSNF wide-load kernels)
torch::Tensor mxint8_dequant_bf16_cuda(
    torch::Tensor scale_exp, torch::Tensor qweight_cm, int64_t OUT, int64_t K, int64_t NB);
std::vector<torch::Tensor> mxint8_kv_write_cuda(
    torch::Tensor X, int64_t H, int64_t L, int64_t D, int64_t NB);

TORCH_LIBRARY(ssnf, m) {
    m.def("kv_write(Tensor X, int H, int L, int D, int NB, int l_b, int g_size) -> Tensor[]", &kv_write_cuda);
    m.def("kv_append_paged_fused(Tensor key, Tensor value, Tensor(a!) kv_cache, Tensor slot_mapping, "
          "int N, int Hk, int D, int NB, int l_b, int g_size, int per_head, int S, Tensor H_k, int radius) -> ()",
          &kv_append_paged_fused_cuda);
    m.def("kv_append_paged_fused_asym(Tensor key, Tensor value, Tensor(a!) kv_cache, Tensor slot_mapping, "
          "int N, int Hk, int D, int NB, int ku, int kgs, int vu, int vgs, int per_head_K, int S, "
          "Tensor H_k, int radius) -> ()",
          &kv_append_paged_fused_asym_cuda);
    m.def("mxint8_kv_append_fused(Tensor key, Tensor value, Tensor(a!) kv_cache, Tensor slot_mapping, "
          "int N, int Hk, int D, int NB, int per_head, int S) -> ()", &mxint8_kv_append_fused_cuda);
    m.def("mxfp8_kv_append_fused(Tensor key, Tensor value, Tensor(a!) kv_cache, Tensor slot_mapping, "
          "int N, int Hk, int D, int NB, int per_head, int S) -> ()", &mxfp8_kv_append_fused_cuda);
    m.def("mxfp6_kv_append_fused(Tensor key, Tensor value, Tensor(a!) kv_cache, Tensor slot_mapping, "
          "int N, int Hk, int D, int NB, int per_head, int S) -> ()", &mxfp6_kv_append_fused_cuda);
    m.def("wonly_gemm_tc(Tensor X, Tensor scale_exp, Tensor upper_cm, Tensor shared_cm, "
          "int M, int OUT, int K, int NB, int l_b, int g_size) -> Tensor", &wonly_gemm_tc_cuda);
    m.def("wonly_dequant_bf16(Tensor scale_exp, Tensor upper_cm, Tensor shared_cm, "
          "int OUT, int K, int NB, int l_b, int g_size) -> Tensor", &wonly_dequant_bf16_cuda);
    m.def("mxint8_gemm_fused_skinny(Tensor X, Tensor scale_exp, Tensor qweight_cm, "
          "int M, int OUT, int K, int NB) -> Tensor", &mxint8_gemm_fused_skinny_cuda);
    m.def("mxfp8_gemm_fused_skinny(Tensor X, Tensor scale_exp, Tensor qweight_cm, "
          "int M, int OUT, int K, int NB) -> Tensor", &mxfp8_gemm_fused_skinny_cuda);
    m.def("mxfp8_dequant_bf16(Tensor scale_exp, Tensor qweight_cm, "
          "int OUT, int K, int NB) -> Tensor", &mxfp8_dequant_bf16_cuda);
    m.def("mxfp6_gemm_fused_skinny(Tensor X, Tensor scale_exp, Tensor qweight_cm, "
          "int M, int OUT, int K, int NB) -> Tensor", &mxfp6_gemm_fused_skinny_cuda);
    m.def("mxfp6_dequant_bf16(Tensor scale_exp, Tensor qweight_cm, "
          "int OUT, int K, int NB) -> Tensor", &mxfp6_dequant_bf16_cuda);
    m.def("kv_decode_paged(Tensor q, Tensor cache, Tensor block_table, Tensor seq_lens, "
          "int B, int H, int Hkv, int D, int NB, int l_b, int g_size, int UB, int SB, "
          "int block_size, int slot_bytes, int per_head) -> Tensor",
          &kv_decode_paged_cuda);
    m.def("kv_decode_paged_asym(Tensor q, Tensor cache, Tensor block_table, Tensor seq_lens, "
          "int B, int H, int Hkv, int D, int NB, int ku, int kgs, int KUB, int KSB, "
          "int vu, int vgs, int VUB, int VSB, int block_size, int slot_bytes, int per_head_K) -> Tensor",
          &kv_decode_paged_asym_cuda);
    m.def("mxint8_kv_decode_paged(Tensor q, Tensor cache, Tensor block_table, Tensor seq_lens, "
          "int B, int H, int Hkv, int D, int NB, int block_size, int slot_bytes, int per_head) -> Tensor",
          &mxint8_kv_decode_paged_cuda);
    m.def("mxfp8_kv_decode_paged(Tensor q, Tensor cache, Tensor block_table, Tensor seq_lens, "
          "int B, int H, int Hkv, int D, int NB, int block_size, int slot_bytes, int per_head) -> Tensor",
          &mxfp8_kv_decode_paged_cuda);
    m.def("mxfp6_kv_decode_paged(Tensor q, Tensor cache, Tensor block_table, Tensor seq_lens, "
          "int B, int H, int Hkv, int D, int NB, int block_size, int slot_bytes, int per_head) -> Tensor",
          &mxfp6_kv_decode_paged_cuda);
    // ---- plain MXINT8 baselines ----
    m.def("mxint8_dequant_bf16(Tensor scale_exp, Tensor qweight_cm, "
          "int OUT, int K, int NB) -> Tensor", &mxint8_dequant_bf16_cuda);
    m.def("mxint8_kv_write(Tensor X, int H, int L, int D, int NB) -> Tensor[]", &mxint8_kv_write_cuda);
    m.def("mxfp8_kv_write(Tensor X, int H, int L, int D, int NB) -> Tensor[]", &mxfp8_kv_write_cuda);
    m.def("mxfp6_kv_write(Tensor X, int H, int L, int D, int NB) -> Tensor[]", &mxfp6_kv_write_cuda);
}
PYBIND11_MODULE(ssnf_cuda, m) {}

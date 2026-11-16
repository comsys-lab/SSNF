"""Build and load the official CUDA kernels without importing the serving scheduler."""
from pathlib import Path
import importlib.util, json, os, sys, types
R=Path(__file__).resolve().parents[1]
K=R/'sources/omniserve/kernels'
SOURCES={
 'qgemm_w4a8_per_chn':['qgemm/w4a8_per_chn/pybind.cpp','qgemm/w4a8_per_chn/gemm_cuda.cu'],
 'qgemm_w4a8_per_group':['qgemm/w4a8_per_group/pybind.cpp','qgemm/w4a8_per_group/gemm_cuda.cu'],
 'fused_kernels':['fused.cpp','fused_kernels.cu'],
 'layernorm_ops':['layernorm.cpp','layernorm_kernels.cu'],
 'activation_ops':['activation.cpp','activation_kernels.cu'],
 'fused_attention_fine_grained_dense':['fused_attention/fused_attention_fine_grained/dense_attention/fused_attention.cpp','fused_attention/fused_attention_fine_grained/dense_attention/decoderMaskedMultiheadAttention.cu','fused_attention/fused_attention_fine_grained/fine_grained_common/update_kv_cache.cu','fused_attention/common/input_metadata_helper.cu'],
 'fused_attention_pure_dense':['fused_attention/fused_attention_pure_dense/fused_attention.cpp','fused_attention/fused_attention_pure_dense/decoderMaskedMultiheadAttention.cu','fused_attention/fused_attention_pure_dense/update_kv_cache.cu','fused_attention/fused_attention_pure_dense/input_metadata_helper.cu'],
}
def load_backend(build=False,names=None):
 os.environ['PATH']=str(Path(sys.executable).parent)+':'+os.environ['PATH']
 import torch
 from torch.utils.cpp_extension import load
 pkg=sys.modules.setdefault('omniserve_backend',types.ModuleType('omniserve_backend'));pkg.__path__=[]
 flags=['-O3','-DENABLE_BF16','--use_fast_math','--threads=2','-gencode=arch=compute_120,code=sm_120','--expt-relaxed-constexpr','--expt-extended-lambda']
 flags += ['-U'+s for s in ['__CUDA_NO_HALF_OPERATORS__','__CUDA_NO_HALF_CONVERSIONS__','__CUDA_NO_BFLOAT16_OPERATORS__','__CUDA_NO_BFLOAT16_CONVERSIONS__','__CUDA_NO_BFLOAT162_OPERATORS__','__CUDA_NO_BFLOAT162_CONVERSIONS__']]
 for name in names or SOURCES:
  short='qserve_'+name;folder=R/'build'/short;folder.mkdir(parents=True,exist_ok=True)
  print(json.dumps(dict(event='backend_start',name=name,build=build)),flush=True)
  if build:
   module=load(name=short,sources=[str(K/'csrc'/p) for p in SOURCES[name]],extra_cuda_cflags=flags,extra_cflags=['-O3'],extra_include_paths=[str(K/'csrc')],build_directory=str(folder),verbose=True)
  else:
   path=folder/(short+'.so');spec=importlib.util.spec_from_file_location(short,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  setattr(pkg,name,module);sys.modules['omniserve_backend.'+name]=module
  print(json.dumps(dict(event='backend_ready',name=name)),flush=True)
 return pkg

if __name__=='__main__':
 import argparse,time
 p=argparse.ArgumentParser();p.add_argument('--names',default=','.join(SOURCES));a=p.parse_args()
 os.environ.setdefault('MAX_JOBS','2');t=time.monotonic()
 load_backend(build=True,names=a.names.split(','))
 (R/'artifacts/build_complete.json').write_text(json.dumps(dict(complete=True,names=a.names.split(','),seconds=time.monotonic()-t),indent=2)+'\n')

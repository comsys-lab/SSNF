# Import alias: the code imports `ssnf_lib.pack`; in this package layout the
# module lives in quant/ssnf_lib/.
import os

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
__path__ = [os.path.join(_root, "quant", "ssnf_lib")]

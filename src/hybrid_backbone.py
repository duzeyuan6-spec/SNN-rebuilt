"""Private PLV implementation namespace for OUR hybrid only.

Load the unmodified vendor source into a separate module, preserving weight keys
and operation order. Overriding its reshape does NOT modify model.PLV_CSNet,
the vendor file, baseline instances, or baseline checkpoints.
"""
import importlib.util
import sys
from pathlib import Path
from torch.nn.functional import pixel_shuffle

reshape_blocks = pixel_shuffle


def _hybrid_reshape(x, blocksize):
    return reshape_blocks(x, blocksize)


_name = 'src._hybrid_plv_implementation'
_path = Path(__file__).resolve().parents[1] / 'baselines/PLV-CSNet/model/PLV_CSNet.py'
_spec = importlib.util.spec_from_file_location(_name, _path)
_implementation = importlib.util.module_from_spec(_spec)
sys.modules[_name] = _implementation
_spec.loader.exec_module(_implementation)
_implementation.My_Reshape_Adap = _hybrid_reshape
HybridPLVBackbone = _implementation.PLV_CSNet

"""Frozen test-set activity accounting, separate from timed execution."""
import argparse,json,sys,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;C=R.parent
sys.path[:0]=[str(C/'scripts'),str(C)]
import torch
from evaluate_reconstruction_v3 import load_model,evaluate_image
from src.data import list_images,load_gray,pil_to_tensor
from src.spiking import BurstIF
from train_reconstruction_v2 import image_hash,atomic_json
torch.set_num_threads(4);torch.manual_seed(123)
torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
parser=argparse.ArgumentParser()
parser.add_argument('--main',required=True)
parser.add_argument('--no-regularizer',required=True)
parser.add_argument('--manifest',required=True)
parser.add_argument('--test-root',required=True)
parser.add_argument('--out',required=True)
args=parser.parse_args()
out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
manifest=json.loads(Path(args.manifest).read_text())
forbidden={r['sha256'] for part in ['train','val'] for r in manifest[part]}
models={'main':Path(args.main),'no_regularizer':Path(args.no_regularizer)}
try:
 for label,ck in models.items():
  model,state=load_model(ck,'cuda'); records=[]; current={}
  def hook(name):
   def collect(node,inputs,output):
    v=node.counts.detach();av=v.abs();n=v.numel()
    current[name]={'elements':n,'nonzero':int((v!=0).sum()),'absolute_count':float(av.sum()),'saturated':int((av==node.levels).sum()),'levels':node.levels}
   return collect
  handles=[node.register_forward_hook(hook(name)) for name,node in model.named_modules() if isinstance(node,BurstIF)]
  for ds,n in [('Set11',11),('BSD68',68),('Urban100',100)]:
   files=list_images(Path(args.test_root)/ds);assert len(files)==n
   for i,f in enumerate(files):
    assert image_hash(f) not in forbidden
    current.clear()
    with torch.inference_mode():
     x=pil_to_tensor(load_gray(f))[None].cuda();quality=evaluate_image(model,x)
    records.append({'dataset':ds,'image':str(f),'psnr':quality['psnr'],'ssim':quality['ssim'],'layers':dict(current)})
    atomic_json(out/'status.json',{'state':'running','model':label,'dataset':ds,'image':i+1,'total':n})
  atomic_json(out/(label+'.json'),{'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest(),'args':state['args'],'rows':records})
  for handle in handles:handle.remove()
  del model,x;torch.cuda.empty_cache()
 atomic_json(out/'protocol.json',{'scope':'frozen full test images; activity includes replicated block padding; quality on original extent','count_not_timestep':True,'normalization':'nonzero/elements; abs(count)/elements; abs(count)/(elements*levels); saturation/elements','timing':'hooks excluded from hardware runs','memory_measurement':False,'selection':'none'})
 atomic_json(out/'status.json',{'state':'completed'})
except BaseException as e:
 atomic_json(out/'status.json',{'state':'failed','error':repr(e)});raise

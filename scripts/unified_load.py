from pathlib import Path
import sys,os
R=Path(__file__).resolve().parent;C=R.parent;B=C.parent
sys.path[:0]=[str(C/'scripts'),str(C)]
def load(checkpoint):
 import torch
 from evaluate_reconstruction_v3 import load_model
 state=torch.load(checkpoint,map_location='cpu',weights_only=False)
 if state.get('args',{}).get('architecture')=='repository':
  model,state=load_model(checkpoint,'cuda')
  return model,model.cs.phi.detach().cpu(),lambda x:model(model.cs(x))
 saved=state['model'];protocol=state['protocol'];args=dict(protocol['args'])
 if os.environ.get('SNN_EXTERNAL_SOURCE'):args['source']=os.environ['SNN_EXTERNAL_SOURCE']
 phi=saved['net.Phiweight'].reshape(-1,1024) if 'net.Phiweight' in saved else saved['phi']
 if protocol.get('method','').startswith('ESDNet'):
  from esdnet_cs_adapter import ESDNetCS
  model=ESDNetCS(args['source'],phi)
 elif 'spikerain' in str(args['out']).lower():
  from spikerain_cs_adapter import SpikeRainCS
  model=SpikeRainCS(args['source'],phi)
 else:
  if args['method']=='octuf':
   from multirate_octuf_adapter import OfficialCS
  else:
   from ann_cs_adapters import OfficialCS
  model=OfficialCS(args['source'],args['method'],phi)
 model.load_state_dict(saved,strict=True);model=model.cuda().eval()
 return model,phi,model

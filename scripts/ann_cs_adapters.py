"""Official CS decoders under a declared common 32x32, frozen-Phi protocol.

Only spatial constants in the ISTA sources are adapted (33->32,1089->1024).
OCTUF keeps its decoder but uses shared frozen sensing instead of joint learning.
"""
import ast
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

FILES={'ista_plus':'ISTA-Net-PyTorch/Train_CS_ISTA_Net_plus.py',
       'ista_pp':'ISTA-Netpp/Train_ISTA_Net_pp.py','octuf':'OCTUF/model_octuf.py'}

def definitions(source,kind,block=32):
    from einops import rearrange
    import numbers
    path=Path(source)/FILES[kind]
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    tree.body=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and (kind=='octuf' or isinstance(n,ast.ClassDef) and n.name!='RandomDataset' or isinstance(n,ast.FunctionDef) and n.name=='PhiTPhi_fun')]
    if kind!='octuf':
        class SpatialConstants(ast.NodeTransformer):
            def visit_Constant(self,n):
                if type(n.value)==int and n.value in (33,1089):return ast.copy_location(ast.Constant(block if n.value==33 else block*block),n)
                return n
        tree=SpatialConstants().visit(tree)
    ns=dict(torch=torch,nn=nn,F=F,init=nn.init,n_output=block*block,rearrange=rearrange,numbers=numbers)
    exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),ns)
    return ns

class OfficialCS(nn.Module):
    def __init__(self,source,kind,phi):
        super().__init__();self.kind=kind;self.aux=None
        ns=definitions(source,kind)
        if kind=='octuf':
            self.net=ns['OCT'](10,.25)
            with torch.no_grad():self.net.Phiweight.copy_(phi.reshape(256,1,32,32))
            self.net.Phiweight.requires_grad_(False)
        else:
            self.register_buffer('phi',phi.clone())
            self.net=ns['ISTANetplus'](9) if kind=='ista_plus' else ns['ISTA_Net_pp'](20)
        if kind=='ista_plus':self.register_buffer('qinit',phi.T.clone())
    def reset(self):self.aux=None
    def forward(self,x):
        self.aux=None
        if self.kind=='octuf':return self.net(x)
        if self.kind=='ista_pp':return self.net([x,x.new_tensor([[.25,0.]])],self.phi,None,256)
        b,c,h,w=x.shape;patches=F.unfold(x,32,stride=32).transpose(1,2).reshape(-1,1024)
        pred,sym=self.net(patches@self.phi.T,self.phi,self.qinit)
        if self.training:self.aux=.01*sum(z.square().mean() for z in sym)
        return F.fold(pred.reshape(b,-1,1024).transpose(1,2),(h,w),32,stride=32)
    def loss(self,pred,target):
        return F.mse_loss(pred,target)+(self.aux if self.aux is not None else 0.)
    @torch.no_grad()
    def fit_initializer(self,entries):
        if self.kind!='ista_plus':return
        from src.data import load_gray,pil_to_tensor
        covariance=torch.zeros(1024,1024,device=self.phi.device,dtype=torch.float64)
        for entry in entries:
            x=pil_to_tensor(load_gray(entry['path']))[None].to(self.phi.device)
            patches=F.unfold(x,32,stride=32)[0].double()
            covariance+=patches@patches.T
        phi=self.phi.double();cross=covariance@phi.T
        self.qinit.copy_((cross@torch.linalg.pinv(phi@cross)).float())
        assert torch.isfinite(self.qinit).all()

def smoke(source):
    torch.set_num_threads(4);torch.manual_seed(123)
    phi=torch.linalg.qr(torch.randn(1024,256)).Q.T.cuda()
    results={}
    for kind in FILES:
        model=OfficialCS(source,kind,phi).cuda();model.train()
        x=torch.rand(1,1,64,96,device='cuda');y=model(x);model.loss(y,x).backward()
        assert y.shape==x.shape and torch.isfinite(y).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
        model.eval()
        with torch.no_grad():
            first=model(x);assert torch.equal(first,model(x))
            # Check wrapper against direct author decoder at adapted dimensions.
            if kind=='ista_plus':
                patches=F.unfold(x,32,stride=32).transpose(1,2).reshape(-1,1024)
                raw,_=model.net(patches@phi.T,phi,model.qinit)
                ref=F.fold(raw.reshape(1,-1,1024).transpose(1,2),(64,96),32,stride=32)
            elif kind=='ista_pp':ref=model.net([x,x.new_tensor([[.25,0.]])],phi,None,256)
            else:ref=model.net(x)
            torch.testing.assert_close(first,ref,rtol=0,atol=0)
        results[kind]=dict(trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),parameters=sum(p.numel() for p in model.parameters()),fixed_sensing=262144,extra_fixed_initializer=262144 if kind=='ista_plus' else 0)
        del model
    print(results,flush=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);a=p.parse_args();smoke(a.source)

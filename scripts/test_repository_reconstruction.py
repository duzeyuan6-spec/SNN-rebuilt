import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.spiking import BurstIF
from src.models import build_repository_reconstruction,build_model,MatchedBurstPrior
from src.energy import estimate_ops


class RepositoryTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(123);torch.set_num_threads(2)

    def test_snn_stride2_preserves_capacity_shared_weights_and_shape(self):
        baseline=build_repository_reconstruction(kind='ann',channels=8,stages=2,block_size=4)
        snn=build_repository_reconstruction(channels=8,stages=2,block_size=4,snn_spatial_stride=2)
        snn.load_state_dict(baseline.state_dict(),strict=True)
        self.assertEqual(sum(p.numel() for p in snn.parameters()),sum(p.numel() for p in baseline.parameters()))
        for h,w in [(12,16),(13,15)]:
            x=torch.rand(2,1,h,w)
            result=snn.priors[0](x)
            self.assertEqual(result.shape,x.shape)
            result.square().mean().backward()
        for prior in snn.priors:
            snn.zero_grad();x=torch.rand(2,1,12,16)
            prior(x).square().mean().backward()
            for parameter in prior.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertGreater(float(parameter.grad.abs().sum()),0.)
        with self.assertRaisesRegex(ValueError,'SNN-only'):
            build_repository_reconstruction(kind='ann',channels=8,stages=2,block_size=4,snn_spatial_stride=2)

    def test_snn_stride2_reduces_counted_convolution_work(self):
        original=build_repository_reconstruction(channels=8,stages=2,block_size=4)
        smaller=build_repository_reconstruction(channels=8,stages=2,block_size=4,snn_spatial_stride=2)
        smaller.load_state_dict(original.state_dict(),strict=True)
        x=torch.rand(1,1,16,16)
        original(original.cs(x));smaller(smaller.cs(x));original.eval();smaller.eval()
        a=estimate_ops(original,original.cs(x));b=estimate_ops(smaller,smaller.cs(x))
        conv=lambda ops:sum(r['dense_equivalent_macs'] for r in ops['layers'] if r['layer']!='measurement_operator')
        self.assertAlmostEqual(conv(b)/conv(a),.25)
        self.assertEqual(len(a['layers']),len(b['layers']))

    def test_mixed_resolution_tail_preserves_capacity_and_arithmetic(self):
        a=build_repository_reconstruction(kind='ann',channels=8,stages=4,block_size=4)
        s=build_repository_reconstruction(channels=8,stages=4,block_size=4,snn_spatial_stride=2,snn_fullres_tail=1)
        s.load_state_dict(a.state_dict(),strict=True)
        self.assertEqual([p.spatial_stride for p in s.priors],[2,2,2,1])
        self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in s.parameters()))
        x=torch.rand(2,1,16,16);y=s.cs(x);out=s(y)
        (out-x).square().mean().backward()
        for p in s.priors:
            self.assertGreater(float(p.out_conv.weight.grad.abs().sum()),0.)
        s.eval();a.eval()
        conv=lambda ops:sum(r['dense_equivalent_macs'] for r in ops['layers'] if r['layer']!='measurement_operator')
        self.assertAlmostEqual(conv(estimate_ops(s,y))/conv(estimate_ops(a,y)),.25*.75+.25)
        with self.assertRaises(ValueError):
            build_repository_reconstruction(snn_spatial_stride=1,snn_fullres_tail=1)

    def test_mse_calibration_minimizes_training_sample_error_and_freezes_at_eval(self):
        node=BurstIF(3,8);node.calibration_mode='mse'
        x=torch.randn(2,3,16,16);x[:,0,::4,::4]*=5
        q=node.calibration_candidate(x)
        sample=x[...,::4,::4]
        rms3=(3*sample.square().mean((0,2,3),keepdim=True).sqrt()/8).clamp_min(1e-4)
        error=lambda quantum:((sample/quantum).round().clamp(-8,8)*quantum-sample).square().mean((0,2,3))
        self.assertTrue(bool((error(q)<=error(rms3)+1e-7).all()))
        node(x);before=node.quantum.clone();node.eval();node(x*10)
        torch.testing.assert_close(node.quantum,before,rtol=0,atol=0)
        models=[build_repository_reconstruction(kind=k,channels=8,stages=2,block_size=4,
                                               burst_calibration='mse') for k in ('spike','ann')]
        self.assertEqual(sum(p.numel() for p in models[0].parameters()),sum(p.numel() for p in models[1].parameters()))

    def test_inference_omits_surrogate_and_preserves_discrete_prediction(self):
        node=BurstIF(3,8);node(torch.randn(2,3,16,16));node.eval()
        x=torch.randn(2,3,16,16,requires_grad=True)
        reference=node(x)
        with torch.inference_mode():
            result=node(x)
            torch.testing.assert_close(result,reference,atol=1e-6,rtol=1e-6)
            torch.testing.assert_close(node.counts,(x/node.quantum).round().clamp(-8,8),rtol=0,atol=0)

    def test_exact_parameter_match_and_usable_shared_initialization(self):
        a=build_repository_reconstruction(kind='ann',channels=8,stages=2,time_steps=4,block_size=4)
        s=build_repository_reconstruction(channels=8,stages=2,time_steps=4,block_size=4)
        self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in s.parameters()))
        s.load_state_dict(a.state_dict(),strict=True)
        torch.testing.assert_close(s.cs.phi,a.cs.phi,atol=0,rtol=0)
        for prior in s.priors:prior.set_training_blend(0.)
        x=torch.rand(2,1,12,16)
        torch.testing.assert_close(s(s.cs(x)),a(a.cs(x)),atol=0,rtol=0)

    def test_eval_is_fully_discrete_and_does_not_calibrate(self):
        n=BurstIF(2,4);x=torch.randn(1,2,8,8,requires_grad=True)
        n(x);n.eval();n.training_blend=0.
        before=n.quantum.clone();out=n(x*3)
        torch.testing.assert_close(out/n.quantum,(out/n.quantum).round(),atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(n.quantum,before,atol=0,rtol=0)
        self.assertLessEqual(float(n.counts.detach().abs().max()),4.)
        out.sum().backward();self.assertGreater(float(x.grad.abs().sum()),0.)

    def test_hidden_gradients_and_no_time_repeated_convolutions(self):
        m=build_repository_reconstruction(channels=8,stages=2,time_steps=8,block_size=4)
        calls=[];hooks=[]
        for module in m.modules():
            if isinstance(module,torch.nn.Conv2d):hooks.append(module.register_forward_hook(lambda *args:calls.append(1)))
        x=torch.rand(2,1,12,16);pred=m(m.cs(x))
        ((pred-x).square().mean()+.001*m.spike_regularization()).backward()
        self.assertEqual(len(calls),8)
        for p in m.priors:
            for layer in p.hidden:self.assertGreater(float(layer.weight.grad.abs().sum()),0.)
        for hook in hooks:hook.remove()

    def test_event_accounting_counts_multiple_events(self):
        m=build_repository_reconstruction(channels=8,stages=2,time_steps=8,block_size=4)
        y=m.cs(torch.rand(1,1,12,16));m(y);m.eval()
        ops=estimate_ops(m,y)
        events=[r for r in ops['layers'] if r.get('event_encoding')=='signed_burst_counts']
        self.assertEqual(len(events),6)
        for r in events:
            self.assertAlmostEqual(r['hypothetical_acs'],r['dense_equivalent_macs']*r['events_per_input'])
        self.assertFalse(ops['hardware_measured'])

    def test_zero_communication_changes_output(self):
        m=build_repository_reconstruction(channels=8,stages=2,time_steps=4,block_size=4)
        x=torch.rand(1,1,12,16);y=m.cs(x);normal=m(y);disabled=m(y,intervention='zero_spikes')
        self.assertGreater(float((normal-disabled).detach().abs().sum()),0.)

    def test_legacy_build_is_preserved(self):
        m=build_model(dict(seed=123,cs=dict(block_size=4,measurement_rate=.25),
                      model=dict(kind='spike',stages=2,channels=8,blocks_per_stage=2,time_steps=4)))
        self.assertFalse(isinstance(m.priors[0],MatchedBurstPrior))
        self.assertEqual(tuple(m(m.cs(torch.rand(1,1,12,16))).shape),(1,1,12,16))

    def test_dilated_context_keeps_active_capacity_and_gradients(self):
        old=build_repository_reconstruction(kind='ann',channels=8,stages=2,time_steps=4,block_size=4)
        s=build_repository_reconstruction(channels=8,stages=2,time_steps=4,block_size=4,spatial_dilation=2)
        a=build_repository_reconstruction(kind='ann',channels=8,stages=2,time_steps=4,block_size=4,spatial_dilation=2)
        for model in (a,s):
            self.assertEqual(sum(p.numel() for p in model.parameters()),sum(p.numel() for p in old.parameters()))
            model.load_state_dict(old.state_dict(),strict=True)
        self.assertEqual(s.priors[0].hidden[1].dilation,(2,2))
        for p in s.priors:p.set_training_blend(0.)
        x=torch.rand(2,1,16,16);pred=s(s.cs(x))
        torch.testing.assert_close(pred,a(a.cs(x)),atol=0,rtol=0)
        (pred-x).square().mean().backward()
        self.assertGreater(float(s.priors[0].hidden[1].weight.grad.abs().sum()),0.)

    def test_activity_statistics_can_be_disabled_without_changing_prediction(self):
        import math
        m=build_repository_reconstruction(channels=8,stages=2,time_steps=4,block_size=4)
        y=m.cs(torch.rand(1,1,12,16));m(y);m.eval()
        with torch.no_grad():
            reference=m(y)
            for p in m.priors:p.collect_activity=False
            torch.testing.assert_close(m(y),reference,rtol=0,atol=0)
            self.assertTrue(math.isnan(m.firing_rate()))
        m.train();m(y)
        self.assertTrue(bool(torch.isfinite(m.spike_regularization())))

    def test_learned_sensing_gradient_and_frozen_shared_operator(self):
        teacher=build_repository_reconstruction(kind='ann',channels=8,stages=2,time_steps=4,
                                               block_size=4,learnable_sensing=True)
        x=torch.rand(2,1,16,16)
        (teacher(teacher.cs(x))-x).square().mean().backward()
        self.assertGreater(float(teacher.cs.phi.grad.abs().sum()),0.)
        with torch.no_grad():
            teacher.cs.phi.add_(teacher.cs.phi.grad,alpha=-.001)
            q,_=torch.linalg.qr(teacher.cs.phi.T,mode='reduced');teacher.cs.phi.copy_(q.T)
        torch.testing.assert_close(teacher.cs.phi@teacher.cs.phi.T,torch.eye(teacher.cs.m),atol=2e-6,rtol=2e-6)
        models=[build_repository_reconstruction(kind=k,channels=8,stages=2,time_steps=4,
                 block_size=4,learnable_sensing=True,freeze_sensing=True) for k in ('spike','ann')]
        for m in models:
            m.load_state_dict(teacher.state_dict(),strict=True)
            before=m.cs.phi.detach().clone();opt=torch.optim.Adam(m.parameters(),lr=.001)
            (m(m.cs(x))-x).square().mean().backward();opt.step()
            self.assertIsNone(m.cs.phi.grad)
            torch.testing.assert_close(m.cs.phi,before,atol=0,rtol=0)
        torch.testing.assert_close(models[0].cs.phi,models[1].cs.phi,atol=0,rtol=0)
        self.assertEqual(sum(p.numel() for p in models[0].parameters()),sum(p.numel() for p in models[1].parameters()))


if __name__=='__main__':unittest.main(verbosity=2)

import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from src.reconstruction_v3 import StableStagewiseRecon
from evaluate_reconstruction_v3 import pad_image,evaluate_image,paired_interval,check_pair


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(12)

    def test_padding_retains_original_pixels_and_small_images(self):
        x=torch.rand(1,1,7,9);p=pad_image(x,32)
        self.assertEqual(tuple(p.shape),(1,1,32,32))
        torch.testing.assert_close(p[...,:7,:9],x,rtol=0,atol=0)
        torch.testing.assert_close(p[...,-1,-1],x[...,-1,-1])

    def test_interventions_modify_binary_communication(self):
        m=StableStagewiseRecon(channels=8,stages=2,time_steps=3,block_size=4).eval()
        y=m.cs(torch.rand(1,1,12,16));seen=[]
        hook=m.priors[0].hidden[0].register_forward_pre_hook(lambda mod,args:seen.append(args[0].clone()))
        with torch.no_grad():
            baseline=m(y);normal=[v.clone() for v in seen];seen.clear()
            m(y,intervention='zero_spikes')
            self.assertTrue(all(torch.count_nonzero(v)==0 for v in seen));seen.clear()
            m(y,intervention='shift_spikes')
            for a,b in zip(normal,seen):
                self.assertEqual(float(a.sum()),float(b.sum()))
                self.assertTrue(bool(((b==0)|(b==1)).all()))
            seen.clear()
            torch.testing.assert_close(m(y),baseline,rtol=0,atol=0)
        hook.remove()
        with self.assertRaises(ValueError):m(y,intervention='typo')

    def test_full_image_metrics_keep_nonmultiple_shape(self):
        m=StableStagewiseRecon(channels=8,stages=2,time_steps=2,block_size=4).eval()
        result=evaluate_image(m,torch.rand(1,1,9,11))
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in result.values()))

    def test_interval_is_paired_and_reproducible(self):
        a=paired_interval([2.,2.,2.],repeats=100)
        self.assertEqual(a['ci95'],[2.,2.])
        self.assertEqual(a,paired_interval([2.,2.,2.],repeats=100))

    def test_mismatched_checkpoint_protocol_rejected(self):
        states=[dict(manifest_hash='other'),dict(manifest_hash='expected')]
        with self.assertRaisesRegex(ValueError,'manifest'):check_pair(states,[],'expected')

    def test_fixed_baseline_requires_exact_checkpoint_digest(self):
        with self.assertRaisesRegex(ValueError,'SHA256'):
            check_pair([],[],'manifest',fixed_ann_sha256='registered',ann_checkpoint_sha256='different')


if __name__=='__main__':unittest.main(verbosity=2)

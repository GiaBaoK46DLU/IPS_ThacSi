"""Scientific S-KAN invariants and historical default forward compatibility."""
import copy
import subprocess
import unittest

import torch

from ips.data import ROOT
from ips.algorithms.kan.models import build_model
from ips.algorithms.kan.models import reference
from ips.algorithms.kan.runtime import read_config
from ips.algorithms.kan.variants.selectable.bases import CandidateFunction,default_pool
from ips.algorithms.kan.variants.selectable.layers import (
    SelectableKANLinear,SelectedKANLinear,compact_layer,make_selectable_layer)
from ips.algorithms.kan.variants.selectable.search import set_phase


class SelectableLayerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        cls.config=read_config(ROOT/'src/ips/algorithms/kan/configs/selectable/uji_v4_matched.json')

    def test_all_eighteen_candidates_have_finite_values_and_gradients(self):
        for spec in default_pool():
            with self.subTest(basis=spec['name']):
                candidate=CandidateFunction(3,2,spec,self.config['kan'])
                x=torch.tensor([[-100.,0.,100.],[-.3,.2,1.]],requires_grad=True)
                result=candidate(x)
                self.assertTrue(torch.isfinite(result).all())
                result.square().mean().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in candidate.parameters() if p.grad is not None))

    def test_selective_phase_freezes_everything_except_node_logits(self):
        model=torch.nn.Sequential(SelectableKANLinear(3,2,self.config['kan']))
        set_phase(model,'selective')
        before={n:p.detach().clone() for n,p in model.named_parameters()}
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
        model(torch.randn(8,3)).square().mean().backward();optimizer.step()
        changed=[n for n,p in model.named_parameters() if not torch.equal(p,before[n])]
        self.assertEqual(changed,['0.gate_logits'])
        set_phase(model,'full')
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_packed_search_projection_matches_explicit_weighted_sum(self):
        layer=SelectableKANLinear(3,2,self.config['kan'])
        with torch.no_grad():
            layer.gate_logits.normal_()
        x=torch.randn(4,3,requires_grad=True)
        explicit=sum(candidate(x,layer.gate_probabilities()[:,index])
                     for index,candidate in enumerate(layer.candidates))
        packed=layer(x)
        torch.testing.assert_close(packed,explicit,atol=1e-7,rtol=1e-5)
        a=torch.autograd.grad(packed.square().sum(),(x,layer.gate_logits),retain_graph=True)
        b=torch.autograd.grad(explicit.square().sum(),(x,layer.gate_logits))
        for left,right in zip(a,b):
            torch.testing.assert_close(left,right,atol=1e-7,rtol=1e-5)

    def test_pruning_is_per_input_and_compaction_preserves_trained_function(self):
        kan=self.config['kan'];count=len(kan['candidates'])
        layer=SelectableKANLinear(count,3,kan)
        with torch.no_grad():
            layer.gate_logits.copy_(torch.eye(count)*10)
        for remaining in range(count-1,0,-1):
            layer.prune_one()
            self.assertTrue((layer.active.sum(-1)==remaining).all())
            self.assertTrue(torch.allclose(layer.gate_probabilities().sum(-1),torch.ones(count),atol=1e-6))
        self.assertEqual(layer.chosen(),list(range(count)))
        with self.assertRaises(ValueError):
            layer.prune_one()
        compact=compact_layer(layer,kan)
        x=torch.randn(2,4,count)
        torch.testing.assert_close(layer(x),compact(x),atol=2e-6,rtol=1e-5)
        self.assertLess(sum(p.numel() for p in compact.parameters()),sum(p.numel() for p in layer.parameters()))
        restored=SelectedKANLinear(count,3,kan,list(range(count)))
        restored.load_state_dict(compact.state_dict(),strict=True)
        torch.testing.assert_close(compact(x),restored(x),atol=0,rtol=0)

    def test_resumed_search_reconstructs_active_candidate_cache(self):
        first=SelectableKANLinear(3,2,self.config['kan'])
        for _ in range(17):
            first.prune_one()
        second=SelectableKANLinear(3,2,self.config['kan'])
        second.load_state_dict(first.state_dict())
        self.assertEqual(second.active_candidates,first.active_candidates)
        x=torch.randn(3,3)
        torch.testing.assert_close(first(x),second(x),atol=0,rtol=0)

    def test_missing_or_mismatched_selection_is_rejected(self):
        kan=copy.deepcopy(self.config['kan']);kan['selection_stage']='selected'
        with self.assertRaises(ValueError):
            make_selectable_layer(3,2,kan,'missing')
        for choices in ([0,1],[0,1,18],[0,1,1.]):
            with self.assertRaises(ValueError):
                SelectedKANLinear(3,2,kan,choices)

    def test_historical_v4_constructor_and_forward_stay_identical(self):
        config=read_config(ROOT/'src/ips/algorithms/kan/configs/v4_balanced_points.json')
        config['positioning'].update(fit_scope='internal_train_only',references=[
            {'class_id':i,'xy':[i/10.,i/20.]} for i in range(13)])
        old_code=subprocess.check_output(['git','show','9f611d1:src/ips/algorithms/kan/models/networks.py'],cwd=ROOT,text=True,encoding='utf-8')
        namespace={'__package__':'ips.algorithms.kan.models','__name__':'legacy_networks'}
        exec(compile(old_code,'legacy_networks.py','exec'),namespace)
        original_code=subprocess.check_output(['git','show','9f611d1:src/ips/algorithms/kan/models/reference.py'],cwd=ROOT,text=True,encoding='utf-8')
        original_scope={'__package__':'ips.algorithms.kan.models','__name__':'legacy_reference'}
        exec(compile(original_code,'legacy_reference.py','exec'),original_scope)
        original_scope['MixerBlock']=namespace['MixerBlock']
        torch.manual_seed(123);old=original_scope['ReferenceKANMixer'](config,13).eval()
        torch.manual_seed(123);new=build_model(config,13).eval()
        self.assertEqual(list(old.state_dict()),list(new.state_dict()))
        for name,value in old.state_dict().items():
            self.assertTrue(torch.equal(value,new.state_dict()[name]),name)
        x=torch.rand(3,520,2);x[:,:,1]=1
        with torch.no_grad():
            a,b=old(x),new(x)
        for key in ('xy','logits','reference_logits'):
            torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)

    def test_all_bspline_selected_model_reduces_to_v4(self):
        config=read_config(ROOT/'src/ips/algorithms/kan/configs/v4_balanced_points.json')
        config['positioning'].update(fit_scope='internal_train_only',references=[
            {'class_id':i,'xy':[i/10.,i/20.]} for i in range(13)])
        torch.manual_seed(777);baseline=build_model(config,13).eval()
        selected=copy.deepcopy(config)
        choices={name:[0]*module.in_features for name,module in baseline.named_modules()
                 if module.__class__.__name__=='KANLinear'}
        selected['kan'].update(backend='selectable',candidates=default_pool(),
                               selection_stage='selected',selected_activations=choices)
        torch.manual_seed(777);model=build_model(selected,13).eval()
        x=torch.rand(3,520,2);x[:,:,1]=1
        with torch.no_grad():
            a,b=baseline(x),model(x)
        for key in ('xy','logits','reference_logits'):
            torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()

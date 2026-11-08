"""Input-node-wise selection and physical compaction for S-KAN Mixer layers."""
from __future__ import annotations

import copy
import torch
from torch import nn
from torch.nn import functional as F

from .bases import CandidateFunction


def validate_pool(pool):
    if len(pool)<2 or len({spec['name'] for spec in pool})!=len(pool):
        raise ValueError('Selectable pool must contain at least2 uniquely named candidates.')


class SelectableKANLinear(nn.Module):
    def __init__(self, inputs, outputs, kan):
        super().__init__()
        pool=kan['candidates'];validate_pool(pool)
        self.in_features,self.out_features=inputs,outputs
        self.candidates=nn.ModuleList([CandidateFunction(inputs,outputs,spec,kan) for spec in pool])
        self.gate_logits=nn.Parameter(torch.zeros(inputs,len(pool)))
        self.register_buffer('active',torch.ones(inputs,len(pool),dtype=torch.bool))
        self.active_candidates=list(range(len(pool)))

    def gate_probabilities(self):
        return self.gate_logits.masked_fill(~self.active,float('-inf')).softmax(-1)

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self.active_candidates=torch.where(self.active.any(0))[0].cpu().tolist()

    def forward(self,x):
        probabilities=self.gate_probabilities()
        shape=x.shape;flat=x.reshape(-1,self.in_features)
        base=None;features=[];coefficients=[]
        for index in self.active_candidates:
            candidate=self.candidates[index]
            weights=probabilities[:,index]
            if candidate.spec['kind']=='bspline':
                basis=candidate.spline.b_splines(flat)
                base_weight=candidate.spline.base_weight
                coefficient=candidate.spline.scaled_spline_weight
            else:
                basis=candidate.basis(flat)
                base_weight=candidate.base_weight
                coefficient=candidate.basis_weight*candidate.basis_scaler[...,None]
            update=base_weight*weights[None]
            base=update if base is None else base+update
            features.append(basis.flatten(1))
            coefficients.append((coefficient*weights[None,:,None]).flatten(1))
        result=F.linear(F.silu(flat),base)+F.linear(torch.cat(features,1),torch.cat(coefficients,1))
        return result.reshape(*shape[:-1],self.out_features)

    @torch.no_grad()
    def prune_one(self):
        counts=self.active.sum(-1)
        if (counts<=1).any():
            raise ValueError('Pruning would remove the final activation of an input node.')
        least=self.gate_logits.masked_fill(~self.active,float('inf')).argmin(-1)
        self.active[torch.arange(self.in_features,device=least.device),least]=False
        self.active_candidates=torch.where(self.active.any(0))[0].cpu().tolist()
        return least.cpu().tolist()

    def chosen(self):
        if not (self.active.sum(-1)==1).all():
            raise ValueError('Selection must finish with exactly one candidate per input node.')
        return self.active.long().argmax(-1).cpu().tolist()


class SelectedKANLinear(nn.Module):
    """Only the chosen basis and parameters exist for each input coordinate."""
    def __init__(self,inputs,outputs,kan,choices):
        super().__init__()
        validate_pool(kan['candidates'])
        if (len(choices)!=inputs or any(type(i) is not int or not 0<=i<len(kan['candidates']) for i in choices)):
            raise ValueError('Selected activation map does not match layer input width/pool.')
        self.in_features,self.out_features=inputs,outputs
        self.register_buffer('choices',torch.tensor(choices,dtype=torch.long))
        self.groups=nn.ModuleDict()
        for index in sorted(set(choices)):
            inputs_for_group=[i for i,k in enumerate(choices) if k==index]
            self.register_buffer(f'inputs_{index}',torch.tensor(inputs_for_group,dtype=torch.long))
            self.groups[str(index)]=CandidateFunction(len(inputs_for_group),outputs,kan['candidates'][index],kan)
            # Initialization uses the entire layer's fan-in, independent of how
            # many nodes happen to choose this family. Compaction is storage,
            # not a reason to inflate small groups' initial weights.
            with torch.no_grad():
                candidate=self.groups[str(index)]
                ratio=(len(inputs_for_group)/inputs)**.5
                if candidate.spec['kind']=='bspline':
                    candidate.spline.base_weight.mul_(ratio)
                    if hasattr(candidate.spline,'spline_scaler'):
                        candidate.spline.spline_scaler.mul_(ratio)
                else:
                    candidate.base_weight.mul_(ratio)
                    candidate.basis_weight.mul_(ratio)
                    candidate.basis_scaler.mul_(ratio)

    def forward(self,x):
        # Concatenation is algebraically the same sum over selected edges, but
        # uses two dense projections per layer instead of two per family.
        shape=x.shape
        inputs=[];features=[];bases=[];coefficients=[]
        for key,candidate in self.groups.items():
            part=x.index_select(-1,getattr(self,f'inputs_{key}')).reshape(-1,candidate.in_features)
            inputs.append(part)
            if candidate.spec['kind']=='bspline':
                features.append(candidate.spline.b_splines(part).flatten(1))
                bases.append(candidate.spline.base_weight)
                coefficients.append(candidate.spline.scaled_spline_weight.flatten(1))
            else:
                features.append(candidate.basis(part).flatten(1))
                bases.append(candidate.base_weight)
                coefficients.append((candidate.basis_weight*candidate.basis_scaler[...,None]).flatten(1))
        result=F.linear(F.silu(torch.cat(inputs,1)),torch.cat(bases,1))
        result=result+F.linear(torch.cat(features,1),torch.cat(coefficients,1))
        return result.reshape(*shape[:-1],self.out_features)


def make_selectable_layer(inputs,outputs,kan,path):
    stage=kan.get('selection_stage')
    if stage=='search':
        return SelectableKANLinear(inputs,outputs,kan)
    if stage=='selected':
        choices=kan.get('selected_activations',{}).get(path)
        if choices is None:
            raise ValueError(f'Missing frozen selection for {path}.')
        return SelectedKANLinear(inputs,outputs,kan,choices)
    raise ValueError('Selectable backend requires an explicit search/selected stage.')


def search_layers(model):
    return {name:module for name,module in model.named_modules() if isinstance(module,SelectableKANLinear)}


def selection_config(model,config):
    result=copy.deepcopy(config)
    layers=search_layers(model)
    if not layers:
        raise ValueError('No selectable layers in search model.')
    result['kan']['selection_stage']='selected'
    result['kan']['selected_activations']={name:layer.chosen() for name,layer in layers.items()}
    return result


@torch.no_grad()
def compact_layer(source, kan):
    """Audit helper: preserve selected trained weights, unlike fresh final training."""
    result=SelectedKANLinear(source.in_features,source.out_features,kan,source.chosen()).to(source.gate_logits.device)
    for key,target in result.groups.items():
        indices=getattr(result,f'inputs_{key}')
        original=source.candidates[int(key)]
        if original.spec['kind']=='bspline':
            target.spline.grid.copy_(original.spline.grid[indices])
            for name in ('base_weight','spline_weight','spline_scaler'):
                getattr(target.spline,name).copy_(getattr(original.spline,name).index_select(1,indices))
        else:
            for name in ('base_weight','basis_weight','basis_scaler'):
                getattr(target,name).copy_(getattr(original,name).index_select(1,indices))
            for name,value in target.basis.named_parameters():
                value.copy_(dict(original.basis.named_parameters())[name].index_select(0,indices))
            for name,value in target.basis.named_buffers():
                value.copy_(dict(original.basis.named_buffers())[name])
    return result

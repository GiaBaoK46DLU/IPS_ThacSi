"""Train-only, per-input-node activation search, followed by a fresh model.

Selection checkpoints resume only at a completed pruning round, retaining all
RNG/loader state. A search checkpoint is never used as an IPS Test predictor.
"""
from __future__ import annotations

import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .....data import ROOT, sha256
from .....data_access import load_bundle
from ...augmentation import training_representation
from ...dataset import FingerprintDataset
from ...models import build_model, parameter_count
from ...models.reference import fit_reference_points
from ...runtime import configure_runtime, save_json, software_info
from ...train import combined_loss
from ...v4_methods import point_sampler
from .layers import search_layers, selection_config


def stratified_indices(labels, fraction, seed):
    """Proportional largest-remainder allocation, at least one per class."""
    labels=np.asarray(labels)
    classes,counts=np.unique(labels,return_counts=True)
    target=round(len(labels)*fraction)
    if not 0<fraction<=1 or target<len(classes):
        raise ValueError('Search subset is too small to represent every class.')
    allocation=np.maximum(1,np.floor(counts*target/len(labels)).astype(int))
    ideal=counts*target/len(labels)
    while allocation.sum()<target:
        index=np.argmax(np.where(allocation<counts,ideal-allocation,-np.inf))
        allocation[index]+=1
    while allocation.sum()>target:
        index=np.argmax(np.where(allocation>1,allocation-ideal,-np.inf))
        allocation[index]-=1
    rng=np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(labels==c),n,replace=False)
                                   for c,n in zip(classes,allocation)]))


def subset_dataset(dataset,indices):
    if dataset.split_name!='train':
        raise ValueError('Activation search is restricted to the internal train split.')
    result=copy.copy(dataset)
    for name in ('features','indices','xy','classes','xy_original'):
        setattr(result,name,getattr(dataset,name)[indices].clone() if torch.is_tensor(getattr(dataset,name))
                else getattr(dataset,name)[indices].copy())
    result.metadata={key:value[indices].copy() for key,value in dataset.metadata.items()}
    return result


def set_phase(model,phase):
    if phase not in ('full','selective'):
        raise ValueError('Unknown S-KAN selection phase.')
    for name,parameter in model.named_parameters():
        parameter.requires_grad_(phase=='full' or name.endswith('.gate_logits'))


def rng_state(generator):
    state={'torch':torch.get_rng_state(),'loader':generator.get_state(),
           'python':random.getstate()}
    if torch.cuda.is_available():
        state['cuda']=torch.cuda.get_rng_state_all()
    return state


def restore_rng(state,generator):
    torch.set_rng_state(state['torch'])
    generator.set_state(state['loader'])
    random.setstate(state['python'])
    if 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])


def run_search(config,settings,seed,destination,device_name='cuda'):
    """No Validation/Test split is loaded here; selection is exclusively train."""
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    from .experiment import read, immutable
    if (destination/'selection.json').exists():
        result=read(destination/'selection.json')
        if result['seed']!=seed or result['search_settings']!=settings:
            raise ValueError('Existing selection belongs to a different seed/protocol.')
        return read(destination/'selected_config.json')
    device=configure_runtime(seed,device_name,config['training']['threads'])
    bundle=load_bundle(ROOT/config['split_dir'])
    train=FingerprintDataset.from_split(bundle.split('train'))
    config=copy.deepcopy(config)
    fit_reference_points(config,train)
    indices=stratified_indices(train.classes.numpy(),settings['fraction'],seed)
    subset=subset_dataset(train,indices)
    immutable(destination/'subset.json',{'seed':seed,'fit_scope':'internal_train_only',
        'fraction':settings['fraction'],'positions':indices.tolist(),'source_rows':subset.indices.tolist(),
        'class_counts':{str(int(c)):int((subset.classes==c).sum()) for c in subset.classes.unique()},
        'split_manifest_sha256':sha256(bundle.directory/'manifest.json')})
    model=build_model(config,len(bundle.classes)).to(device)
    layers=search_layers(model)
    if len(layers)!=8 or sum(layer.in_features for layer in layers.values())!=500:
        raise ValueError('The matched V4 Mixer must contain eight KAN layers and500 input nodes.')
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(training_representation(subset,config['training']),
        batch_size=settings['batch_size'],sampler=point_sampler(subset,config['training'],generator),
        generator=generator,num_workers=0)
    rounds=len(config['kan']['candidates'])-1
    history=[];start_round=0;prior_seconds=0.
    resume=destination/'round_checkpoint.pt'
    if resume.exists():
        saved=torch.load(resume,map_location='cpu',weights_only=True)
        if saved['seed']!=seed or saved['config']!=config or saved['settings']!=settings:
            raise ValueError('Search checkpoint differs from this experiment.')
        model.load_state_dict(saved['model'],strict=True)
        history=saved['history'];start_round=saved['completed_rounds'];prior_seconds=saved['seconds']
        restore_rng(saved['rng'],generator)
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(device)
    started=time.perf_counter()
    for round_index in range(start_round,rounds):
        phase_results=[]
        for phase,epochs in [('full',settings['full_epochs']),('selective',settings['selective_epochs'])]:
            set_phase(model,phase)
            gates=[parameter for name,parameter in model.named_parameters() if name.endswith('.gate_logits')]
            ordinary=[parameter for name,parameter in model.named_parameters()
                      if not name.endswith('.gate_logits') and parameter.requires_grad]
            groups=[{'params':gates,'weight_decay':0.}]
            if ordinary:
                groups.append({'params':ordinary,'weight_decay':config['training']['weight_decay']})
            optimizer=torch.optim.AdamW(groups,lr=settings['learning_rate'])
            for epoch in range(1,epochs+1):
                model.train();total=0.;samples=0
                for features,classes,xy in loader:
                    optimizer.zero_grad(set_to_none=True)
                    output=model(features.to(device))
                    loss,_,_=combined_loss(output,classes.to(device),xy.to(device),config['loss'])
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite S-KAN selection loss.')
                    loss.backward()
                    nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                        config['training']['gradient_clip_l2'],error_if_nonfinite=True)
                    optimizer.step();total+=loss.item()*len(features);samples+=len(features)
                phase_results.append({'phase':phase,'epoch':epoch,'train_loss':total/samples})
        pruned={name:layer.prune_one() for name,layer in layers.items()}
        probabilities={name:layer.gate_probabilities().detach().cpu() for name,layer in layers.items()}
        if any(not torch.allclose(p.sum(-1),torch.ones(len(p)),atol=1e-6) for p in probabilities.values()):
            raise ValueError('Selection weights must renormalize after pruning.')
        history.append({'round':round_index+1,'remaining_per_node':rounds-round_index,
                        'epochs':phase_results,'pruned_candidate_ids':pruned,
                        'probabilities_after_pruning':{name:p.tolist() for name,p in probabilities.items()}})
        payload={'seed':seed,'config':config,'settings':settings,'model':model.state_dict(),
            'completed_rounds':round_index+1,'history':history,'rng':rng_state(generator),
            'seconds':prior_seconds+time.perf_counter()-started}
        temporary=destination/'round_checkpoint.tmp'
        torch.save(payload,temporary);temporary.replace(resume)
        save_json(destination/'progress.json',{'seed':seed,'round':round_index+1,'rounds':rounds,
                  'remaining_per_node':rounds-round_index,'elapsed_seconds':payload['seconds']})
        print(f'SKAN_SEARCH seed={seed} round={round_index+1}/{rounds} '
              f'remaining={rounds-round_index} loss={phase_results[-1]["train_loss"]:.4f}',flush=True)
    selected=selection_config(model,config)
    # Only the structure crosses this boundary. Final training reconstructs all
    # parameters from the seed, including the embedding and the IPS head.
    selected['positioning'].pop('references',None)
    selected['positioning'].pop('fit_scope',None)
    immutable(destination/'selected_config.json',selected)
    immutable(destination/'selection_history.json',history)
    immutable(destination/'selection.json',{'seed':seed,'status':'completed','fit_scope':'internal_train_only',
        'search_settings':settings,'subset_rows':len(subset),'catalogue_fit_rows':len(train),
        'selected_nodes':sum(len(v) for v in selected['kan']['selected_activations'].values()),
        'selected_config_sha256':sha256(destination/'selected_config.json'),
        'selected_activations':selected['kan']['selected_activations'],
        'search_parameter_count':parameter_count(model.requires_grad_(True)),
        'seconds':prior_seconds+time.perf_counter()-started,'software':software_info(device),
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
        'test_evaluated':False,'validation_evaluated':False,'weights_transferred_to_training':False})
    return selected

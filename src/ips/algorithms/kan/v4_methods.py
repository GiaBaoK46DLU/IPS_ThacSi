"""Opt-in V4 inductive biases; fit only the declared internal training data."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def fit_ap_groups(config, dataset, *, final_refit=False):
    settings = config.get('positioning', {})
    method = settings.get('ap_grouping', 'source_order')
    if method == 'source_order':
        return
    if method != 'coobservation':
        raise ValueError('Unknown AP grouping method.')
    expected = 'development' if final_refit else 'train'
    if dataset.split_name != expected:
        raise ValueError('AP grouping requires explicitly identified training data.')
    if final_refit:
        if settings.get('ap_permutation_fit_scope') != 'internal_train_only':
            raise ValueError('Refit must retain the frozen internal-train AP ordering.')
        validate_permutation(settings.get('ap_permutation'))
        return
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    mask = dataset.features[:, :, 1].numpy().astype(np.float64)
    count = mask.sum(0)
    active = np.flatnonzero(count > 0)
    absent = np.flatnonzero(count == 0)
    if len(active) > 1:
        observed = mask[:, active]
        similarity = (observed.T @ observed) / np.sqrt(count[active, None] * count[None, active])
        distance = np.clip(1 - similarity, 0, 1)
        distance = (distance + distance.T) / 2
        np.fill_diagonal(distance, 0)
        order = leaves_list(linkage(squareform(distance), method='average', optimal_ordering=True))
        active = active[order]
    settings['ap_permutation'] = np.concatenate([active, absent]).tolist()
    settings['ap_permutation_fit_scope'] = 'internal_train_only'
    validate_permutation(settings['ap_permutation'])


def validate_permutation(order):
    if (not isinstance(order, list) or len(order) != 520 or
            any(type(x) is not int for x in order) or sorted(order) != list(range(520))):
        raise ValueError('AP permutation must preserve each of the 520 AP identities exactly once.')


def point_sampler(dataset, settings, generator):
    mode = settings.get('sampling', 'rows')
    if mode == 'rows':
        return None
    if mode != 'reference_points' or dataset.split_name not in ('train', 'development'):
        raise ValueError('Point-balanced sampling requires a training split and a known sampling mode.')
    keys = np.column_stack([dataset.classes.numpy(), dataset.xy_original])
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    weights = torch.from_numpy(1. / counts[inverse])
    return WeightedRandomSampler(weights, len(dataset), replacement=True, generator=generator)


def spatial_risk(output, target_xy):
    """W1 to a point target: sum_j p_j ||a_j-y||, in original normalized XY.

    The probability distribution spans ALL classes. Unlike the true-class
    coordinate loss, spatially distant wrong-class mass also receives gradient.
    No invented floor-height penalty or metric change is introduced.
    """
    distance = torch.cdist(target_xy, output['reference_xy'])
    return (output['reference_logits'].softmax(1) * distance).sum(1).mean()

"""Train-only augmentation between fingerprints measured at the SAME point."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class SamePointMixing(Dataset):
    def __init__(self, train, probability, *, final_refit=False, partner_policy='any_same_point'):
        expected = 'development' if final_refit else 'train'
        if train.split_name != expected or not 0 <= probability <= 1:
            raise ValueError(f'Same-point mixing requires {expected} and a probability in [0,1].')
        self.train, self.probability = train, probability
        if partner_policy not in ('any_same_point','different_phone_first'):
            raise ValueError(f'Unknown same-point partner policy: {partner_policy}')
        keys = np.column_stack([train.classes.numpy(), train.xy_original])
        _, membership = np.unique(keys, axis=0, return_inverse=True)
        members = {g: np.flatnonzero(membership == g) for g in np.unique(membership)}
        self.partners = [members[g][members[g] != i] for i, g in enumerate(membership)]
        if partner_policy == 'different_phone_first':
            phones = train.metadata['PHONEID']
            for i, candidates in enumerate(self.partners):
                cross_phone = candidates[phones[candidates] != phones[i]]
                if len(cross_phone):
                    self.partners[i] = cross_phone

    def __len__(self):
        return len(self.train)

    def __getitem__(self, index):
        x, label, xy = self.train[index]
        candidates = self.partners[index]
        if not len(candidates) or torch.rand(()) >= self.probability:
            return x, label, xy
        other = int(candidates[torch.randint(len(candidates), ()).item()])
        partner = self.train.features[other]
        alpha = .2 + .6 * torch.rand(())
        # Keep a binary presence mask. At APs detected by both sources, interpolate
        # RSSI. Otherwise choose one parent's signal+mask, preserving absence.
        choose = torch.rand(len(x)) < alpha
        mixed = torch.where(choose[:, None], x, partner)
        both = (x[:, 1] == 1) & (partner[:, 1] == 1)
        mixed[both, 0] = alpha*x[both, 0] + (1-alpha)*partner[both, 0]
        return mixed, label, xy


class SamePointPairs(Dataset):
    """Two train views for ONE network; partner metadata never enters forward."""
    def __init__(self, train, mixing_probability=0., *, final_refit=False,
                 partner_policy='different_phone_first'):
        self.sampler = SamePointMixing(train,mixing_probability,final_refit=final_refit,
                                      partner_policy=partner_policy)
        self.train = train

    def __len__(self):
        return len(self.train)

    def __getitem__(self,index):
        x,label,xy = self.sampler[index]
        candidates = self.sampler.partners[index]
        partner = int(candidates[torch.randint(len(candidates),()).item()]) if len(candidates) else index
        return x,label,xy,self.train.features[partner]


class NearbyPointMixing(Dataset):
    """Train-only spatial mixup; no neighbour lookup exists at inference.

    Partners have the same building/floor and a DIFFERENT original coordinate
    within radius_m in the unchanged native benchmark plane. Sample a point
    uniformly, then one fingerprint of that point. Interpolate the target as
    well as jointly observed RSSI; this is a hypothesis, not physical ground truth.
    """
    def __init__(self, train, probability, radius_m, *, final_refit=False):
        expected = 'development' if final_refit else 'train'
        if train.split_name != expected or not 0 <= probability <= 1:
            raise ValueError(f'Nearby-point mixing requires {expected} and probability in [0,1].')
        if not np.isfinite(radius_m) or radius_m <= 0:
            raise ValueError('Nearby-point radius must be positive and finite.')
        self.train, self.probability = train, probability
        keys = np.column_stack([train.classes.numpy(), train.xy_original])
        points, self.membership = np.unique(keys, axis=0, return_inverse=True)
        self.members = [np.flatnonzero(self.membership == group) for group in range(len(points))]
        self.neighbours = []
        for point in points:
            d2 = np.square(points[:, 1:] - point[1:]).sum(1)
            valid = ((points[:, 0] == point[0]) & np.any(points[:, 1:] != point[1:], axis=1)
                     & (d2 <= radius_m ** 2))
            self.neighbours.append(np.flatnonzero(valid))

    def __len__(self):
        return len(self.train)

    def __getitem__(self, index):
        x, label, xy = self.train[index]
        groups = self.neighbours[self.membership[index]]
        if not len(groups) or torch.rand(()) >= self.probability:
            return x, label, xy
        group = int(groups[torch.randint(len(groups), ()).item()])
        candidates = self.members[group]
        other = int(candidates[torch.randint(len(candidates), ()).item()])
        partner, _, partner_xy = self.train[other]
        alpha = .2 + .6 * torch.rand(())
        choose = torch.rand(len(x)) < alpha
        mixed = torch.where(choose[:, None], x, partner)
        both = (x[:, 1] == 1) & (partner[:, 1] == 1)
        mixed[both, 0] = alpha*x[both, 0] + (1-alpha)*partner[both, 0]
        return mixed, label, alpha*xy + (1-alpha)*partner_xy


def training_representation(train, settings, *, final_refit=False):
    """The same augmentation choice for train and post-selection refit."""
    consistency = settings.get('same_point_consistency_weight', 0.)
    same_point = settings.get('same_point_mixing', 0.)
    nearby = settings.get('nearby_point_mixing', 0.)
    if consistency < 0 or settings.get('consistency_temperature', 1.) <= 0:
        raise ValueError('Invalid same-point consistency hyperparameters.')
    if not 0 <= nearby <= 1:
        raise ValueError('nearby_point_mixing must be in [0,1].')
    if nearby:
        if consistency or same_point:
            raise ValueError('Nearby-point mixing is an alternative to same-point mixing/consistency.')
        return NearbyPointMixing(train, nearby, settings.get('nearby_point_radius_m', 10.), final_refit=final_refit)
    if consistency:
        return SamePointPairs(train, same_point, final_refit=final_refit,
            partner_policy=settings.get('same_point_partner_policy', 'different_phone_first'))
    if same_point:
        return SamePointMixing(train, same_point, final_refit=final_refit,
            partner_policy=settings.get('same_point_partner_policy', 'any_same_point'))
    return train

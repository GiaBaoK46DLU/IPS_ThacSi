"""Explicit univariate candidate functions for the IPS S-KAN experiment.

This is an independently implemented, declared IPS pool, not a claim to have
the unreleased authors' exact18-function factory. No extra normalization or
output activation is added to the V4 Mixer. Each candidate retains a SiLU
base path, and all nonlinear functions act separately on input coordinates.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from ...models.vendor.efficient_kan import KANLinear


def default_pool():
    return [
        {"name": "bspline_g5_k3", "kind": "bspline"},
        {"name": "gaussian_rbf8", "kind": "rbf", "count": 8},
        {"name": "gaussian_rbf16", "kind": "rbf", "count": 16},
        {"name": "chebyshev3", "kind": "chebyshev", "degree": 3},
        {"name": "chebyshev5", "kind": "chebyshev", "degree": 5},
        {"name": "gram3", "kind": "gram", "degree": 3},
        {"name": "gram5", "kind": "gram", "degree": 5},
        {"name": "jacobi3", "kind": "jacobi", "degree": 3, "alpha": 1., "beta": 1.},
        {"name": "jacobi5", "kind": "jacobi", "degree": 5, "alpha": 1., "beta": 1.},
        {"name": "bernstein3", "kind": "bernstein", "degree": 3},
        {"name": "bernstein5", "kind": "bernstein", "degree": 5},
        {"name": "mexican_hat", "kind": "mexican_hat"},
        {"name": "derivative_gaussian", "kind": "dog"},
        {"name": "morlet", "kind": "morlet"},
        {"name": "meyer_window", "kind": "meyer"},
        {"name": "shannon_bandpass", "kind": "shannon"},
        {"name": "relu_g5_k3", "kind": "relu", "grid_size": 5, "overlap": 3},
        {"name": "rswaf8", "kind": "rswaf", "count": 8},
    ]


class BasisFeatures(nn.Module):
    """Return [N,I,K] features; learned scales/shifts are per input node.

    Gram is the discrete Gram recurrence, initialized to its fixed recurrence
    and allowed to learn its beta coefficients. Shannon is the real bandpass
    form2*sinc(2u)-sinc(u), deliberately independent of batch/feature ordering.
    """
    def __init__(self, inputs, spec, grid_range):
        super().__init__()
        self.kind, self.spec = spec['kind'], dict(spec)
        self.inputs = inputs
        self.degree = int(spec.get('degree', 0))
        if self.kind in ('rbf', 'rswaf'):
            self.width = int(spec['count'])
            self.register_buffer('centers', torch.linspace(*grid_range, self.width))
            self.denominator = (grid_range[1]-grid_range[0])/(self.width-1)
        elif self.kind in ('chebyshev','gram','jacobi','bernstein'):
            self.width = self.degree+1
            if self.kind == 'gram':
                m = self.degree+1
                coefficients = [0., 0.] + [((m+n)*(m-n)*n*n)/(m*m*(4*n*n-1))
                                            for n in range(1,self.degree)]
                self.gram_beta = nn.Parameter(torch.tensor(coefficients)[None].repeat(inputs,1))
        elif self.kind == 'relu':
            grid, overlap = spec['grid_size'], spec['overlap']
            self.width = grid+overlap
            step = (grid_range[1]-grid_range[0])/grid
            low = grid_range[0]+torch.arange(-overlap,grid)*step
            self.low = nn.Parameter(low[None].repeat(inputs,1))
            initial_width = step*(overlap+1)
            self.log_width = nn.Parameter(torch.full((inputs,self.width),math.log(initial_width)))
        elif self.kind in ('mexican_hat','dog','morlet','meyer','shannon'):
            self.width = 1
            self.translation = nn.Parameter(torch.zeros(inputs))
            self.log_scale = nn.Parameter(torch.zeros(inputs))
        else:
            raise ValueError(f'Unknown selectable basis: {self.kind}')

    def forward(self, x):
        kind = self.kind
        if kind in ('rbf','rswaf'):
            u = (x[...,None]-self.centers)/self.denominator
            return torch.exp(-u.square()) if kind=='rbf' else 1-torch.tanh(u).square()
        if kind == 'relu':
            width = self.log_width.clamp(-4,4).exp()
            u = F.relu(x[...,None]-self.low)*F.relu(self.low+width-x[...,None])
            return (u*4/width.square()).square()
        if kind == 'bernstein':
            u=x.sigmoid()
            return torch.stack([math.comb(self.degree,j)*u.pow(j)*(1-u).pow(self.degree-j)
                                for j in range(self.degree+1)],-1)
        if kind in ('chebyshev','gram','jacobi'):
            u=x.tanh()
            values=[torch.ones_like(u)]
            if self.degree:
                if kind=='jacobi':
                    a,b=self.spec['alpha'],self.spec['beta']
                    values.append(((a-b)+(a+b+2)*u)/2)
                else:
                    values.append(u)
            for n in range(2,self.degree+1):
                if kind=='chebyshev':
                    value=2*u*values[-1]-values[-2]
                elif kind=='gram':
                    value=u*values[-1]-self.gram_beta[:,n]*values[-2]
                else:
                    a,b=self.spec['alpha'],self.spec['beta']
                    denominator=2*n*(n+a+b)*(2*n+a+b-2)
                    term1=(2*n+a+b-1)*((2*n+a+b)*(2*n+a+b-2)*u+a*a-b*b)*values[-1]
                    term2=2*(n+a-1)*(n+b-1)*(2*n+a+b)*values[-2]
                    value=(term1-term2)/denominator
                values.append(value)
            return torch.stack(values,-1)
        u=(x-self.translation)/self.log_scale.clamp(-4,4).exp()
        if kind=='mexican_hat':
            value=(2/(math.sqrt(3)*math.pi**.25))*(u.square()-1)*torch.exp(-.5*u.square())
        elif kind=='dog':
            value=-u*torch.exp(-.5*u.square())
        elif kind=='morlet':
            value=torch.cos(5*u)*torch.exp(-.5*u.square())
        elif kind=='meyer':
            v=u.abs();t=(2*v-1).clamp(0,1)
            nu=t.pow(4)*(35-84*t+70*t.square()-20*t.pow(3))
            window=torch.where(v<=.5,torch.ones_like(v),
                               torch.where(v>=1,torch.zeros_like(v),torch.cos(math.pi/2*nu)))
            value=torch.sin(math.pi*v)*window
        else:
            value=2*torch.sinc(2*u)-torch.sinc(u)
        return value[...,None]


class CandidateFunction(nn.Module):
    """One candidate family for all incoming coordinates, summed over edges."""
    def __init__(self, inputs, outputs, spec, kan):
        super().__init__()
        self.in_features, self.out_features = inputs, outputs
        self.spec = dict(spec)
        if spec['kind']=='bspline':
            options={key:kan[key] for key in ('grid_size','spline_order','scale_noise','scale_base',
                'scale_spline','enable_standalone_scale_spline','grid_eps','grid_range')}
            self.spline = KANLinear(inputs,outputs,**options)
        else:
            self.basis=BasisFeatures(inputs,spec,kan['grid_range'])
            self.base_weight=nn.Parameter(torch.empty(outputs,inputs))
            self.basis_weight=nn.Parameter(torch.empty(outputs,inputs,self.basis.width))
            self.basis_scaler=nn.Parameter(torch.empty(outputs,inputs))
            nn.init.kaiming_uniform_(self.base_weight,a=math.sqrt(5))
            nn.init.normal_(self.basis_weight,std=kan['scale_noise']/math.sqrt(inputs*self.basis.width))
            nn.init.kaiming_uniform_(self.basis_scaler,a=math.sqrt(5))

    def forward(self,x,input_weights=None):
        shape=x.shape;flat=x.reshape(-1,self.in_features)
        weights=1. if input_weights is None else input_weights
        if self.spec['kind']=='bspline':
            result=F.linear(F.silu(flat)*weights,self.spline.base_weight)
            basis=self.spline.b_splines(flat)*weights if input_weights is None else self.spline.b_splines(flat)*weights[None,:,None]
            result=result+F.linear(basis.reshape(len(flat),-1),self.spline.scaled_spline_weight.reshape(self.out_features,-1))
        else:
            features=self.basis(flat)
            if input_weights is not None:
                features=features*weights[None,:,None]
            result=F.linear(F.silu(flat)*weights,self.base_weight)
            result=result+F.linear(features.reshape(len(flat),-1),
                (self.basis_weight*self.basis_scaler[...,None]).reshape(self.out_features,-1))
        return result.reshape(*shape[:-1],self.out_features)

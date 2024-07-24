# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

import logging
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.nvtx import range as nvtx_range
from torch import Tensor
import dgl
from dgl import DGLGraph
from dgl.ops import edge_softmax
from dgl.nn.pytorch import AvgPooling, MaxPooling
from enum import Enum
import e3nn.o3 as o3
from functools import lru_cache

from typing import Optional, Literal, Dict, List, Union

# from .basis import get_basis, update_basis_with_fused
from collections import namedtuple
from itertools import product
from typing import Dict

import torch
from torch import Tensor

FiberEl = namedtuple("FiberEl", ["degree", "channels"])


def aggregate_residual(feats1, feats2, method: str):
    """Add or concatenate two fiber features together. If degrees don't match, will use the ones of feats2."""
    if method in ["add", "sum"]:
        return {k: (v + feats1[k]) if k in feats1 else v for k, v in feats2.items()}
    elif method in ["cat", "concat"]:
        return {
            k: torch.cat([v, feats1[k]], dim=1) if k in feats1 else v
            for k, v in feats2.items()
        }
    else:
        raise ValueError("Method must be add/sum or cat/concat")


def degree_to_dim(degree: int) -> int:
    return 2 * degree + 1


def unfuse_features(features: Tensor, degrees: List[int]) -> Dict[str, Tensor]:
    return dict(
        zip(
            map(str, degrees),
            features.split([degree_to_dim(deg) for deg in degrees], dim=-1),
        )
    )


def str2bool(v: Union[bool, str]) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_spherical_harmonics(relative_pos: Tensor, max_degree: int) -> List[Tensor]:
    all_degrees = list(range(2 * max_degree + 1))
    with nvtx_range("spherical harmonics"):
        sh = o3.spherical_harmonics(all_degrees, relative_pos, normalize=True)
        return torch.split(sh, [degree_to_dim(d) for d in all_degrees], dim=1)


@lru_cache(maxsize=None)
def get_clebsch_gordon(J: int, d_in: int, d_out: int, device) -> Tensor:
    """Get the (cached) Q^{d_out,d_in}_J matrices from equation (8)"""
    return o3.wigner_3j(J, d_in, d_out, dtype=torch.float64, device=device).permute(
        2, 1, 0
    )


@lru_cache(maxsize=None)
def get_all_clebsch_gordon(max_degree: int, device) -> List[List[Tensor]]:
    all_cb = []
    for d_in in range(max_degree + 1):
        for d_out in range(max_degree + 1):
            K_Js = []
            for J in range(abs(d_in - d_out), d_in + d_out + 1):
                K_Js.append(get_clebsch_gordon(J, d_in, d_out, device))
            all_cb.append(K_Js)
    return all_cb


@torch.jit.script
def get_basis_script(
    max_degree: int,
    use_pad_trick: bool,
    spherical_harmonics: List[Tensor],
    clebsch_gordon: List[List[Tensor]],
    amp: bool,
) -> Dict[str, Tensor]:
    """
    Compute pairwise bases matrices for degrees up to max_degree
    :param max_degree:            Maximum input or output degree
    :param use_pad_trick:         Pad some of the odd dimensions for a better use of Tensor Cores
    :param spherical_harmonics:   List of computed spherical harmonics
    :param clebsch_gordon:        List of computed CB-coefficients
    :param amp:                   When true, return bases in FP16 precision
    """
    basis = {}
    idx = 0
    # Double for loop instead of product() because of JIT script
    for d_in in range(max_degree + 1):
        for d_out in range(max_degree + 1):
            key = f"{d_in},{d_out}"
            K_Js = []
            for freq_idx, J in enumerate(range(abs(d_in - d_out), d_in + d_out + 1)):
                Q_J = clebsch_gordon[idx][freq_idx]
                K_Js.append(
                    torch.einsum(
                        "n f, k l f -> n l k",
                        spherical_harmonics[J].float(),
                        Q_J.float(),
                    )
                )

            basis[key] = torch.stack(K_Js, 2)  # Stack on second dim so order is n l f k
            if amp:
                basis[key] = basis[key].half()
            if use_pad_trick:
                basis[key] = F.pad(
                    basis[key], (0, 1)
                )  # Pad the k dimension, that can be sliced later

            idx += 1

    return basis


@torch.jit.script
def update_basis_with_fused(
    basis: Dict[str, Tensor], max_degree: int, use_pad_trick: bool, fully_fused: bool
) -> Dict[str, Tensor]:
    """Update the basis dict with partially and optionally fully fused bases"""
    num_edges = basis["0,0"].shape[0]
    device = basis["0,0"].device
    dtype = basis["0,0"].dtype
    sum_dim = torch.sum(torch.tensor([degree_to_dim(d) for d in range(max_degree + 1)]))

    # Fused per output degree
    for d_out in range(max_degree + 1):
        sum_freq = torch.sum(
            torch.tensor([degree_to_dim(min(d, d_out)) for d in range(max_degree + 1)])
        )
        basis_fused = torch.zeros(
            num_edges,
            sum_dim,
            sum_freq,
            degree_to_dim(d_out) + int(use_pad_trick),
            device=device,
            dtype=dtype,
        )
        acc_d, acc_f = 0, 0
        for d_in in range(max_degree + 1):
            basis_fused[
                :,
                acc_d : acc_d + degree_to_dim(d_in),
                acc_f : acc_f + degree_to_dim(min(d_out, d_in)),
                : degree_to_dim(d_out),
            ] = basis[f"{d_in},{d_out}"][:, :, :, : degree_to_dim(d_out)]

            acc_d += degree_to_dim(d_in)
            acc_f += degree_to_dim(min(d_out, d_in))

        basis[f"out{d_out}_fused"] = basis_fused

    # Fused per input degree
    for d_in in range(max_degree + 1):
        sum_freq = torch.sum(
            torch.tensor([degree_to_dim(min(d, d_in)) for d in range(max_degree + 1)])
        )
        basis_fused = torch.zeros(
            num_edges,
            degree_to_dim(d_in),
            sum_freq,
            sum_dim,
            device=device,
            dtype=dtype,
        )
        acc_d, acc_f = 0, 0
        for d_out in range(max_degree + 1):
            basis_fused[
                :,
                :,
                acc_f : acc_f + degree_to_dim(min(d_out, d_in)),
                acc_d : acc_d + degree_to_dim(d_out),
            ] = basis[f"{d_in},{d_out}"][:, :, :, : degree_to_dim(d_out)]

            acc_d += degree_to_dim(d_out)
            acc_f += degree_to_dim(min(d_out, d_in))

        basis[f"in{d_in}_fused"] = basis_fused

    if fully_fused:
        # Fully fused
        # Double sum this way because of JIT script
        # sum_freq = sum([
        #    torch.sum(torch.tensor([degree_to_dim(min(d_in, d_out)) for d_in in range(max_degree + 1)]) for d_out in range(max_degree + 1))
        # ])
        sum_freq = torch.sum(
            torch.tensor(
                [
                    torch.sum(
                        torch.tensor(
                            [
                                degree_to_dim(min(d_in, d_out))
                                for d_in in range(max_degree + 1)
                            ]
                        )
                    )
                    for d_out in range(max_degree + 1)
                ]
            )
        )
        # sum_freq = torch.sum(torch.tensor([degree_to_dim(min(d_in, d_out)) for d_in in range(max_degree + 1)]) for d_out in range(max_degree + 1))
        basis_fused = torch.zeros(
            num_edges, sum_dim, sum_freq, sum_dim, device=device, dtype=dtype
        )

        acc_d, acc_f = 0, 0
        for d_out in range(max_degree + 1):
            b = basis[f"out{d_out}_fused"]
            basis_fused[
                :, :, acc_f : acc_f + b.shape[2], acc_d : acc_d + degree_to_dim(d_out)
            ] = b[:, :, :, : degree_to_dim(d_out)]
            acc_f += b.shape[2]
            acc_d += degree_to_dim(d_out)

        basis["fully_fused"] = basis_fused

    del basis["0,0"]  # We know that the basis for l = k = 0 is filled with a constant
    return basis


def get_basis(
    relative_pos: Tensor,
    max_degree: int = 4,
    compute_gradients: bool = False,
    use_pad_trick: bool = False,
    amp: bool = False,
) -> Dict[str, Tensor]:
    with nvtx_range("spherical harmonics"):
        spherical_harmonics = get_spherical_harmonics(relative_pos, max_degree)
    with nvtx_range("CB coefficients"):
        clebsch_gordon = get_all_clebsch_gordon(max_degree, relative_pos.device)

    with torch.autograd.set_grad_enabled(compute_gradients):
        with nvtx_range("bases"):
            basis = get_basis_script(
                max_degree=max_degree,
                use_pad_trick=use_pad_trick,
                spherical_harmonics=spherical_harmonics,
                clebsch_gordon=clebsch_gordon,
                amp=amp,
            )
            return basis


class GPooling(nn.Module):
    """
    Graph max/average pooling on a given feature type.
    The average can be taken for any feature type, and equivariance will be maintained.
    The maximum can only be taken for invariant features (type 0).
    If you want max-pooling for type > 0 features, look into Vector Neurons.
    """

    def __init__(self, feat_type: int = 0, pool: Literal["max", "avg"] = "max"):
        """
        :param feat_type: Feature type to pool
        :param pool: Type of pooling: max or avg
        """
        super().__init__()
        assert pool in ["max", "avg"], f"Unknown pooling: {pool}"
        assert (
            feat_type == 0 or pool == "avg"
        ), "Max pooling on type > 0 features will break equivariance"
        self.feat_type = feat_type
        self.pool = MaxPooling() if pool == "max" else AvgPooling()

    def forward(self, features: Dict[str, Tensor], graph: DGLGraph, **kwargs) -> Tensor:
        pooled = self.pool(graph, features[str(self.feat_type)])
        return pooled.squeeze(dim=-1)


class Fiber(dict):
    """
    Describes the structure of some set of features.
    Features are split into types (0, 1, 2, 3, ...). A feature of type k has a dimension of 2k+1.
    Type-0 features: invariant scalars
    Type-1 features: equivariant 3D vectors
    Type-2 features: equivariant symmetric traceless matrices
    ...

    As inputs to a SE3 layer, there can be many features of the same types, and many features of different types.
    The 'multiplicity' or 'number of channels' is the number of features of a given type.
    This class puts together all the degrees and their multiplicities in order to describe
        the inputs, outputs or hidden features of SE3 layers.
    """

    def __init__(self, structure):
        if isinstance(structure, dict):
            structure = [
                FiberEl(int(d), int(m))
                for d, m in sorted(structure.items(), key=lambda x: x[1])
            ]
        elif not isinstance(structure[0], FiberEl):
            structure = list(
                map(lambda t: FiberEl(*t), sorted(structure, key=lambda x: x[1]))
            )
        self.structure = structure
        super().__init__({d: m for d, m in self.structure})

    @property
    def degrees(self):
        return sorted([t.degree for t in self.structure])

    @property
    def channels(self):
        return [self[d] for d in self.degrees]

    @property
    def num_features(self):
        """Size of the resulting tensor if all features were concatenated together"""
        return sum(t.channels * degree_to_dim(t.degree) for t in self.structure)

    @staticmethod
    def create(num_degrees: int, num_channels: int):
        """Create a Fiber with degrees 0..num_degrees-1, all with the same multiplicity"""
        return Fiber([(degree, num_channels) for degree in range(num_degrees)])

    @staticmethod
    def from_features(feats: Dict[str, Tensor]):
        """Infer the Fiber structure from a feature dict"""
        structure = {}
        for k, v in feats.items():
            degree = int(k)
            assert len(v.shape) == 3, "Feature shape should be (N, C, 2D+1)"
            assert v.shape[-1] == degree_to_dim(degree)
            structure[degree] = v.shape[-2]
        return Fiber(structure)

    def __getitem__(self, degree: int):
        """fiber[degree] returns the multiplicity for this degree"""
        return dict(self.structure).get(degree, 0)

    def __iter__(self):
        """Iterate over namedtuples (degree, channels)"""
        return iter(self.structure)

    def __mul__(self, other):
        """
        If other in an int, multiplies all the multiplicities by other.
        If other is a fiber, returns the cartesian product.
        """
        if isinstance(other, Fiber):
            return product(self.structure, other.structure)
        elif isinstance(other, int):
            return Fiber({t.degree: t.channels * other for t in self.structure})

    def __add__(self, other):
        """
        If other in an int, add other to all the multiplicities.
        If other is a fiber, add the multiplicities of the fibers together.
        """
        if isinstance(other, Fiber):
            return Fiber(
                {t.degree: t.channels + other[t.degree] for t in self.structure}
            )
        elif isinstance(other, int):
            return Fiber({t.degree: t.channels + other for t in self.structure})

    def __repr__(self):
        return str(self.structure)

    @staticmethod
    def combine_max(f1, f2):
        """Combine two fiber by taking the maximum multiplicity for each degree in both fibers"""
        new_dict = dict(f1.structure)
        for k, m in f2.structure:
            new_dict[k] = max(new_dict.get(k, 0), m)

        return Fiber(list(new_dict.items()))

    @staticmethod
    def combine_selectively(f1, f2):
        """Combine two fiber by taking the sum of multiplicities for each degree in the first fiber"""
        # only use orders which occur in fiber f1
        new_dict = dict(f1.structure)
        for k in f1.degrees:
            if k in f2.degrees:
                new_dict[k] += f2[k]
        return Fiber(list(new_dict.items()))

    def to_attention_heads(self, tensors: Dict[str, Tensor], num_heads: int):
        # dict(N, num_channels, 2d+1) -> (N, num_heads, -1)
        fibers = [
            tensors[str(degree)].reshape(
                *tensors[str(degree)].shape[:-2], num_heads, -1
            )
            for degree in self.degrees
        ]
        fibers = torch.cat(fibers, -1)
        return fibers


class ConvSE3FuseLevel(Enum):
    """
    Enum to select a maximum level of fusing optimizations that will be applied when certain conditions are met.
    If a desired level L is picked and the level L cannot be applied to a level, other fused ops < L are considered.
    A higher level means faster training, but also more memory usage.
    If you are tight on memory and want to feed large inputs to the network, choose a low value.
    If you want to train fast, choose a high value.
    Recommended value is FULL with AMP.

    Fully fused TFN convolutions requirements:
    - all input channels are the same
    - all output channels are the same
    - input degrees span the range [0, ..., max_degree]
    - output degrees span the range [0, ..., max_degree]

    Partially fused TFN convolutions requirements:
    * For fusing by output degree:
    - all input channels are the same
    - input degrees span the range [0, ..., max_degree]
    * For fusing by input degree:
    - all output channels are the same
    - output degrees span the range [0, ..., max_degree]

    Original TFN pairwise convolutions: no requirements
    """

    FULL = 2
    PARTIAL = 1
    NONE = 0


class RadialProfile(nn.Module):
    """
    Radial profile function.
    Outputs weights used to weigh basis matrices in order to get convolution kernels.
    In TFN notation: $R^{l,k}$
    In SE(3)-Transformer notation: $\phi^{l,k}$

    Note:
        In the original papers, this function only depends on relative node distances ||x||.
        Here, we allow this function to also take as input additional invariant edge features.
        This does not break equivariance and adds expressive power to the model.

    Diagram:
        invariant edge features (node distances included) ───> MLP layer (shared across edges) ───> radial weights
    """

    def __init__(
        self,
        num_freq: int,
        channels_in: int,
        channels_out: int,
        edge_dim: int = 1,
        mid_dim: int = 32,
        use_layer_norm: bool = False,
    ):
        """
        :param num_freq:         Number of frequencies
        :param channels_in:      Number of input channels
        :param channels_out:     Number of output channels
        :param edge_dim:         Number of invariant edge features (input to the radial function)
        :param mid_dim:          Size of the hidden MLP layers
        :param use_layer_norm:   Apply layer normalization between MLP layers
        """
        super().__init__()
        modules = [
            nn.Linear(edge_dim, mid_dim),
            nn.LayerNorm(mid_dim) if use_layer_norm else None,
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.LayerNorm(mid_dim) if use_layer_norm else None,
            nn.ReLU(),
            nn.Linear(mid_dim, num_freq * channels_in * channels_out, bias=False),
        ]

        self.net = nn.Sequential(*[m for m in modules if m is not None])

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)


class VersatileConvSE3(nn.Module):
    """
    Building block for TFN convolutions.
    This single module can be used for fully fused convolutions, partially fused convolutions, or pairwise convolutions.
    """

    def __init__(
        self,
        freq_sum: int,
        channels_in: int,
        channels_out: int,
        edge_dim: int,
        use_layer_norm: bool,
        fuse_level: ConvSE3FuseLevel,
    ):
        super().__init__()
        self.freq_sum = freq_sum
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.fuse_level = fuse_level
        self.radial_func = RadialProfile(
            num_freq=freq_sum,
            channels_in=channels_in,
            channels_out=channels_out,
            edge_dim=edge_dim,
            use_layer_norm=use_layer_norm,
        )

    def forward(self, features: Tensor, invariant_edge_feats: Tensor, basis: Tensor):
        with nvtx_range("VersatileConvSE3"):
            num_edges = features.shape[0]
            in_dim = features.shape[2]
            with nvtx_range("RadialProfile"):
                radial_weights = self.radial_func(invariant_edge_feats).view(
                    -1, self.channels_out, self.channels_in * self.freq_sum
                )

            if basis is not None:
                # This block performs the einsum n i l, n o i f, n l f k -> n o k
                out_dim = basis.shape[-1]
                if self.fuse_level != ConvSE3FuseLevel.FULL:
                    out_dim += out_dim % 2 - 1  # Account for padded basis
                basis_view = basis.view(num_edges, in_dim, -1)
                tmp = (features @ basis_view).view(num_edges, -1, basis.shape[-1])
                return (radial_weights @ tmp)[:, :, :out_dim]
            else:
                # k = l = 0 non-fused case
                return radial_weights @ features


class ConvSE3(nn.Module):
    """
    SE(3)-equivariant graph convolution (Tensor Field Network convolution).
    This convolution can map an arbitrary input Fiber to an arbitrary output Fiber, while preserving equivariance.
    Features of different degrees interact together to produce output features.

    Note 1:
        The option is given to not pool the output. This means that the convolution sum over neighbors will not be
        done, and the returned features will be edge features instead of node features.

    Note 2:
        Unlike the original paper and implementation, this convolution can handle edge feature of degree greater than 0.
        Input edge features are concatenated with input source node features before the kernel is applied.
    """

    def __init__(
        self,
        fiber_in: Fiber,
        fiber_out: Fiber,
        fiber_edge: Fiber,
        pool: bool = True,
        use_layer_norm: bool = False,
        self_interaction: bool = False,
        max_degree: int = 4,
        fuse_level: ConvSE3FuseLevel = ConvSE3FuseLevel.FULL,
        allow_fused_output: bool = False,
    ):
        """
        :param fiber_in:           Fiber describing the input features
        :param fiber_out:          Fiber describing the output features
        :param fiber_edge:         Fiber describing the edge features (node distances excluded)
        :param pool:               If True, compute final node features by averaging incoming edge features
        :param use_layer_norm:     Apply layer normalization between MLP layers
        :param self_interaction:   Apply self-interaction of nodes
        :param max_degree:         Maximum degree used in the bases computation
        :param fuse_level:         Maximum fuse level to use in TFN convolutions
        :param allow_fused_output: Allow the module to output a fused representation of features
        """
        super().__init__()
        self.pool = pool
        self.fiber_in = fiber_in
        self.fiber_out = fiber_out
        self.self_interaction = self_interaction
        self.max_degree = max_degree
        self.allow_fused_output = allow_fused_output

        # channels_in: account for the concatenation of edge features
        channels_in_set = set(
            [f.channels + fiber_edge[f.degree] * (f.degree > 0) for f in self.fiber_in]
        )
        channels_out_set = set([f.channels for f in self.fiber_out])
        unique_channels_in = len(channels_in_set) == 1
        unique_channels_out = len(channels_out_set) == 1
        degrees_up_to_max = list(range(max_degree + 1))
        common_args = dict(edge_dim=fiber_edge[0] + 1, use_layer_norm=use_layer_norm)

        if (
            fuse_level.value >= ConvSE3FuseLevel.FULL.value
            and unique_channels_in
            and fiber_in.degrees == degrees_up_to_max
            and unique_channels_out
            and fiber_out.degrees == degrees_up_to_max
        ):
            # Single fused convolution
            self.used_fuse_level = ConvSE3FuseLevel.FULL

            sum_freq = sum(
                [
                    degree_to_dim(min(d_in, d_out))
                    for d_in, d_out in product(degrees_up_to_max, degrees_up_to_max)
                ]
            )

            self.conv = VersatileConvSE3(
                sum_freq,
                list(channels_in_set)[0],
                list(channels_out_set)[0],
                fuse_level=self.used_fuse_level,
                **common_args,
            )

        elif (
            fuse_level.value >= ConvSE3FuseLevel.PARTIAL.value
            and unique_channels_in
            and fiber_in.degrees == degrees_up_to_max
        ):
            # Convolutions fused per output degree
            self.used_fuse_level = ConvSE3FuseLevel.PARTIAL
            self.conv_out = nn.ModuleDict()
            for d_out, c_out in fiber_out:
                sum_freq = sum([degree_to_dim(min(d_out, d)) for d in fiber_in.degrees])
                self.conv_out[str(d_out)] = VersatileConvSE3(
                    sum_freq,
                    list(channels_in_set)[0],
                    c_out,
                    fuse_level=self.used_fuse_level,
                    **common_args,
                )

        elif (
            fuse_level.value >= ConvSE3FuseLevel.PARTIAL.value
            and unique_channels_out
            and fiber_out.degrees == degrees_up_to_max
        ):
            # Convolutions fused per input degree
            self.used_fuse_level = ConvSE3FuseLevel.PARTIAL
            self.conv_in = nn.ModuleDict()
            for d_in, c_in in fiber_in:
                sum_freq = sum([degree_to_dim(min(d_in, d)) for d in fiber_out.degrees])
                self.conv_in[str(d_in)] = VersatileConvSE3(
                    sum_freq,
                    c_in,
                    list(channels_out_set)[0],
                    fuse_level=self.used_fuse_level,
                    **common_args,
                )
        else:
            # Use pairwise TFN convolutions
            self.used_fuse_level = ConvSE3FuseLevel.NONE
            self.conv = nn.ModuleDict()
            for (degree_in, channels_in), (degree_out, channels_out) in (
                self.fiber_in * self.fiber_out
            ):
                dict_key = f"{degree_in},{degree_out}"
                channels_in_new = channels_in + fiber_edge[degree_in] * (degree_in > 0)
                sum_freq = degree_to_dim(min(degree_in, degree_out))
                self.conv[dict_key] = VersatileConvSE3(
                    sum_freq,
                    channels_in_new,
                    channels_out,
                    fuse_level=self.used_fuse_level,
                    **common_args,
                )

        if self_interaction:
            self.to_kernel_self = nn.ParameterDict()
            for degree_out, channels_out in fiber_out:
                if fiber_in[degree_out]:
                    self.to_kernel_self[str(degree_out)] = nn.Parameter(
                        torch.randn(channels_out, fiber_in[degree_out])
                        / np.sqrt(fiber_in[degree_out])
                    )

    def forward(
        self,
        node_feats: Dict[str, Tensor],
        edge_feats: Dict[str, Tensor],
        graph: DGLGraph,
        basis: Dict[str, Tensor],
    ):
        with nvtx_range("ConvSE3"):
            invariant_edge_feats = edge_feats["0"].squeeze(-1)
            src, dst = graph.edges()
            out = {}
            in_features = []

            # Fetch all input features from edge and node features
            for degree_in in self.fiber_in.degrees:
                src_node_features = node_feats[str(degree_in)][src]
                if degree_in > 0 and str(degree_in) in edge_feats:
                    # Handle edge features of any type by concatenating them to node features
                    src_node_features = torch.cat(
                        [src_node_features, edge_feats[str(degree_in)]], dim=1
                    )
                in_features.append(src_node_features)

            if self.used_fuse_level == ConvSE3FuseLevel.FULL:
                in_features_fused = torch.cat(in_features, dim=-1)
                out = self.conv(
                    in_features_fused, invariant_edge_feats, basis["fully_fused"]
                )

                if not self.allow_fused_output or self.self_interaction or self.pool:
                    out = unfuse_features(out, self.fiber_out.degrees)

            elif self.used_fuse_level == ConvSE3FuseLevel.PARTIAL and hasattr(
                self, "conv_out"
            ):
                in_features_fused = torch.cat(in_features, dim=-1)
                for degree_out in self.fiber_out.degrees:
                    out[str(degree_out)] = self.conv_out[str(degree_out)](
                        in_features_fused,
                        invariant_edge_feats,
                        basis[f"out{degree_out}_fused"],
                    )

            elif self.used_fuse_level == ConvSE3FuseLevel.PARTIAL and hasattr(
                self, "conv_in"
            ):
                out = 0
                for degree_in, feature in zip(self.fiber_in.degrees, in_features):
                    out += self.conv_in[str(degree_in)](
                        feature, invariant_edge_feats, basis[f"in{degree_in}_fused"]
                    )
                if not self.allow_fused_output or self.self_interaction or self.pool:
                    out = unfuse_features(out, self.fiber_out.degrees)
            else:
                # Fallback to pairwise TFN convolutions
                for degree_out in self.fiber_out.degrees:
                    out_feature = 0
                    for degree_in, feature in zip(self.fiber_in.degrees, in_features):
                        dict_key = f"{degree_in},{degree_out}"
                        out_feature = out_feature + self.conv[dict_key](
                            feature, invariant_edge_feats, basis.get(dict_key, None)
                        )
                    out[str(degree_out)] = out_feature

            for degree_out in self.fiber_out.degrees:
                if self.self_interaction and str(degree_out) in self.to_kernel_self:
                    with nvtx_range("self interaction"):
                        dst_features = node_feats[str(degree_out)][dst]
                        kernel_self = self.to_kernel_self[str(degree_out)]
                        out[str(degree_out)] += kernel_self @ dst_features

                if self.pool:
                    with nvtx_range("pooling"):
                        if isinstance(out, dict):
                            out[str(degree_out)] = dgl.ops.copy_e_sum(
                                graph, out[str(degree_out)]
                            )
                        else:
                            out = dgl.ops.copy_e_sum(graph, out)
                        # if isinstance(out, dict):
                        #     out[str(degree_out)] = dgl.ops.copy_e_mean(graph, out[str(degree_out)])
                        # else:
                        #     out = dgl.ops.copy_e_mean(graph, out)
            return out


class AttentionSE3(nn.Module):
    """Multi-headed sparse graph self-attention (SE(3)-equivariant)"""

    def __init__(self, num_heads: int, key_fiber: Fiber, value_fiber: Fiber):
        """
        :param num_heads:     Number of attention heads
        :param key_fiber:     Fiber for the keys (and also for the queries)
        :param value_fiber:   Fiber for the values
        """
        super().__init__()
        self.num_heads = num_heads
        self.key_fiber = key_fiber
        self.value_fiber = value_fiber

    def forward(
        self,
        value: Union[Tensor, Dict[str, Tensor]],  # edge features (may be fused)
        key: Union[Tensor, Dict[str, Tensor]],  # edge features (may be fused)
        query: Dict[str, Tensor],  # node features
        graph: DGLGraph,
    ):
        with nvtx_range("AttentionSE3"):
            with nvtx_range("reshape keys and queries"):
                if isinstance(key, Tensor):
                    # case where features of all types are fused
                    key = key.reshape(key.shape[0], self.num_heads, -1)
                    # need to reshape queries that way to keep the same layout as keys
                    out = torch.cat(
                        [query[str(d)] for d in self.key_fiber.degrees], dim=-1
                    )
                    query = out.reshape(
                        list(query.values())[0].shape[0], self.num_heads, -1
                    )
                else:
                    # features are not fused, need to fuse and reshape them
                    key = self.key_fiber.to_attention_heads(key, self.num_heads)
                    query = self.key_fiber.to_attention_heads(query, self.num_heads)

            with nvtx_range("attention dot product + softmax"):
                # Compute attention weights (softmax of inner product between key and query)
                edge_weights = dgl.ops.e_dot_v(graph, key, query).squeeze(-1)
                edge_weights /= np.sqrt(self.key_fiber.num_features)
                edge_weights = edge_softmax(graph, edge_weights)
                edge_weights = edge_weights[..., None, None]

            with nvtx_range("weighted sum"):
                if isinstance(value, Tensor):
                    # features of all types are fused
                    v = value.view(value.shape[0], self.num_heads, -1, value.shape[-1])
                    weights = edge_weights * v
                    feat_out = dgl.ops.copy_e_sum(graph, weights)
                    feat_out = feat_out.view(
                        feat_out.shape[0], -1, feat_out.shape[-1]
                    )  # merge heads
                    out = unfuse_features(feat_out, self.value_fiber.degrees)
                else:
                    out = {}
                    for degree, channels in self.value_fiber:
                        v = value[str(degree)].view(
                            -1,
                            self.num_heads,
                            channels // self.num_heads,
                            degree_to_dim(degree),
                        )
                        weights = edge_weights * v
                        res = dgl.ops.copy_e_sum(graph, weights)
                        out[str(degree)] = res.view(
                            -1, channels, degree_to_dim(degree)
                        )  # merge heads

                return out


class LinearSE3(nn.Module):
    """
    Graph Linear SE(3)-equivariant layer, equivalent to a 1x1 convolution.
    Maps a fiber to a fiber with the same degrees (channels may be different).
    No interaction between degrees, but interaction between channels.

    type-0 features (C_0 channels) ────> Linear(bias=False) ────> type-0 features (C'_0 channels)
    type-1 features (C_1 channels) ────> Linear(bias=False) ────> type-1 features (C'_1 channels)
                                                 :
    type-k features (C_k channels) ────> Linear(bias=False) ────> type-k features (C'_k channels)
    """

    def __init__(self, fiber_in: Fiber, fiber_out: Fiber):
        super().__init__()
        self.weights = nn.ParameterDict(
            {
                str(degree_out): nn.Parameter(
                    torch.randn(channels_out, fiber_in[degree_out])
                    / np.sqrt(fiber_in[degree_out])
                )
                for degree_out, channels_out in fiber_out
            }
        )

    def forward(
        self, features: Dict[str, Tensor], *args, **kwargs
    ) -> Dict[str, Tensor]:
        return {
            degree: self.weights[degree] @ features[degree]
            for degree, weight in self.weights.items()
        }


class AttentionBlockSE3(nn.Module):
    """Multi-headed sparse graph self-attention block with skip connection, linear projection (SE(3)-equivariant)"""

    def __init__(
        self,
        fiber_in: Fiber,
        fiber_out: Fiber,
        fiber_edge: Optional[Fiber] = None,
        num_heads: int = 4,
        channels_div: int = 2,
        use_layer_norm: bool = False,
        max_degree: bool = 4,
        fuse_level: ConvSE3FuseLevel = ConvSE3FuseLevel.FULL,
        **kwargs,
    ):
        """
        :param fiber_in:         Fiber describing the input features
        :param fiber_out:        Fiber describing the output features
        :param fiber_edge:       Fiber describing the edge features (node distances excluded)
        :param num_heads:        Number of attention heads
        :param channels_div:     Divide the channels by this integer for computing values
        :param use_layer_norm:   Apply layer normalization between MLP layers
        :param max_degree:       Maximum degree used in the bases computation
        :param fuse_level:       Maximum fuse level to use in TFN convolutions
        """
        super().__init__()
        if fiber_edge is None:
            fiber_edge = Fiber({})
        self.fiber_in = fiber_in
        # value_fiber has same structure as fiber_out but #channels divided by 'channels_div'
        value_fiber = Fiber(
            [(degree, channels // channels_div) for degree, channels in fiber_out]
        )
        # key_query_fiber has the same structure as fiber_out, but only degrees which are in in_fiber
        # (queries are merely projected, hence degrees have to match input)
        key_query_fiber = Fiber(
            [
                (fe.degree, fe.channels)
                for fe in value_fiber
                if fe.degree in fiber_in.degrees
            ]
        )

        self.to_key_value = ConvSE3(
            fiber_in,
            value_fiber + key_query_fiber,
            pool=False,
            fiber_edge=fiber_edge,
            use_layer_norm=use_layer_norm,
            max_degree=max_degree,
            fuse_level=fuse_level,
            allow_fused_output=True,
        )
        self.to_query = LinearSE3(fiber_in, key_query_fiber)
        self.attention = AttentionSE3(num_heads, key_query_fiber, value_fiber)
        self.project = LinearSE3(value_fiber + fiber_in, fiber_out)

    def forward(
        self,
        node_features: Dict[str, Tensor],
        edge_features: Dict[str, Tensor],
        graph: DGLGraph,
        basis: Dict[str, Tensor],
    ):
        with nvtx_range("AttentionBlockSE3"):
            with nvtx_range("keys / values"):
                fused_key_value = self.to_key_value(
                    node_features, edge_features, graph, basis
                )
                key, value = self._get_key_value_from_fused(fused_key_value)

            with nvtx_range("queries"):
                query = self.to_query(node_features)

            z = self.attention(value, key, query, graph)
            z_concat = aggregate_residual(node_features, z, "cat")
            return self.project(z_concat)

    def _get_key_value_from_fused(self, fused_key_value):
        # Extract keys and queries features from fused features
        if isinstance(fused_key_value, Tensor):
            # Previous layer was a fully fused convolution
            value, key = torch.chunk(fused_key_value, chunks=2, dim=-2)
        else:
            key, value = {}, {}
            for degree, feat in fused_key_value.items():
                if int(degree) in self.fiber_in.degrees:
                    value[degree], key[degree] = torch.chunk(feat, chunks=2, dim=-2)
                else:
                    value[degree] = feat

        return key, value


class NormSE3(nn.Module):
    """
    Norm-based SE(3)-equivariant nonlinearity.

                 ┌──> feature_norm ──> LayerNorm() ──> ReLU() ──┐
    feature_in ──┤                                              * ──> feature_out
                 └──> feature_phase ────────────────────────────┘
    """

    NORM_CLAMP = 2**-24  # Minimum positive subnormal for FP16

    def __init__(self, fiber: Fiber, nonlinearity: nn.Module = nn.ReLU()):
        super().__init__()
        self.fiber = fiber
        self.nonlinearity = nonlinearity

        if len(set(fiber.channels)) == 1:
            # Fuse all the layer normalizations into a group normalization
            self.group_norm = nn.GroupNorm(
                num_groups=len(fiber.degrees), num_channels=sum(fiber.channels)
            )
        else:
            # Use multiple layer normalizations
            self.layer_norms = nn.ModuleDict(
                {str(degree): nn.LayerNorm(channels) for degree, channels in fiber}
            )

    def forward(
        self, features: Dict[str, Tensor], *args, **kwargs
    ) -> Dict[str, Tensor]:
        with nvtx_range("NormSE3"):
            output = {}
            if hasattr(self, "group_norm"):
                # Compute per-degree norms of features
                norms = [
                    features[str(d)]
                    .norm(dim=-1, keepdim=True)
                    .clamp(min=self.NORM_CLAMP)
                    for d in self.fiber.degrees
                ]
                fused_norms = torch.cat(norms, dim=-2)

                # Transform the norms only
                new_norms = self.nonlinearity(
                    self.group_norm(fused_norms.squeeze(-1))
                ).unsqueeze(-1)
                new_norms = torch.chunk(
                    new_norms, chunks=len(self.fiber.degrees), dim=-2
                )

                # Scale features to the new norms
                for norm, new_norm, d in zip(norms, new_norms, self.fiber.degrees):
                    output[str(d)] = features[str(d)] / norm * new_norm
            else:
                for degree, feat in features.items():
                    norm = feat.norm(dim=-1, keepdim=True).clamp(min=self.NORM_CLAMP)
                    new_norm = self.nonlinearity(
                        self.layer_norms[degree](norm.squeeze(-1)).unsqueeze(-1)
                    )
                    output[degree] = new_norm * feat / norm

            return output


class Sequential(nn.Sequential):
    """Sequential module with arbitrary forward args and kwargs. Used to pass graph, basis and edge features."""

    def forward(self, input, *args, **kwargs):
        for module in self:
            input = module(input, *args, **kwargs)
        return input


def get_populated_edge_features(
    relative_pos: Tensor, edge_features: Optional[Dict[str, Tensor]] = None
):
    """Add relative positions to existing edge features"""
    edge_features = edge_features.copy() if edge_features else {}
    r = relative_pos.norm(dim=-1, keepdim=True)
    if "0" in edge_features:
        edge_features["0"] = torch.cat([edge_features["0"], r[..., None]], dim=1)
    else:
        edge_features["0"] = r[..., None]

    return edge_features


class SE3Transformer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        fiber_in: Fiber,
        fiber_hidden: Fiber,
        fiber_out: Fiber,
        num_heads: int,
        channels_div: int,
        fiber_edge: Fiber = Fiber({}),
        return_type: Optional[int] = None,
        pooling: Optional[Literal["avg", "max"]] = None,
        norm: bool = True,
        use_layer_norm: bool = True,
        tensor_cores: bool = False,
        low_memory: bool = False,
        **kwargs,
    ):
        """
        :param num_layers:          Number of attention layers
        :param fiber_in:            Input fiber description
        :param fiber_hidden:        Hidden fiber description
        :param fiber_out:           Output fiber description
        :param fiber_edge:          Input edge fiber description
        :param num_heads:           Number of attention heads
        :param channels_div:        Channels division before feeding to attention layer
        :param return_type:         Return only features of this type
        :param pooling:             'avg' or 'max' graph pooling before MLP layers
        :param norm:                Apply a normalization layer after each attention block
        :param use_layer_norm:      Apply layer normalization between MLP layers
        :param tensor_cores:        True if using Tensor Cores (affects the use of fully fused convs, and padded bases)
        :param low_memory:          If True, will use slower ops that use less memory
        """
        super().__init__()
        self.num_layers = num_layers
        self.fiber_edge = fiber_edge
        self.num_heads = num_heads
        self.channels_div = channels_div
        self.return_type = return_type
        self.pooling = pooling
        self.max_degree = max(
            *fiber_in.degrees, *fiber_hidden.degrees, *fiber_out.degrees
        )
        self.tensor_cores = tensor_cores
        self.low_memory = low_memory

        if low_memory and not tensor_cores:
            logging.warning("Low memory mode will have no effect with no Tensor Cores")

        # Fully fused convolutions when using Tensor Cores (and not low memory mode)
        fuse_level = (
            ConvSE3FuseLevel.FULL
            if tensor_cores and not low_memory
            else ConvSE3FuseLevel.PARTIAL
        )

        graph_modules = []
        for i in range(num_layers):
            graph_modules.append(
                AttentionBlockSE3(
                    fiber_in=fiber_in,
                    fiber_out=fiber_hidden,
                    fiber_edge=fiber_edge,
                    num_heads=num_heads,
                    channels_div=channels_div,
                    use_layer_norm=use_layer_norm,
                    max_degree=self.max_degree,
                    fuse_level=fuse_level,
                )
            )
            if norm:
                graph_modules.append(NormSE3(fiber_hidden))
            fiber_in = fiber_hidden

        graph_modules.append(
            ConvSE3(
                fiber_in=fiber_in,
                fiber_out=fiber_out,
                fiber_edge=fiber_edge,
                self_interaction=True,
                use_layer_norm=use_layer_norm,
                max_degree=self.max_degree,
            )
        )
        self.graph_modules = Sequential(*graph_modules)

        if pooling is not None:
            assert return_type is not None, "return_type must be specified when pooling"
            self.pooling_module = GPooling(pool=pooling, feat_type=return_type)

    def forward(
        self,
        graph: DGLGraph,
        node_feats: Dict[str, Tensor],
        edge_feats: Optional[Dict[str, Tensor]] = None,
        basis: Optional[Dict[str, Tensor]] = None,
    ):
        # Compute bases in case they weren't precomputed as part of the data loading
        basis = basis or get_basis(
            graph.edata["rel_pos"],
            max_degree=self.max_degree,
            compute_gradients=False,
            use_pad_trick=self.tensor_cores and not self.low_memory,
            amp=torch.is_autocast_enabled(),
        )

        # Add fused bases (per output degree, per input degree, and fully fused) to the dict
        basis = update_basis_with_fused(
            basis,
            self.max_degree,
            use_pad_trick=self.tensor_cores and not self.low_memory,
            fully_fused=self.tensor_cores and not self.low_memory,
        )

        edge_feats = get_populated_edge_features(graph.edata["rel_pos"], edge_feats)

        node_feats = self.graph_modules(
            node_feats, edge_feats, graph=graph, basis=basis
        )

        if self.pooling is not None:
            return self.pooling_module(node_feats, graph=graph)

        if self.return_type is not None:
            return node_feats[str(self.return_type)]

        return node_feats

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument(
            "--num_layers",
            type=int,
            default=7,
            help="Number of stacked Transformer layers",
        )
        parser.add_argument(
            "--num_heads", type=int, default=8, help="Number of heads in self-attention"
        )
        parser.add_argument(
            "--channels_div",
            type=int,
            default=2,
            help="Channels division before feeding to attention layer",
        )
        parser.add_argument(
            "--pooling",
            type=str,
            default=None,
            const=None,
            nargs="?",
            choices=["max", "avg"],
            help="Type of graph pooling",
        )
        parser.add_argument(
            "--norm",
            type=str2bool,
            nargs="?",
            const=True,
            default=False,
            help="Apply a normalization layer after each attention block",
        )
        parser.add_argument(
            "--use_layer_norm",
            type=str2bool,
            nargs="?",
            const=True,
            default=False,
            help="Apply layer normalization between MLP layers",
        )
        parser.add_argument(
            "--low_memory",
            type=str2bool,
            nargs="?",
            const=True,
            default=False,
            help="If true, will use fused ops that are slower but that use less memory "
            "(expect 25 percent less memory). "
            "Only has an effect if AMP is enabled on Volta GPUs, or if running on Ampere GPUs",
        )

        return parser


class SE3TransformerPooled(nn.Module):
    def __init__(
        self,
        fiber_in: Fiber,
        fiber_out: Fiber,
        fiber_edge: Fiber,
        num_degrees: int,
        num_channels: int,
        output_dim: int,
        **kwargs,
    ):
        super().__init__()
        kwargs["pooling"] = kwargs["pooling"] or "max"
        self.transformer = SE3Transformer(
            fiber_in=fiber_in,
            fiber_hidden=Fiber.create(num_degrees, num_channels),
            fiber_out=fiber_out,
            fiber_edge=fiber_edge,
            return_type=0,
            **kwargs,
        )

        n_out_features = fiber_out.num_features
        self.mlp = nn.Sequential(
            nn.Linear(n_out_features, n_out_features),
            nn.ReLU(),
            nn.Linear(n_out_features, output_dim),
        )

    def forward(self, graph, node_feats, edge_feats, basis=None):
        feats = self.transformer(graph, node_feats, edge_feats, basis).squeeze(-1)
        y = self.mlp(feats).squeeze(-1)
        return y

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("Model architecture")
        SE3Transformer.add_argparse_args(parser)
        parser.add_argument(
            "--num_degrees",
            help="Number of degrees to use. Hidden features will have types [0, ..., num_degrees - 1]",
            type=int,
            default=4,
        )
        parser.add_argument(
            "--num_channels",
            help="Number of channels for the hidden features",
            type=int,
            default=32,
        )
        return parent_parser

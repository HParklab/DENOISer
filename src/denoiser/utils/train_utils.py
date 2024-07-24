import os
import torch
from torch import nn, Tensor
import time
import subprocess as sp
import random
import numpy as np
from pathlib import Path

from typing import Union


def set_random_seed() -> None:
    seed = 1004
    deterministic = False

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def cosine_schedule(
    total_timesteps: int,
    cosine_timesteps: int,
    clip_min_max: tuple = (0.0001, 0.8),
    s: float = 0.008,
) -> Tensor:
    f = lambda t: torch.cos((t / cosine_timesteps + s) / (1 + s) * 0.5 * torch.pi) ** 2
    x = torch.linspace(0, cosine_timesteps, cosine_timesteps + 1)
    alphas_cumprod = f(x) / f(torch.tensor([0]))
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = torch.clip(betas, clip_min_max[0], clip_min_max[1])
    betas = torch.concat(
        (betas, torch.full((total_timesteps - cosine_timesteps,), clip_min_max[1]))
    )
    return 1 - betas


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cur_dir_exist(file: str) -> bool:
    return file in os.listdir("./")


def files_only_pdb(path: Union[str, Path]) -> list:
    if isinstance(path, str):
        path = Path(path)
    return [t.name for t in path.glob("*.pdb")]


def pairwise_2d(input_1d: Tensor) -> Tensor:
    repeated = input_1d.repeat(len(input_1d), 1)
    rows, cols = repeated.shape
    u_mask = torch.triu(torch.ones(rows, cols), diagonal=1)
    l_mask = torch.tril(torch.ones(rows, cols), diagonal=-1)
    upper_triangular_elements = repeated[u_mask == 1]
    lower_triangular_elements = repeated.T[l_mask.T == 1]
    return torch.cat(
        [upper_triangular_elements[None,], lower_triangular_elements[None,]], dim=0
    )


def split_energy(info: dict, scaling: bool = True) -> tuple:
    """Return vdw, solv, elec"""
    all_energy = info["rosetta_e"]
    vdw = all_energy[:, 0]
    solv = all_energy[:, 1]
    coulomb = all_energy[:, 2]
    hbond = all_energy[:, 3]
    elec = coulomb + hbond
    if scaling:
        vdw /= 20
        solv /= 10
        elec /= 5
    return vdw, solv, elec


def upsample_auxil(fnat: np.ndarray) -> np.ndarray:
    over05 = (fnat > 0.5).astype(float)
    over09 = (fnat > 0.85).astype(float)
    p = over05 + over09
    return p / np.sum(p)


def filter_clash(vdw: np.ndarray) -> np.ndarray:
    p = vdw < 20
    return p / np.sum(p)

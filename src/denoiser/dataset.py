import sys
import os
from pathlib import Path
import numpy as np
import dgl

import torch
from torch import Tensor
from torch.utils import data
from torch.utils.data.sampler import Sampler
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from copy import deepcopy
from easydict import EasyDict
from typing import Tuple, Optional, Tuple, Callable, Dict

from denoiser.utils.chem_utils import (
    N_AATYPE,
    gentype2num,
    find_gentype2num,
    cosine_angle,
)


class LocalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        targets,
        root_dir: Optional[Path] = None,
        ball_radius: int = 10,
        high_ball_radius: int = 6,
        randomize: float = 0.0,
        upsample: Optional[Callable] = None,
        num_channels: int = 32,
        sasa_method: str = "none",
        edgemode: str = "topk",
        edgek: Tuple[int, int] = (12, 12),
        edgedist: Tuple[float, float, float] = (8.0, 4.5, 3.0),
        ballmode: str = "com",
        distance_feat: str = "std",
        normalize_q: bool = False,
        nsamples_per_p=1,
        sample_mode: str = "random",
        use_AF: bool = False,
        subset_size: int = 1,
        masking_coeff: int = 0,
        all_cross: bool = False,
        training: bool = False,
        distance_map_dist: float = 5.0,
        scaling_apol_pol: bool = False,
        scaling_all: bool = False,
        AF_plddt_path: Optional[Path] = None,
        extra_edgefeat: bool = False,
        ros_dir: Optional[Path] = None,
        atom_property: Optional[Path] = None,
    ):
        self.proteins = targets
        self.datadir = root_dir
        self.ball_radius = ball_radius
        self.high_ball_radius = high_ball_radius
        self.randomize = randomize
        self.sasa_method = sasa_method
        self.num_channels = num_channels
        self.ballmode = ballmode  # ["com","all"]
        self.dist_fn_res = lambda x: self.get_dist_neighbors(
            x, mode=edgemode, top_k=edgek[0], dcut=edgedist[0]
        )
        self.dist_fn_atm = lambda x: self.get_dist_neighbors(
            x, mode=edgemode, top_k=edgek[1], dcut=edgedist[1]
        )
        self.dist_fn_high = lambda x: self.get_dist_neighbors(
            x, mode=edgemode, top_k=edgek[1], dcut=edgedist[2]
        )
        self.distance_feat = distance_feat
        self.nsamples_per_p = nsamples_per_p
        self.nsamples = max(1, len(self.proteins) * nsamples_per_p)
        self.use_AF = use_AF
        self.subset_size = subset_size
        self.masking_coeff = masking_coeff
        self.all_cross = all_cross
        self.distance_map_dist = distance_map_dist
        self.extra_edgefeat = extra_edgefeat

        self.normalize_q = normalize_q
        self.sample_mode = sample_mode
        self.training = training
        self.scaling_apol_pol = scaling_apol_pol
        self.scaling_all = scaling_all
        self.rosdata_dir = ros_dir
        self.atom_property = atom_property
        self.W, self.S = self.vdw_radius_depth

        self.AF_plddt_path = AF_plddt_path

        if upsample is None:
            self.upsample = sample_uniform
        else:
            self.upsample = upsample

    def __len__(self):
        return int(self.nsamples)

    def __getitem__(self, index):
        # Select a sample decoy
        info = {}
        info["sname"] = "none"
        if self.sample_mode == "serial":
            ip = int(index / self.nsamples_per_p)
            pname = self.proteins[ip]
        else:
            pname = self.proteins[index]
        info["pname"] = pname

        fname = self.datadir / (pname + ".lig.npz")
        if not os.path.exists(fname):
            return LocalDataset._skip_getitem(info)
        try:
            if self.rosdata_dir is not None:
                samples_ros = self.get_all_energy(pname)
            samples, pindex = self.get_a_sample(pname, index)
            prop = np.load(self.datadir / (pname + ".prop.npz"))
        except:
            print("Files does not exist!", self.datadir / pname)
            return LocalDataset._skip_getitem(info)

        sname = samples["name"][pindex].reshape(self.subset_size, -1)
        info["pindex"] = pindex
        info["sname"] = sname

        # Receptor features (prop)
        charges_rec, atypes_rec, aas_rec, repsatm_idx, r2a, sasa_rec, reschain = (
            self.receptor_features(prop)
        )

        # Ligand features (lig)
        (
            xyz_ligs,
            xyz_recs,
            lddt,
            fnat,
            atypes_lig,
            bnds_lig,
            charges_lig,
            aas_lig,
            repsatm_lig,
        ) = self.per_ligand_features(samples, pindex)

        # Rosetta_energy
        if self.rosdata_dir is not None:
            rosetta_e = self.select_energy(samples_ros, sname)
            if rosetta_e is None:
                return LocalDataset._skip_getitem(info)
        else:
            rosetta_e = [[0.0, 0.0, 0.0, 0.0] for _ in range(len(sname))]

        aas = np.concatenate([aas_lig, aas_rec]).astype(int)
        aas1hot = np.eye(N_AATYPE)[aas]

        # Representative atoms
        r2a = np.concatenate([np.array([0 for _ in range(xyz_ligs.shape[1])]), r2a])
        r2a1hot = np.eye(max(r2a) + 1)[r2a]
        repsatm_idx = np.concatenate(
            [
                np.array(repsatm_lig),
                np.array(repsatm_idx, dtype=int) + xyz_ligs.shape[1],
            ]
        )  # shape: (subset_size, num_atom_rep)

        # Bond properties
        bnds_rec = prop["bnds_rec"] + xyz_ligs.shape[1]  # shift index;

        # Concatenate receptor & ligand: ligand comes first
        charges = np.expand_dims(np.concatenate([charges_lig, charges_rec]), axis=1)
        if self.normalize_q:
            charges = 1.0 / (1.0 + np.exp(-2.0 * charges))

        islig = np.array(
            [1 for _ in range(xyz_ligs.shape[1])]
            + [0 for _ in range(xyz_recs.shape[1])]
        )
        islig = np.expand_dims(islig, axis=1)

        xyz = np.concatenate([xyz_ligs, xyz_recs], axis=1)
        atypes = np.concatenate([atypes_lig, atypes_rec])
        if "Null" in atypes:
            print("There is 'Null' atom types in the PDB file. Skip this.")
            return LocalDataset._skip_getitem(info)
        w_vdw = torch.tensor([self.W[atype] for atype in list(atypes)])  # vdw depth
        s_vdw = torch.tensor([self.S[atype] for atype in list(atypes)])  # vdw radius

        atypes_int = np.array(
            [find_gentype2num(at) for at in atypes]
        )  # string to integers
        atypes = np.eye(max(gentype2num.values()) + 1)[atypes_int]

        if samples["xyz_rec"].shape[1] != prop["xyz_rec"].shape[0]:
            return LocalDataset._skip_getitem(info)

        sasa = []
        sasa_lig = np.array([0.5 for _ in range(xyz_ligs.shape[1])])  # neutral value
        sasa = np.concatenate([sasa_lig, sasa_rec])
        sasa = np.expand_dims(sasa, axis=1)
        bnds = np.concatenate([bnds_lig, bnds_rec])

        # Xtal properties
        if self.training:
            xtal_xyz_ligs = self.xtal_ligand_coordinate(samples)  # [N_lig_atom, 3]
            if not isinstance(xtal_xyz_ligs, np.ndarray) and xtal_xyz_ligs is None:
                return LocalDataset._skip_getitem(info)
            if xtal_xyz_ligs.shape[0] != xyz_ligs.shape[1]:
                return LocalDataset._skip_getitem(info)
            xtal_xyz_recs = prop["xyz_rec"]  # [N_rec_atom, 3]
            xtal_xyz_ligs = np.repeat(
                np.expand_dims(xtal_xyz_ligs, axis=0), repeats=self.subset_size, axis=0
            )  # [subset_size, N_lig_atom, 3]
            xtal_xyz_recs = np.repeat(
                np.expand_dims(xtal_xyz_recs, axis=0), repeats=self.subset_size, axis=0
            )  # [subset_size, N_rec_atom, 3]

        # Generate masks
        hbond_mask, polar_apolar_mask, apolar_apolar_mask, distance_map = (
            self.interaction_masks(
                atypes_int, islig, xyz_ligs, xyz_recs, charges_lig, charges_rec
            )
        )
        if self.training:
            (
                xtal_hbond_mask,
                xtal_polar_apolar_mask,
                xtal_apolar_apolar_mask,
                xtal_distance_map,
            ) = self.interaction_masks(
                atypes_int,
                islig,
                xtal_xyz_ligs,
                xtal_xyz_recs,
                charges_lig,
                charges_rec,
            )

        # orient around ligand-COM
        center_xyz = np.mean(xyz_ligs, axis=1)
        center_xyz = np.expand_dims(center_xyz, axis=1)
        xyz = xyz - center_xyz
        xyz_ligs = xyz_ligs - center_xyz
        center_xyz[:, :, :] = 0.0

        ball_xyzs = []
        if self.ballmode == "com":
            ball_xyzs = [center_xyz]
        elif self.ballmode == "all":
            ball_xyzs = [[a[None, :] for a in xyz_lig] for xyz_lig in xyz_ligs]

        # randomize coordinate
        if self.randomize > 1e-3:
            # randxyz = 2.0*self.randomize*(0.5 - np.random.rand(self.subset_size, xyz.shape[1],3)) # -0.2 ~ 0.2
            randxyz = self.randomize * np.random.randn(
                self.subset_size, xyz.shape[1], 3
            )
            xyz = xyz + randxyz

        resfeat = [islig, aas1hot, sasa]
        resfeat_extra = []  # from res-index

        if self.use_AF:
            # Dims: islig(1) + aas1hot(21) + atypes(65) + charges(1) -> 88
            input_features = [islig, aas1hot, atypes, charges]

            # AF features (dim=1) -> 89
            input_features = self.add_extra_features(
                input_features, pname, sname, prop, samples, self.training
            )
            if input_features is None:
                return LocalDataset._skip_getitem(info)

            G_atm_list, G_high_atm_list, idx_ord_list = [], [], []
            dist_idx_ord_list = []
            for idx in range(self.subset_size):
                G_atm, G_high_atm, idx_ord = self.make_atm_graphs(
                    xyz[idx],
                    ball_xyzs[idx],
                    input_features,
                    bnds,
                    xyz_ligs.shape[1],
                    atypes_int,
                    charges,
                    s_vdw,
                    w_vdw,
                )
                dist_idx_ord = self.get_nearest_protein_atom_index(
                    xyz[idx], ball_xyzs[idx], distance=self.distance_map_dist
                )
                G_atm_list.append(G_atm)
                G_high_atm_list.append(G_high_atm)
                idx_ord_list.append(idx_ord)
                dist_idx_ord_list.append(dist_idx_ord)
        else:
            G_atm_list, G_high_atm_list, idx_ord_list = [], [], []
            for idx in range(self.subset_size):
                G_atm, G_high_atm, idx_ord = self.make_atm_graphs(
                    xyz[idx],
                    ball_xyzs[idx],
                    [
                        islig,
                        aas1hot,
                        atypes,
                        charges,
                    ],  # dims: islig(1) + aas1hot(33) + atypes(65) + charges(1) -> 100
                    bnds,
                    xyz_ligs.shape[1],
                    atypes_int,
                    charges,
                    s_vdw,
                    w_vdw,
                )
                G_atm_list.append(G_atm)
                G_high_atm_list.append(G_high_atm)
                idx_ord_list.append(idx_ord)

        # Re-indexing masks
        hbond_mask_list, polar_apolar_mask_list, apolar_apolar_mask_list = [], [], []
        distance_map_list = []
        rec_idx_list = []
        if self.training:
            (
                xtal_hbond_mask_list,
                xtal_polar_apolar_mask_list,
                xtal_apolar_apolar_mask_list,
            ) = [], [], []
            xtal_distance_map_list = []

        N_lig_atoms = xyz_ligs.shape[1]
        for n, idx_ord in enumerate(
            idx_ord_list
        ):  # idx_ord: Atom's indexes in Atom graph
            rec_idx_ord = (np.array(idx_ord[N_lig_atoms:]) - N_lig_atoms).astype(int)
            dist_rec_idx_ord = (
                np.array(dist_idx_ord_list[n][N_lig_atoms:]) - N_lig_atoms
            ).astype(int)
            if len(dist_rec_idx_ord) == 0 or min(dist_rec_idx_ord) < 0:
                return LocalDataset._skip_getitem(info)
            if ("Graph" not in G_atm.__class__.__name__) or not (
                G_atm_list[n].number_of_nodes() == len(rec_idx_ord) + N_lig_atoms
            ):
                return LocalDataset._skip_getitem(info)

            rec_idx = np.array(
                [idx for idx, num in enumerate(rec_idx_ord) if num in dist_rec_idx_ord]
            )
            rec_idx_list.append(rec_idx)
            hbond_mask_list.append(hbond_mask[n][:, rec_idx_ord])
            polar_apolar_mask_list.append(polar_apolar_mask[n][:, rec_idx_ord])
            apolar_apolar_mask_list.append(apolar_apolar_mask[n][:, rec_idx_ord])
            distance_map_list.append(distance_map[n][:, dist_rec_idx_ord])
            if self.training:
                xtal_hbond_mask_list.append(xtal_hbond_mask[n][:, rec_idx_ord])
                xtal_polar_apolar_mask_list.append(
                    xtal_polar_apolar_mask[n][:, rec_idx_ord]
                )
                xtal_apolar_apolar_mask_list.append(
                    xtal_apolar_apolar_mask[n][:, rec_idx_ord]
                )
                xtal_distance_map_list.append(xtal_distance_map[n][:, dist_rec_idx_ord])

        info["islig"] = torch.tensor(islig).float()

        # Reorder by where atm idx in G_atm are
        rsds_ord_list = [r2a[idx_ord] for idx_ord in idx_ord_list]
        G_res_list, r2amap_list = [], []
        for idx in range(self.subset_size):
            G_res, r2amap = self.make_res_graph(
                xyz[idx],
                center_xyz[idx],
                resfeat,
                repsatm_idx,
                rsds_ord_list[idx],
                resfeat_extra,
            )
            G_res_list.append(G_res)
            r2amap_list.append(r2amap)

        # store which indices go to ligand atms
        ligidx_list, high_ligidx_list = [], []
        for idx in range(self.subset_size):
            ligidx = np.zeros((len(idx_ord_list[idx]), xyz_ligs.shape[1]))
            high_ligidx = np.zeros(
                (G_high_atm_list[idx].num_nodes(), xyz_ligs.shape[1])
            )
            for i in range(xyz_ligs.shape[1]):
                ligidx[i, i] = 1.0
                high_ligidx[i, i] = 1.0
            ligidx_list.append(torch.tensor(ligidx).float())
            high_ligidx_list.append(torch.tensor(high_ligidx).float())
        info["ligidx"] = ligidx_list
        info["high_ligidx"] = high_ligidx_list

        info["fnat"] = torch.tensor(fnat).float()
        info["lddt"] = torch.tensor(lddt).float()
        info["rosetta_e"] = torch.tensor(np.array(rosetta_e)).float()
        info["r2amap"] = [torch.tensor(r2amap).float() for r2amap in r2amap_list]
        info["r2a"] = torch.tensor(r2a1hot).float()
        info["repsatm_idx"] = torch.tensor(repsatm_idx).float()
        info["hbond_masks"] = [torch.tensor(mask).float() for mask in hbond_mask_list]
        info["polar_apolar_masks"] = [
            torch.tensor(mask).float() for mask in polar_apolar_mask_list
        ]
        info["apolar_apolar_masks"] = [
            torch.tensor(mask).float() for mask in apolar_apolar_mask_list
        ]
        info["distance_masks"] = [
            torch.tensor(mask).float() for mask in distance_map_list
        ]
        info["dist_rec_indices"] = [
            torch.tensor(rec_idx).long() for rec_idx in rec_idx_list
        ]
        if self.training:
            info["xtal_hbond_masks"] = [
                torch.tensor(mask).float() for mask in xtal_hbond_mask_list
            ]
            info["xtal_polar_apolar_masks"] = [
                torch.tensor(mask).float() for mask in xtal_polar_apolar_mask_list
            ]
            info["xtal_apolar_apolar_masks"] = [
                torch.tensor(mask).float() for mask in xtal_apolar_apolar_mask_list
            ]
            info["xtal_distance_masks"] = [
                torch.tensor(mask).float() for mask in xtal_distance_map_list
            ]

        return G_atm_list, G_res_list, G_high_atm_list, info

    def get_a_sample(self, pname, index):
        samples = np.load(self.datadir / (pname + ".lig.npz"), allow_pickle=True)
        pindices = list(range(len(samples["name"])))
        fnats = samples["fnat"]
        if self.sample_mode == "random":
            pindex = np.random.choice(
                pindices, size=self.subset_size, replace=False, p=self.upsample(fnats)
            )
        elif self.sample_mode == "serial":
            pindex = [index % len(pindices)]
        return samples, pindex

    def receptor_features(self, prop):
        charges_rec = prop["charge_rec"]
        atypes_rec = prop["atypes_rec"]
        aas_rec = prop["aas_rec"]
        repsatm_idx = prop[
            "repsatm_idx"
        ]  # representative atm idx for each residue (e.g. CA); receptor only
        r2a = (
            np.array(prop["residue_idx"], dtype=int) + 1
        )  # Add ligand as the first residue
        reschains = prop["reschains"]
        sasa_rec = 0
        if "sasa_rec" in prop:
            sasa_rec = prop["sasa_rec"]
        return charges_rec, atypes_rec, aas_rec, repsatm_idx, r2a, sasa_rec, reschains

    def per_ligand_features(self, samples, pindex):
        xyz_ligs = samples["xyz"][pindex].reshape(
            self.subset_size, -1, 3
        )  # shape: (subset_size, atom_num, 3)
        xyz_recs = samples["xyz_rec"][pindex].reshape(self.subset_size, -1, 3)
        lddt = samples["lddt"][pindex].reshape(
            self.subset_size, -1
        )  # shape: (subset_size, num_atom_ligand)
        fnat = samples["fnat"][pindex].reshape(
            self.subset_size, -1
        )  # shape: (subset_size, 1)

        atypes_lig = samples["atypes_lig"][0]
        bnds_lig = samples["bnds_lig"][0]
        charges_lig = samples["charge_lig"][0]
        aas_lig = samples["aas_lig"][0]

        if "repsatm_lig" in samples:
            repsatm_lig = samples["repsatm_lig"][0]
        else:
            repsatm_lig = 0
        return (
            xyz_ligs,
            xyz_recs,
            lddt,
            fnat,
            atypes_lig,
            bnds_lig,
            charges_lig,
            aas_lig,
            repsatm_lig,
        )

    def interaction_masks(
        self, atypes_int, islig, xyz_ligs, xyz_recs, charges_lig, charges_rec
    ):
        hbond_mask = self.generate_hbond_mask(
            atypes_int, islig, xyz_ligs, xyz_recs, self.subset_size, self.all_cross
        )  # [subset_size, N_all, N_all]
        polar_apolar_mask, apolar_apolar_mask = self.generate_polar_interaction_mask(
            charges_lig, charges_rec, xyz_ligs, xyz_recs, self.all_cross
        )  # [subset_size, N_all, N_all]
        if self.all_cross is False:
            distance_map = self.generate_distance_map(xyz_ligs, xyz_recs)
        else:
            xyz = np.concatenate([xyz_ligs, xyz_recs], axis=1)
            distance_map = self.generate_distance_map(xyz, xyz)

        return hbond_mask, polar_apolar_mask, apolar_apolar_mask, distance_map

    def make_res_graph(
        self, xyz, center_xyz, obt_fs, repsatm_idx, rsds_in_Gatm, extra_fs
    ):
        xyz_reps = xyz[repsatm_idx]
        xyz_reps = torch.tensor(xyz_reps - center_xyz).float()

        # Grabbing a < dist neighbor
        kd = cKDTree(xyz_reps)
        kd_ca = cKDTree(center_xyz)
        indices = kd_ca.query_ball_tree(kd, 1000.0)[
            0
        ]  # any huge number to cover entire protein
        reps_idx = [repsatm_idx[i] for i in indices]

        r2amap = np.zeros(len(rsds_in_Gatm), dtype=int)
        for i, rsd in enumerate(rsds_in_Gatm):
            if rsd in indices:
                r2amap[i] = indices.index(rsd)
            else:
                sys.exit("unexpected resno %d" % (rsd))
        r2amap = np.eye(max(indices) + 1)[r2amap]  # one_hot

        obt = []
        for f in obt_fs:
            if len(f) > 0:
                obt.append(f[reps_idx])

        for f in extra_fs:
            if len(f) > 0:
                obt.append(f[indices][:, None])  # should match the size

        obt = np.concatenate(obt, axis=-1)

        u, v = self.dist_fn_res(xyz_reps[None,])
        D = torch.sqrt(torch.sum((xyz_reps[v] - xyz_reps[u]) ** 2, axis=-1) + 1e-6)[
            ..., None
        ]
        D1hot = self.distance_feature(self.distance_feat, D, 1.0, 10.0)

        G_res = dgl.graph((u, v))
        G_res.ndata["0"] = torch.tensor(obt).float()
        G_res.ndata["x"] = xyz_reps[:, None, :]
        G_res.edata["rel_pos"] = xyz_reps[v] - xyz_reps[u]
        G_res.edata["0"] = D1hot

        return G_res, r2amap

    def make_atm_graphs(
        self, xyz, ball_xyzs, obt_fs, bnds, nlig, atypes_int, charges, s_vdw, w_vdw
    ):
        kd = cKDTree(xyz)
        indices, high_indices = [], []
        for ball_xyz in ball_xyzs:
            kd_ca = cKDTree(ball_xyz)
            indices += kd_ca.query_ball_tree(kd, self.ball_radius)[0]
            high_indices += kd_ca.query_ball_tree(kd, self.high_ball_radius)[0]
        indices = np.unique(indices)
        high_indices = np.unique(high_indices)

        # Make sure ligand atms are ALL INCLUDED
        idx_ord = [i for i in indices if i < nlig]
        idx_ord += [i for i in indices if i >= nlig]
        old_idx_map = {new_i: old_i for new_i, old_i in enumerate(idx_ord)}
        high_idx_ord = [i for i in high_indices if i < nlig]
        high_idx_ord += [i for i in high_indices if i >= nlig]

        # Concatenate all one-body-features
        obt, high_obt = [], []
        for f in obt_fs:
            if len(f) > 0:
                obt.append(f[idx_ord])
                high_obt.append(f[high_idx_ord])
        obt = np.concatenate(obt, axis=-1)
        high_obt = np.concatenate(high_obt, axis=-1)

        xyz_old = xyz
        bnds_old = bnds
        xyz = xyz[idx_ord]
        bnds = [bnd for bnd in bnds if (bnd[0] in idx_ord) and (bnd[1] in idx_ord)]
        bnds_bin = np.zeros((len(xyz), len(xyz)))

        high_xyz = xyz_old[high_idx_ord]
        high_bnds = [
            bnd for bnd in bnds if (bnd[0] in high_idx_ord) and (bnd[1] in high_idx_ord)
        ]
        high_bnds_bin = np.zeros((len(high_xyz), len(high_xyz)))

        for i, j in bnds:
            k, l = idx_ord.index(i), idx_ord.index(j)
            bnds_bin[k, l] = bnds_bin[l, k] = 1
        for i in range(len(xyz)):
            bnds_bin[i, i] = 1  # self

        for i, j in high_bnds:
            k, l = high_idx_ord.index(i), high_idx_ord.index(j)
            high_bnds_bin[k, l] = high_bnds_bin[l, k] = 1
        for i in range(len(high_xyz)):
            high_bnds_bin[i, i] = 1  # self

        # Concatenate coord & centralize xyz to ca.
        xyz = torch.tensor(xyz).float()
        high_xyz = torch.tensor(high_xyz).float()

        # for G_atm
        u, v = self.dist_fn_atm(xyz[None,])  # Num_edges
        high_u, high_v = self.dist_fn_high(high_xyz[None,])

        # Edge feature: distance
        w = torch.sqrt(torch.sum((xyz[v] - xyz[u]) ** 2, axis=-1) + 1e-6)[..., None]
        w1hot = self.distance_feature(self.distance_feat, w, 0.5, 5.0)
        bnds_bin = torch.tensor(
            bnds_bin[v, u]
        ).float()  # replace first bin (0.0~0.5 Ang) to bond info
        w1hot[:, 0] = bnds_bin  # Shape: (Num_edges, 2)

        high_w = torch.sqrt(
            torch.sum((high_xyz[high_v] - high_xyz[high_u]) ** 2, axis=-1) + 1e-6
        )[..., None]
        high_w1hot = self.distance_feature(self.distance_feat, high_w, 0.5, 5.0)
        high_bnds_bin = torch.tensor(high_bnds_bin[high_v, high_u]).float()
        high_w1hot[:, 0] = high_bnds_bin

        # Other node feature
        S_vdw = [s_vdw[i] for i in idx_ord]
        W_vdw = [w_vdw[i] for i in idx_ord]

        # Other edge features
        if self.extra_edgefeat:
            uv = np.array(torch.concatenate([u[None,], v[None,]]).T)
            edge_hbond = torch.tensor(
                self.is_hbond(uv, bnds_old, atypes_int, xyz_old, old_idx_map),
                dtype=torch.int,
            )
            edge_pol_apol = torch.tensor(
                self.is_polar_apolar(
                    uv, bnds_old, charges, atypes_int, xyz_old, old_idx_map
                ),
                dtype=torch.int,
            )
            w1hot = torch.concatenate(
                [w1hot, edge_hbond[:, None], edge_pol_apol], dim=-1
            )

        ## Construct graphs
        # G_atm: graph for all atms
        G_atm = dgl.graph((u, v))
        G_atm.ndata["0"] = torch.tensor(obt).float()
        G_atm.ndata["x"] = xyz[:, None, :]
        G_atm.ndata["vdw_sig"] = torch.tensor(S_vdw).float()
        G_atm.ndata["vdw_dep"] = torch.tensor(W_vdw).float()
        G_atm.edata["rel_pos"] = xyz[v] - xyz[u]
        G_atm.edata["0"] = w1hot
        # High resolution atom graph
        G_high_atm = dgl.graph((high_u, high_v))
        G_high_atm.ndata["0"] = torch.tensor(high_obt).float()
        G_high_atm.ndata["x"] = high_xyz[:, None, :]
        G_high_atm.edata["rel_pos"] = high_xyz[high_v] - high_xyz[high_u]
        G_high_atm.edata["0"] = high_w1hot

        return G_atm, G_high_atm, idx_ord

    def get_nearest_protein_atom_index(self, xyz, ball_xyzs, distance=5):
        kd = cKDTree(xyz)
        indices = []
        for ball_xyz in ball_xyzs:
            kd_ca = cKDTree(ball_xyz)
            indices += kd_ca.query_ball_tree(kd, distance)[0]
        indices = np.unique(indices)
        return indices

    def get_all_energy(self, pname: str):
        samples_ros = np.load(self.rosdata_dir / pname + ".npz", allow_pickle=True)
        return samples_ros

    def select_energy(self, samples_ros, sname):
        rosetta_e = []
        for i in range(len(sname)):
            name = str(sname[i][0] + ".pdb")
            index = np.where(samples_ros["pdb"] == name)[0]
            if len(index) == 0:
                return None
            rosetta_e.append(
                [
                    samples_ros["vdw"][index][0],
                    samples_ros["solvation"][index][0],
                    samples_ros["Coulomb"][index][0],
                    samples_ros["Hbond"][index][0],
                ]
            )
        return rosetta_e

    def add_extra_features(
        self, input_features, pname, sname, prop, samples, training: bool = False
    ):
        """
        Using AF feature as extra features.
        """
        af_feature_path = (
            Path(self.AF_plddt_path)
            if self.AF_plddt_path is not None
            else self.AF_plddt_path
        )

        # try:
        if training and ("native_dock" in sname or "near_native.native" in sname):
            af_feature = np.ones([samples["xyz_rec"].shape[1], 1])
        elif af_feature_path is None:
            af_feature = np.ones([samples["xyz_rec"].shape[1], 1])
        else:
            af_feature = np.load(af_feature_path.joinpath(f"{pname}_conf.npy"))
            af_feature = af_feature[prop["residue_idx"]]
        lig_feature = np.zeros(
            [samples["xyz"].shape[1], af_feature.shape[-1]]
        )  # samples['xyz'].shape[1]: ligand atom num
        af_feature = np.concatenate([lig_feature, af_feature], axis=0)

        input_features.append(af_feature)
        # except:
        #     return None

        return input_features

    def distance_feature(self, mode, d, binsize=0.5, maxd=5.0):
        if mode == "1hot":
            b = (d / binsize).long()
            nbin = int(maxd / binsize)
            b[b >= nbin] = nbin - 1
            d1hot = torch.eye(nbin)[b].float()
            feat = d1hot.squeeze()
        elif mode == "std":  # sigmoid
            d0 = 0.5 * (maxd - binsize)  # center
            m = 5.0 / d0  # slope
            feat = 1.0 / (1.0 + torch.exp(-m * (d - d0)))
            feat = feat.repeat(1, 2)  # hacky!! make as 2-dim
        return feat

    # Given a list of coordinates X, gets top-k neighbours based on eucledian distance
    def get_dist_neighbors(self, X, mode="topk", top_k=16, dcut=4.5, eps=1e-6):
        """Pairwise euclidean distances"""
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = torch.sqrt(torch.sum(dX**2, 3) + eps)

        if mode in ["topk", "mink"]:
            D_neighbors, E_idx = torch.topk(D, top_k + 1, dim=-1, largest=False)
            # exclude self-connection
            D_neighbor = D_neighbors[:, :, 1:]
            E_idx = E_idx[:, :, 1:]
            u = (
                torch.tensor(np.arange(E_idx.shape[1]))[:, None]
                .repeat(1, E_idx.shape[2])
                .reshape(-1)
            )
            v = E_idx[0,].reshape(-1)

            # append more than k that are within dcut
            if mode == "mink":
                nprv = len(u)
                mask = torch.where(torch.tril(D) < 1.0e-6, 100.0, 1.0)
                _, uD, vD = torch.where((mask * D) < dcut)
                uv = np.array(list(zip(u, v)) + list(zip(uD, vD)))
                uv = np.unique(uv, axis=0)
                u = [a for a, b in uv]
                v = [b for a, b in uv]
                # print("nedge:",dcut,nprv,len(uD),len(u))

        elif mode == "distT":
            mask = torch.where(torch.tril(D) < 1.0e-6, 100.0, 1.0)
            _, u, v = torch.where((mask * D) < dcut)

        elif mode == "dist":
            _, u, v = torch.where(D < dcut)

        return u, v

    def normalize_hbond_distance(self, x):
        return np.clip((1 / (x + 1e-9) ** 4) / (1 / 1.8**4), 0, 1)

    def generate_hbond_mask(
        self, atypes, islig, xyz_ligs, xyz_recs, subset_size, all_cross
    ):
        """
        Returns:
            hbond_mask: [subset_size, N_lig, N_rec]
        """
        polar_hydrogen = [15, 16]  # HO, NH
        h_acceptor = [
            20,
            24,
            31,
            32,
            34,
            35,
            36,
            38,
            39,
            40,
            44,
        ]  # Nad, Ngu1, Ohx, Oet, Oad, Oat(35), Ofu, OG2, OG3, OG31(40), SG2

        lig_mask = islig.squeeze().astype(bool)
        lig_atype = atypes[lig_mask]
        rec_atype = atypes[~lig_mask]

        hbond_row_temp = np.outer(
            np.isin(lig_atype, polar_hydrogen).astype(float),
            np.isin(rec_atype, h_acceptor).astype(float),
        )
        hbond_col_temp = np.outer(
            np.isin(lig_atype, h_acceptor).astype(float),
            np.isin(rec_atype, polar_hydrogen).astype(float),
        )
        hbond_mask = hbond_row_temp + hbond_col_temp
        hbond_mask = np.where(hbond_mask == 0.0, self.masking_coeff, hbond_mask)
        hbond_mask = np.repeat(
            np.expand_dims(hbond_mask, axis=0), repeats=subset_size, axis=0
        )

        distance_tensor = []
        for i in range(len(xyz_ligs)):
            distance_tensor.append(cdist(xyz_ligs[i], xyz_recs[i]))
        distance_tensor = np.stack(distance_tensor, axis=0)

        hbond_mask = hbond_mask * self.normalize_hbond_distance(distance_tensor)
        if self.scaling_all:
            hbond_mask *= 10
        return hbond_mask

    def generate_polar_interaction_mask(
        self, charges_lig, charges_rec, xyz_ligs, xyz_recs, all_cross
    ):
        """
        Each elements in masking tensor is caculated by the equation (-(q1 * q2) / d**2)

        Returns:
            polar_apolar_mask and apolar_apolar_mask: [subset_size, N_lig, N_rec]
        """
        polar_apolar_row = np.outer(
            (abs(charges_lig) < 0.15).astype(float),
            (abs(charges_rec) > 0.3).astype(float),
        )
        polar_apolar_col = np.outer(
            (abs(charges_lig) > 0.3).astype(float),
            (abs(charges_rec) < 0.15).astype(float),
        )
        polar_apolar_mask = polar_apolar_row + polar_apolar_col

        apolar_apolar_mask = np.outer(
            (abs(charges_lig) < 0.15).astype(float),
            (abs(charges_rec) < 0.15).astype(float),
        )

        # Get distance tensor
        distance_square_tensor = []
        for idx in range(len(xyz_ligs)):
            distance_square_tensor.append(cdist(xyz_ligs[idx], xyz_recs[idx]))
        distance_square_tensor = np.square(np.stack(distance_square_tensor, axis=0))

        apol_ones_lig = np.where(abs(charges_lig) < 0.15, 1.0, charges_lig)
        apol_ones_rec = np.where(abs(charges_rec) < 0.15, 1.0, charges_rec)
        coulomb_force = np.abs(
            np.divide(np.outer(apol_ones_lig, apol_ones_rec), distance_square_tensor)
        )

        if self.scaling_all:
            coulomb_force = coulomb_force * 100
        else:
            if self.scaling_apol_pol:
                coulomb_force = coulomb_force * 10

        coulomb_force_pol_apol = coulomb_force
        coulomb_force_apol_apol = coulomb_force

        polar_apolar_mask = np.where(
            polar_apolar_mask.astype(bool), coulomb_force_pol_apol, self.masking_coeff
        )
        apolar_apolar_mask = np.where(
            apolar_apolar_mask.astype(bool), coulomb_force_apol_apol, self.masking_coeff
        )

        return polar_apolar_mask, apolar_apolar_mask

    def generate_distance_map(
        self,
        xyz_ligs: np.ndarray,
        xyz_recs: np.ndarray,
        interaction_radius: Optional[int] = None,
    ) -> np.ndarray:
        distance_map_list = []
        for idx in range(len(xyz_ligs)):
            distance_map_list.append(cdist(xyz_ligs[idx], xyz_recs[idx]))
        distance_map_list = np.stack(distance_map_list, axis=0)
        if interaction_radius is not None:
            distance_map_list = np.where(distance_map_list < interaction_radius, 1, 0)
        return distance_map_list

    def xtal_ligand_coordinate(self, samples: np.ndarray) -> np.ndarray:
        idx = list(samples["name"]).index("near_native.native")  # native structure
        native_ligand_xyz = samples["xyz"][idx]
        return native_ligand_xyz

    def is_hbond(
        self,
        edges: np.ndarray,
        bnds: np.ndarray,
        atypes_int: list,
        xyz: np.ndarray,
        old_idx_map: dict,
    ) -> np.ndarray:
        edges = np.sort(edges, axis=1)
        src_indices = np.vectorize(old_idx_map.get)(edges[:, 0])
        dst_indices = np.vectorize(old_idx_map.get)(edges[:, 1])
        edges = np.concatenate([src_indices[:, None], dst_indices[:, None]], axis=1)

        # Filter out self-edges and covalant bond
        result = np.logical_and(
            src_indices != dst_indices,
            np.invert((edges[:, None] == bnds).all(-1).any(-1)),
        )
        polar_hydrogen = [15, 16]  # HO, NH
        h_acceptor = [
            20,
            24,
            31,
            32,
            34,
            35,
            36,
            38,
            39,
            40,
            44,
        ]  # Nad, Ngu1, Ohx, Oet, Oad, Oat(35), Ofu, OG2, OG3, OG31(40), SG2
        src_atypes, dst_atypes = atypes_int[src_indices], atypes_int[dst_indices]
        src_xyz, dst_xyz = xyz[src_indices], xyz[dst_indices]

        # Atom atypes
        is_donor_acceptor = np.logical_and(
            np.isin(src_atypes, polar_hydrogen), np.isin(dst_atypes, h_acceptor)
        )
        is_acceptor_donor = np.logical_and(
            np.isin(src_atypes, h_acceptor), np.isin(dst_atypes, polar_hydrogen)
        )
        result *= np.logical_or(is_donor_acceptor, is_acceptor_donor)

        # Distance
        distances = np.linalg.norm(src_xyz - dst_xyz, axis=1)
        result *= distances <= 2.5

        # Acceptor and Donor conditions
        angle_mask = np.ones_like(result)
        for i in range(len(result)):
            if result[i] is False:
                angle_mask[i] = False
                continue
            if is_donor_acceptor[i]:
                dst = dst_indices[i]
                if (bnds[:, 0] == dst).sum():
                    neigh_index = bnds[bnds[:, 0] == dst, 1]
                    neigh_xyz = xyz[neigh_index].mean(axis=0)
                else:
                    neigh_index = bnds[bnds[:, 1] == dst, 0]
                    neigh_xyz = xyz[neigh_index].mean(axis=0)
                v1 = src_xyz[i] - dst_xyz[i]
                v2 = neigh_xyz - dst_xyz[i]
                cos_angle = cosine_angle(v1, v2)
                if not (cos_angle < np.radians(180) and cos_angle > np.radians(120)):
                    angle_mask[i] = False
                src = src_indices[i]
                if (bnds[:, 0] == src).sum():
                    neigh_index = bnds[bnds[:, 0] == src, 1]
                    neigh_xyz = xyz[neigh_index].mean(axis=0)
                else:
                    neigh_index = bnds[bnds[:, 1] == src, 0]
                    neigh_xyz = xyz[neigh_index].mean(axis=0)
                v1 = dst_xyz[i] - src_xyz[i]
                v2 = neigh_xyz - src_xyz[i]
                cos_angle = cosine_angle(v1, v2)
                if not (cos_angle < np.radians(180) and cos_angle > np.radians(150)):
                    angle_mask[i] = False
            elif is_acceptor_donor[i]:
                src = src_indices[i]
                if (bnds[:, 0] == src).sum():
                    neigh_index = bnds[bnds[:, 0] == src, 1]
                    neigh_xyz = xyz[neigh_index].mean(axis=0)
                else:
                    neigh_index = bnds[bnds[:, 1] == src, 0]
                    neigh_xyz = xyz[neigh_index].mean(axis=0)
                v1 = dst_xyz[i] - src_xyz[i]
                v2 = neigh_xyz - src_xyz[i]
                cos_angle = cosine_angle(v1, v2)
                if not (cos_angle < np.radians(180) and cos_angle > np.radians(120)):
                    angle_mask[i] = False
                dst = dst_indices[i]
                if (bnds[:, 0] == dst).sum():
                    neigh_index = bnds[bnds[:, 0] == dst, 1]
                    neigh_xyz = xyz[neigh_index].mean(axis=0)
                else:
                    neigh_index = bnds[bnds[:, 1] == dst, 0]
                    neigh_xyz = xyz[neigh_index].mean(axis=0)
                v1 = src_xyz[i] - dst_xyz[i]
                v2 = neigh_xyz - dst_xyz[i]
                cos_angle = cosine_angle(v1, v2)
                if not (cos_angle < np.radians(180) and cos_angle > np.radians(150)):
                    angle_mask[i] = False
        result *= angle_mask
        return result

    def is_polar_apolar(
        self,
        edge: np.ndarray,
        bnds: np.ndarray,
        charges: np.ndarray,
        atypes_int: list,
        xyz: np.ndarray,
        old_idx_map: dict,
    ) -> Tuple[bool, bool]:
        """
        Returns:
            is_polar_apolar, is_apolar_apolar
        """

        def set_dist_cut(sa, da):
            hydrogens = [15, 16, 17]
            if sa in hydrogens and da in hydrogens:
                dist_pol, dist_apol = 4.0, 4.0
            elif sa in hydrogens or da in hydrogens:
                dist_pol, dist_apol = 4.5, 4.5
            else:
                dist_pol, dist_apol = 5.0, 5.0
            return dist_pol, dist_apol

        edge = np.sort(edge, axis=1)
        src = np.vectorize(old_idx_map.get)(edge[:, 0])
        dst = np.vectorize(old_idx_map.get)(edge[:, 1])
        src_atype, dst_atype = atypes_int[src], atypes_int[dst]
        dist_cut = np.array(list(map(set_dist_cut, src_atype, dst_atype)))
        dist_pol, dist_apol = dist_cut[:, 0], dist_cut[:, 1]
        result = src != dst
        result *= np.invert((edge[:, None] == bnds).all(-1).any(-1))
        result = np.concatenate([result[:, None], result[:, None]], axis=1)
        src_charge, dst_charge = charges[src], charges[dst]
        src_xyz, dst_xyz = xyz[src], xyz[dst]
        # Apolar-apolar
        cond = ((abs(src_charge) < 0.15) & (abs(dst_charge) < 0.15)).squeeze()
        result[:, 1] = result[:, 1] * cond
        cond = np.linalg.norm(src_xyz - dst_xyz, axis=1) < dist_apol
        result[:, 1] = result[:, 1] * cond
        # Polar-apolar
        cond = (
            (abs(src_charge) > 0.3) & (abs(dst_charge) < 0.15)
            | (abs(src_charge) < 0.15) & (abs(dst_charge) > 0.3)
        ).squeeze()
        result[:, 0] = result[:, 0] * cond
        cond = np.linalg.norm(src_xyz - dst_xyz, axis=1) < dist_pol
        result[:, 0] = result[:, 0] * cond
        return result

    @property
    def vdw_radius_depth(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        with open(self.atom_property) as file:
            W, S = {}, {}
            for line in file:
                if (
                    not line.strip()
                    or line.strip().startswith("#")
                    or line.strip().startswith("NAME")
                ):
                    continue
                fields = line.split()
                atom_name = fields[0]
                if fields[3] != "":
                    w = float(fields[3])
                if fields[2] != "":
                    s = float(fields[2])
                W[atom_name] = w
                S[atom_name] = s
        return W, S

    @staticmethod
    def _skip_getitem(info):
        return False, False, False, info


class LocalDatasetFineTune(LocalDataset):
    def __getitem__(self, index):
        info = {}
        info["sname"] = "none"
        if self.sample_mode == "serial":
            ip = int(index / self.nsamples_per_p)
            pname = self.proteins[ip]
        else:
            pname = self.proteins[index]
        info["pname"] = pname

        fname = self.datadir / (pname + ".lig.npz")
        if not os.path.exists(fname):
            return self._skip_getitem(info)
        try:
            if self.rosdata_dir is not None:
                samples_ros = self.get_all_energy(pname)
            samples, native_samples, dock_class, pindex = self.get_a_sample(
                pname, index
            )
            prop = np.load(self.datadir / (pname + ".prop.npz"), allow_pickle=True)
            prop = prop[dock_class].tolist()
        except Exception as e:
            print("raise error:", e)
            return LocalDataset._skip_getitem(info)

        sname = samples["name"][pindex].reshape(self.subset_size, -1)
        info["pindex"] = pindex
        info["sname"] = sname

        # Receptor features (prop)
        charges_rec, atypes_rec, aas_rec, repsatm_idx, r2a, sasa_rec, reschain = (
            self.receptor_features(prop)
        )

        # Ligand features (lig)
        (
            xyz_ligs,
            xyz_recs,
            lddt,
            fnat,
            atypes_lig,
            bnds_lig,
            charges_lig,
            aas_lig,
            repsatm_lig,
        ) = self.per_ligand_features(samples, pindex)
        if len(bnds_lig) == 0:
            return self._skip_getitem(info)

        # Rosetta_energy
        if self.rosdata_dir is not None:
            rosetta_e = self.select_energy(samples_ros, sname)
            if rosetta_e is None:
                return self._skip_getitem(info)
        else:
            rosetta_e = [[0.0, 0.0, 0.0, 0.0] for _ in range(len(sname))]

        aas = np.concatenate([aas_lig, aas_rec]).astype(int)
        aas1hot = np.eye(N_AATYPE)[aas]

        # Representative atoms
        r2a = np.concatenate([np.array([0 for _ in range(xyz_ligs.shape[1])]), r2a])
        r2a1hot = np.eye(max(r2a) + 1)[r2a]
        repsatm_idx = np.concatenate(
            [
                np.array(repsatm_lig),
                np.array(repsatm_idx, dtype=int) + xyz_ligs.shape[1],
            ]
        )  # shape: (subset_size, num_atom_rep)

        # Bond properties
        bnds_rec = prop["bnds_rec"] + xyz_ligs.shape[1]  # shift index;

        # Concatenate receptor & ligand: ligand comes first
        charges = np.expand_dims(np.concatenate([charges_lig, charges_rec]), axis=1)
        if self.normalize_q:
            charges = 1.0 / (1.0 + np.exp(-2.0 * charges))

        islig = np.array(
            [1 for _ in range(xyz_ligs.shape[1])]
            + [0 for _ in range(xyz_recs.shape[1])]
        )
        islig = np.expand_dims(islig, axis=1)

        xyz = np.concatenate([xyz_ligs, xyz_recs], axis=1)
        atypes = np.concatenate([atypes_lig, atypes_rec])
        if "Null" in atypes:
            print("There is 'Null' atom types in the PDB file. Skip this.")
            return self._skip_getitem(info)
        w_vdw = torch.tensor([self.W[atype] for atype in list(atypes)])  # vdw depth
        s_vdw = torch.tensor([self.S[atype] for atype in list(atypes)])  # vdw radius

        atypes_int = np.array(
            [find_gentype2num(at) for at in atypes]
        )  # string to integers
        atypes = np.eye(max(gentype2num.values()) + 1)[atypes_int]

        if samples["xyz_rec"].shape[1] != prop["xyz_rec"].shape[0]:
            return self._skip_getitem(info)

        sasa = []
        sasa_lig = np.array([0.5 for _ in range(xyz_ligs.shape[1])])  # neutral value
        sasa = np.concatenate([sasa_lig, sasa_rec])
        sasa = np.expand_dims(sasa, axis=1)
        bnds = np.concatenate([bnds_lig, bnds_rec])

        # Xtal properties
        if self.training:
            if "near_native.native" not in list(native_samples["name"]):
                return self._skip_getitem(info)
            xtal_xyz_ligs = self.xtal_ligand_coordinate(
                native_samples
            )  # [N_lig_atom, 3]
            if not isinstance(xtal_xyz_ligs, np.ndarray) and xtal_xyz_ligs is None:
                return self._skip_getitem(info)
            if xtal_xyz_ligs.shape[0] != xyz_ligs.shape[1]:
                return self._skip_getitem(info)
            xtal_xyz_recs = prop["xyz_rec"]  # [N_rec_atom, 3]
            xtal_xyz_ligs = np.repeat(
                np.expand_dims(xtal_xyz_ligs, axis=0), repeats=self.subset_size, axis=0
            )  # [subset_size, N_lig_atom, 3]
            xtal_xyz_recs = np.repeat(
                np.expand_dims(xtal_xyz_recs, axis=0), repeats=self.subset_size, axis=0
            )  # [subset_size, N_rec_atom, 3]

        # Generate masks
        hbond_mask, polar_apolar_mask, apolar_apolar_mask, distance_map = (
            self.interaction_masks(
                atypes_int, islig, xyz_ligs, xyz_recs, charges_lig, charges_rec
            )
        )
        if self.training:
            (
                xtal_hbond_mask,
                xtal_polar_apolar_mask,
                xtal_apolar_apolar_mask,
                xtal_distance_map,
            ) = self.interaction_masks(
                atypes_int,
                islig,
                xtal_xyz_ligs,
                xtal_xyz_recs,
                charges_lig,
                charges_rec,
            )

        # orient around ligand-COM
        center_xyz = np.mean(xyz_ligs, axis=1)
        center_xyz = np.expand_dims(center_xyz, axis=1)
        xyz = xyz - center_xyz
        xyz_ligs = xyz_ligs - center_xyz
        center_xyz[:, :, :] = 0.0

        ball_xyzs = []
        if self.ballmode == "com":
            ball_xyzs = [center_xyz]
        elif self.ballmode == "all":
            ball_xyzs = [[a[None, :] for a in xyz_lig] for xyz_lig in xyz_ligs]

        # randomize coordinate
        if self.randomize > 1e-3:
            # randxyz = 2.0*self.randomize*(0.5 - np.random.rand(self.subset_size, xyz.shape[1],3)) # -0.2 ~ 0.2
            randxyz = self.randomize * np.random.randn(
                self.subset_size, xyz.shape[1], 3
            )
            xyz = xyz + randxyz

        resfeat = [islig, aas1hot, sasa]
        resfeat_extra = []  # from res-index

        if self.use_AF:
            # Dims: islig(1) + aas1hot(21) + atypes(65) + charges(1) -> 88
            input_features = [islig, aas1hot, atypes, charges]

            # AF features (dim=1) -> 89
            input_features = self.add_extra_features(
                input_features, pname, sname, prop, samples, self.training
            )
            if input_features is None:
                return LocalDataset._skip_getitem(info)

            G_atm_list, G_high_atm_list, idx_ord_list = [], [], []
            dist_idx_ord_list = []
            for idx in range(self.subset_size):
                G_atm, G_high_atm, idx_ord = self.make_atm_graphs(
                    xyz[idx],
                    ball_xyzs[idx],
                    input_features,
                    bnds,
                    xyz_ligs.shape[1],
                    atypes_int,
                    charges,
                    s_vdw,
                    w_vdw,
                )
                dist_idx_ord = self.get_nearest_protein_atom_index(
                    xyz[idx], ball_xyzs[idx], distance=self.distance_map_dist
                )
                G_atm_list.append(G_atm)
                G_high_atm_list.append(G_high_atm)
                idx_ord_list.append(idx_ord)
                dist_idx_ord_list.append(dist_idx_ord)
        else:
            G_atm_list, G_high_atm_list, idx_ord_list = [], [], []
            for idx in range(self.subset_size):
                G_atm, G_high_atm, idx_ord = self.make_atm_graphs(
                    xyz[idx],
                    ball_xyzs[idx],
                    [
                        islig,
                        aas1hot,
                        atypes,
                        charges,
                    ],  # dims: islig(1) + aas1hot(33) + atypes(65) + charges(1) -> 100
                    bnds,
                    xyz_ligs.shape[1],
                    atypes_int,
                    charges,
                    s_vdw,
                    w_vdw,
                )
                G_atm_list.append(G_atm)
                G_high_atm_list.append(G_high_atm)
                idx_ord_list.append(idx_ord)

        # Re-indexing masks
        hbond_mask_list, polar_apolar_mask_list, apolar_apolar_mask_list = [], [], []
        distance_map_list = []
        rec_idx_list = []
        if self.training:
            (
                xtal_hbond_mask_list,
                xtal_polar_apolar_mask_list,
                xtal_apolar_apolar_mask_list,
            ) = [], [], []
            xtal_distance_map_list = []

        N_lig_atoms = xyz_ligs.shape[1]
        for n, idx_ord in enumerate(
            idx_ord_list
        ):  # idx_ord: Atom's indexes in Atom graph
            rec_idx_ord = (np.array(idx_ord[N_lig_atoms:]) - N_lig_atoms).astype(int)
            dist_rec_idx_ord = (
                np.array(dist_idx_ord_list[n][N_lig_atoms:]) - N_lig_atoms
            ).astype(int)
            if len(dist_rec_idx_ord) == 0 or min(dist_rec_idx_ord) < 0:
                return LocalDataset._skip_getitem(info)
            if ("Graph" not in G_atm.__class__.__name__) or not (
                G_atm_list[n].number_of_nodes() == len(rec_idx_ord) + N_lig_atoms
            ):
                return LocalDataset._skip_getitem(info)

            rec_idx = np.array(
                [idx for idx, num in enumerate(rec_idx_ord) if num in dist_rec_idx_ord]
            )
            rec_idx_list.append(rec_idx)
            hbond_mask_list.append(hbond_mask[n][:, rec_idx_ord])
            polar_apolar_mask_list.append(polar_apolar_mask[n][:, rec_idx_ord])
            apolar_apolar_mask_list.append(apolar_apolar_mask[n][:, rec_idx_ord])
            distance_map_list.append(distance_map[n][:, dist_rec_idx_ord])
            if self.training:
                xtal_hbond_mask_list.append(xtal_hbond_mask[n][:, rec_idx_ord])
                xtal_polar_apolar_mask_list.append(
                    xtal_polar_apolar_mask[n][:, rec_idx_ord]
                )
                xtal_apolar_apolar_mask_list.append(
                    xtal_apolar_apolar_mask[n][:, rec_idx_ord]
                )
                xtal_distance_map_list.append(xtal_distance_map[n][:, dist_rec_idx_ord])

        info["islig"] = torch.tensor(islig).float()

        # Reorder by where atm idx in G_atm are
        rsds_ord_list = [r2a[idx_ord] for idx_ord in idx_ord_list]
        G_res_list, r2amap_list = [], []
        for idx in range(self.subset_size):
            G_res, r2amap = self.make_res_graph(
                xyz[idx],
                center_xyz[idx],
                resfeat,
                repsatm_idx,
                rsds_ord_list[idx],
                resfeat_extra,
            )
            G_res_list.append(G_res)
            r2amap_list.append(r2amap)

        # store which indices go to ligand atms
        ligidx_list, high_ligidx_list = [], []
        for idx in range(self.subset_size):
            ligidx = np.zeros((len(idx_ord_list[idx]), xyz_ligs.shape[1]))
            high_ligidx = np.zeros(
                (G_high_atm_list[idx].num_nodes(), xyz_ligs.shape[1])
            )
            for i in range(xyz_ligs.shape[1]):
                ligidx[i, i] = 1.0
                high_ligidx[i, i] = 1.0
            ligidx_list.append(torch.tensor(ligidx).float())
            high_ligidx_list.append(torch.tensor(high_ligidx).float())
        info["ligidx"] = ligidx_list
        info["high_ligidx"] = high_ligidx_list

        info["fnat"] = torch.tensor(fnat).float()
        info["lddt"] = torch.tensor(lddt).float()
        info["rosetta_e"] = torch.tensor(np.array(rosetta_e)).float()
        info["r2amap"] = [torch.tensor(r2amap).float() for r2amap in r2amap_list]
        info["r2a"] = torch.tensor(r2a1hot).float()
        info["repsatm_idx"] = torch.tensor(repsatm_idx).float()
        info["hbond_masks"] = [torch.tensor(mask).float() for mask in hbond_mask_list]
        info["polar_apolar_masks"] = [
            torch.tensor(mask).float() for mask in polar_apolar_mask_list
        ]
        info["apolar_apolar_masks"] = [
            torch.tensor(mask).float() for mask in apolar_apolar_mask_list
        ]
        info["distance_masks"] = [
            torch.tensor(mask).float() for mask in distance_map_list
        ]
        info["dist_rec_indices"] = [
            torch.tensor(rec_idx).long() for rec_idx in rec_idx_list
        ]
        if self.training:
            info["xtal_hbond_masks"] = [
                torch.tensor(mask).float() for mask in xtal_hbond_mask_list
            ]
            info["xtal_polar_apolar_masks"] = [
                torch.tensor(mask).float() for mask in xtal_polar_apolar_mask_list
            ]
            info["xtal_apolar_apolar_masks"] = [
                torch.tensor(mask).float() for mask in xtal_apolar_apolar_mask_list
            ]
            info["xtal_distance_masks"] = [
                torch.tensor(mask).float() for mask in xtal_distance_map_list
            ]

        return G_atm_list, G_res_list, G_high_atm_list, info

    def get_a_sample(self, pname, index):
        samples = np.load(self.datadir / (pname + ".lig.npz"), allow_pickle=True)
        # Set docking class in ['generated_ligands', 'model_dock', 'native_dock']
        dock_classes = list(samples)

        def prob_map(x):
            if "generated" in x or "model" in x:
                p = 1.0
            else:
                p = 0.1
            return p

        p = np.array(list(map(prob_map, dock_classes)))
        p /= sum(p)
        dock_class = np.random.choice(dock_classes, p=p)
        native_samples = samples["native_dock"].tolist()
        samples = samples[dock_class].tolist()

        pindices = list(range(len(samples["name"])))
        fnats = samples["fnat"]
        if self.sample_mode == "random":
            pindex = np.random.choice(
                pindices, size=self.subset_size, replace=False, p=self.upsample(fnats)
            )
        elif self.sample_mode == "serial":
            pindex = [index % len(pindices)]
        return samples, native_samples, dock_class, pindex

    def add_extra_features(
        self, input_features, pname, sname, prop, samples, training: bool = False
    ):
        af_feature_path = (
            Path(self.AF_plddt_path)
            if self.AF_plddt_path is not None
            else self.AF_plddt_path
        )
        sname = sname[0, 0]

        if "native_dock" in sname or "near_native.native" in sname:
            dir_name = "native_dock"
        elif "model_dock_corina" in sname:
            dir_name = "generated_ligands"
        else:
            dir_name = "model_dock"

        if training and ("native_dock" in sname or "near_native.native" in sname):
            af_feature = np.ones([samples["xyz_rec"].shape[1], 1])
        elif af_feature_path is None:
            af_feature = np.ones([samples["xyz_rec"].shape[1], 1])
        else:
            af_feature = np.load(
                af_feature_path.joinpath(dir_name, f"{pname}_conf.npy")
            )
            af_feature = af_feature[prop["residue_idx"]]
        lig_feature = np.zeros(
            [samples["xyz"].shape[1], af_feature.shape[-1]]
        )  # samples['xyz'].shape[1]: ligand atom num
        af_feature = np.concatenate([lig_feature, af_feature], axis=0)

        input_features.append(af_feature)

        return input_features

    def xtal_ligand_coordinate(self, samples: np.ndarray) -> np.ndarray:
        idx = list(samples["name"]).index("near_native.native")  # native structure
        native_ligand_xyz = samples["xyz"][idx]
        return native_ligand_xyz


def sample_uniform(fnats: np.ndarray):
    return np.array([1.0 for _ in fnats]) / len(fnats)


def collate(samples):
    graphs_atm, graphs_res, graphs_high, info = samples[0]

    binfo = {
        "ligidx": [],
        "r2a": [],
        "fnat": [],
        "lddt": [],
        "pname": [],
        "sname": [],
        "nligatms": [],
        "hbond_masks": [],
        "polar_apolar_masks": [],
        "apolar_apolar_masks": [],
        "dist_rec_indices": [],
        "rosetta_e": [],
    }
    try:
        bgraph_atm = dgl.batch(graphs_atm)
        bgraph_res = dgl.batch(graphs_res)
        bgraph_high = dgl.batch(graphs_high)

        asum, bsum, lsum = 0, 0, 0
        # Reformat r2a, ligidx
        binfo["r2a"] = torch.zeros(
            (bgraph_atm.number_of_nodes(), bgraph_res.number_of_nodes())
        )

        nligsum = sum([s.shape[1] for s in info["ligidx"]])
        binfo["ligidx"] = torch.zeros((bgraph_atm.number_of_nodes(), nligsum))
        binfo["high_ligidx"] = torch.zeros((bgraph_high.number_of_nodes(), nligsum))
        binfo["sname"] = [s[0] for s in info["sname"]]
        binfo["pname"] = [info["pname"]] * len(info["sname"])
        binfo["rosetta_e"] = info["rosetta_e"]

        for idx, (a, b) in enumerate(
            zip(bgraph_atm.batch_num_nodes(), bgraph_res.batch_num_nodes())
        ):
            l = info["ligidx"][idx].shape[1]
            binfo["ligidx"][asum : asum + a, lsum : lsum + l] = info["ligidx"][idx]

            binfo["r2a"][asum : asum + a, bsum : bsum + b] = info["r2amap"][idx]
            binfo["fnat"].append(info["fnat"][idx])
            binfo["lddt"].append(info["lddt"][idx])
            binfo["nligatms"].append(l)

            binfo["hbond_masks"].append(info["hbond_masks"][idx])
            binfo["polar_apolar_masks"].append(info["polar_apolar_masks"][idx])
            binfo["apolar_apolar_masks"].append(info["apolar_apolar_masks"][idx])
            binfo["dist_rec_indices"].append(info["dist_rec_indices"][idx])

            asum += a
            bsum += b
            lsum += l

        csum, lsum = 0, 0
        for idx, c in enumerate(bgraph_high.batch_num_nodes()):
            l = info["high_ligidx"][idx].shape[1]
            binfo["high_ligidx"][csum : csum + c, lsum : lsum + l] = info[
                "high_ligidx"
            ][idx]
            csum += c
            lsum += l

        binfo["fnat"] = torch.tensor(binfo["fnat"])
        binfo["lddt"] = torch.cat(binfo["lddt"], axis=0)
        binfo["nligatms"] = torch.tensor(binfo["nligatms"])

    except:
        bgraph_atm, bgraph_high, bgraph_res = False, False, False

    return bgraph_atm, bgraph_res, bgraph_high, binfo


def collate_energy(samples):
    graphs_atm, graphs_res, info = samples[0]

    binfo = {
        "ligidx": [],
        "r2a": [],
        "fnat": [],
        "pname": [],
        "sname": [],
        "nligatms": [],
        "rosetta_e": [],
    }

    try:
        bgraph_atm = dgl.batch(graphs_atm)
        bgraph_res = dgl.batch(graphs_res)

        asum, bsum, lsum = 0, 0, 0
        # Reformat r2a, ligidx
        binfo["r2a"] = torch.zeros(
            (bgraph_atm.number_of_nodes(), bgraph_res.number_of_nodes())
        )

        nligsum = sum([s.shape[1] for s in info["ligidx"]])
        binfo["ligidx"] = torch.zeros((bgraph_atm.number_of_nodes(), nligsum))
        binfo["sname"] = [s[0] for s in info["sname"]]
        binfo["pname"] = [info["pname"]] * len(info["sname"])
        binfo["rosetta_e"] = info["rosetta_e"]

        for idx, (a, b) in enumerate(
            zip(bgraph_atm.batch_num_nodes(), bgraph_res.batch_num_nodes())
        ):
            l = info["ligidx"][idx].shape[1]
            binfo["ligidx"][asum : asum + a, lsum : lsum + l] = info["ligidx"][idx]

            binfo["r2a"][asum : asum + a, bsum : bsum + b] = info["r2amap"][idx]
            binfo["fnat"].append(info["fnat"][idx])
            binfo["nligatms"].append(l)

            asum += a
            bsum += b
            lsum += l

        binfo["fnat"] = torch.tensor(binfo["fnat"])
        binfo["nligatms"] = torch.tensor(binfo["nligatms"])

    except:
        bgraph_atm, bgraph_res = False, False

    return bgraph_atm, bgraph_res, binfo


def collate_training(samples):
    graphs_atm, graphs_res, graphs_high, info = samples[0]
    if isinstance(graphs_atm, bool) and graphs_atm is False:
        return False, False, False, info

    binfo = {
        "ligidx": [],
        "r2a": [],
        "fnat": [],
        "lddt": [],
        "pname": [],
        "sname": [],
        "nligatms": [],
        "hbond_masks": [],
        "polar_apolar_masks": [],
        "apolar_apolar_masks": [],
        "distance_masks": [],
        "xtal_hbond_masks": [],
        "xtal_polar_apolar_masks": [],
        "xtal_apolar_apolar_masks": [],
        "xtal_distance_masks": [],
        "dist_rec_indices": [],
    }
    try:
        bgraph_atm = dgl.batch(graphs_atm)
        bgraph_res = dgl.batch(graphs_res)
        bgraph_high = dgl.batch(graphs_high)

        asum, bsum, lsum = 0, 0, 0
        # Reformat r2a, ligidx
        binfo["r2a"] = torch.zeros(
            (bgraph_atm.number_of_nodes(), bgraph_res.number_of_nodes())
        )

        nligsum = sum([s.shape[1] for s in info["ligidx"]])
        binfo["ligidx"] = torch.zeros((bgraph_atm.number_of_nodes(), nligsum))
        binfo["high_ligidx"] = torch.zeros((bgraph_high.number_of_nodes(), nligsum))
        binfo["sname"] = [s[0] for s in info["sname"]]
        binfo["pname"] = [info["pname"]] * len(info["sname"])

        for idx, (a, b) in enumerate(
            zip(bgraph_atm.batch_num_nodes(), bgraph_res.batch_num_nodes())
        ):
            l = info["ligidx"][idx].shape[1]
            binfo["ligidx"][asum : asum + a, lsum : lsum + l] = info["ligidx"][idx]

            binfo["r2a"][asum : asum + a, bsum : bsum + b] = info["r2amap"][idx]
            binfo["fnat"].append(info["fnat"][idx])
            binfo["lddt"].append(info["lddt"][idx])
            binfo["nligatms"].append(l)

            binfo["hbond_masks"].append(info["hbond_masks"][idx])
            binfo["polar_apolar_masks"].append(info["polar_apolar_masks"][idx])
            binfo["apolar_apolar_masks"].append(info["apolar_apolar_masks"][idx])
            binfo["distance_masks"].append(info["distance_masks"][idx])

            binfo["xtal_hbond_masks"].append(info["xtal_hbond_masks"][idx])
            binfo["xtal_polar_apolar_masks"].append(
                info["xtal_polar_apolar_masks"][idx]
            )
            binfo["xtal_apolar_apolar_masks"].append(
                info["xtal_apolar_apolar_masks"][idx]
            )
            binfo["xtal_distance_masks"].append(info["xtal_distance_masks"][idx])

            binfo["dist_rec_indices"].append(info["dist_rec_indices"][idx])

            asum += a
            bsum += b
            lsum += l

        csum, lsum = 0, 0
        for idx, c in enumerate(bgraph_high.batch_num_nodes()):
            l = info["high_ligidx"][idx].shape[1]
            binfo["high_ligidx"][csum : csum + c, lsum : lsum + l] = info[
                "high_ligidx"
            ][idx]
            csum += c
            lsum += l

        binfo["fnat"] = torch.tensor(binfo["fnat"])
        binfo["lddt"] = torch.cat(binfo["lddt"], axis=0)
        binfo["nligatms"] = torch.tensor(binfo["nligatms"])

    except:
        bgraph_atm, bgraph_high, bgraph_res = False, False, False

    return bgraph_atm, bgraph_res, bgraph_high, binfo


class DfAFSampler(Sampler):
    def __init__(self, weights, replacement=True):
        weights = np.array(weights) / np.sum(weights)
        self.weights = weights
        self.num_samples = len(weights)
        self.replacement = replacement

    def __iter__(self):
        retval = np.random.choice(
            len(self.weights),
            self.num_samples,
            replace=self.replacement,
            p=self.weights,
        )
        return iter(retval.tolist())

    def __len__(self):
        return self.num_samples


def load_dataset(
    set_params: EasyDict,
    split_path: Path,
    generator_params: EasyDict,
    setsuffix: str = "Clean_MT2",
    gen_fine_tune: bool = False,
):
    if gen_fine_tune:
        dataset_cls = LocalDatasetFineTune
    else:
        dataset_cls = LocalDataset

    train_set = dataset_cls(
        np.load(split_path.joinpath(f"train{setsuffix}.npy")),
        training=True,
        **set_params,
    )

    valid_params = deepcopy(set_params)
    valid_params.randomize = 0.0
    val_set = dataset_cls(
        np.load(split_path.joinpath(f"valid{setsuffix}.npy")),
        training=True,
        **valid_params,
    )

    train_generator = data.DataLoader(
        train_set,
        worker_init_fn=lambda _: np.random.seed(),
        shuffle=False,
        sampler=DfAFSampler(
            np.load(split_path.joinpath(f"train_weight{setsuffix}.npy"))
        ),
        **generator_params,
    )

    valid_generator = data.DataLoader(
        val_set,
        worker_init_fn=lambda _: np.random.seed(),
        shuffle=True,
        **generator_params,
    )

    return train_generator, valid_generator


class EnergyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        targets,
        root_dir="/projects/ml/ligands/v5",  # featurized with extended aa defs! UNK goes to 0
        verbose: bool = False,
        ball_radius: int = 10,
        randomize: float = 0.0,
        tag_substr=[""],
        upsample=None,
        num_channels: int = 32,
        sasa_method: str = "none",
        bndgraph_type: str = "bonded",
        edgemode: str = "topk",
        edgek: Tuple[int, int] = (12, 12),
        edgedist: Tuple[float, float] = (8.0, 4.5),
        ballmode: str = "com",
        distance_feat: str = "std",
        normalize_q: bool = False,
        aa_as_het: bool = False,
        nsamples_per_p: int = 1,
        sample_mode: str = "random",
        use_AF: bool = False,
        subset_size: int = 1,
        training: bool = False,
        distance_map_dist=5.0,
        occlusion_path: Optional[Path] = None,
        AF_plddt_path: Optional[Path] = None,
        extra_edgefeat: bool = False,
        ros_dir: Optional[Path] = None,
        atom_property: Optional[Path] = None,
    ):
        super().__init__()

        self.proteins = targets
        self.datadir = root_dir
        self.verbose = verbose
        self.ball_radius = ball_radius
        self.randomize = randomize
        self.tag_substr = tag_substr
        self.sasa_method = sasa_method
        self.num_channels = num_channels
        self.bndgraph_type = bndgraph_type
        self.ballmode = ballmode  # ["com","all"]
        self.dist_fn_res = lambda x: self.get_dist_neighbors(
            x, mode=edgemode, top_k=edgek[0], dcut=edgedist[0]
        )
        self.dist_fn_atm = lambda x: self.get_dist_neighbors(
            x, mode=edgemode, top_k=edgek[1], dcut=edgedist[1]
        )
        self.distance_feat = distance_feat
        self.nsamples_per_p = nsamples_per_p
        self.nsamples = max(1, len(self.proteins) * nsamples_per_p)
        self.use_AF = use_AF
        self.subset_size = subset_size
        self.distance_map_dist = distance_map_dist
        self.extra_edgefeat = extra_edgefeat

        self.normalize_q = normalize_q
        self.sample_mode = sample_mode
        self.aa_as_het = aa_as_het
        self.training = training

        self.occlusion_path = occlusion_path
        self.AF_plddt_path = AF_plddt_path

        if upsample is None:
            self.upsample = sample_uniform
        else:
            self.upsample = upsample

        self.rosdata_dir = ros_dir
        self.atom_property = atom_property
        self.W, self.S = self.vdw_radius_depth

    def __len__(self):
        return int(self.nsamples)

    def __getitem__(self, index):
        # Select a sample decoy
        info = {}
        info["sname"] = "none"
        if self.sample_mode == "serial":
            ip = int(index / self.nsamples_per_p)
            pname = self.proteins[ip]
        else:
            pname = self.proteins[index]
        info["pname"] = pname

        fname = self.datadir / (pname + ".lig.npz")
        if not os.path.exists(fname):
            return self._skip_getitem(info)
        try:
            samples_ros = self.get_all_energy(pname)
            samples, pindex = self.get_a_sample(pname, index, samples_ros)
            prop = np.load(self.datadir / (pname + ".prop.npz"))
        except:
            print("Files does not exist!", self.datadir / pname)
            return self._skip_getitem(info)

        sname = samples["name"][pindex].reshape(self.subset_size, -1)
        info["pindex"] = pindex
        info["sname"] = sname

        # Receptor features (prop)
        charges_rec, atypes_rec, aas_rec, repsatm_idx, r2a, sasa_rec, reschain = (
            self.receptor_features(prop)
        )

        # Ligand features (lig)
        (
            xyz_ligs,
            xyz_recs,
            lddt,
            fnat,
            atypes_lig,
            bnds_lig,
            charges_lig,
            aas_lig,
            repsatm_lig,
        ) = self.per_ligand_features(samples, pindex)

        # Rosetta_energy
        rosetta_e = self.select_energy(samples_ros, sname)
        if rosetta_e is None:
            print("Raise error in getting rosetta Energy.")
            return self._skip_getitem(info)

        # aa type should directly from feature instead
        aas = np.concatenate([aas_lig, aas_rec]).astype(int)
        aas1hot = np.eye(N_AATYPE)[aas]

        # Representative atoms
        r2a = np.concatenate([np.array([0 for _ in range(xyz_ligs.shape[1])]), r2a])
        r2a1hot = np.eye(max(r2a) + 1)[r2a]
        repsatm_idx = np.concatenate(
            [
                np.array(repsatm_lig),
                np.array(repsatm_idx, dtype=int) + xyz_ligs.shape[1],
            ]
        )  # shape: (subset_size, num_atom_rep)

        # Bond properties
        bnds_rec = prop["bnds_rec"] + xyz_ligs.shape[1]  # shift index;

        # Concatenate receptor & ligand: ligand comes first
        charges = np.expand_dims(np.concatenate([charges_lig, charges_rec]), axis=1)
        if self.normalize_q:
            charges = 1.0 / (1.0 + np.exp(-2.0 * charges))

        islig = np.array(
            [1 for _ in range(xyz_ligs.shape[1])]
            + [0 for _ in range(xyz_recs.shape[1])]
        )
        islig = np.expand_dims(islig, axis=1)

        xyz = np.concatenate([xyz_ligs, xyz_recs], axis=1)
        atypes = np.concatenate([atypes_lig, atypes_rec])
        w_vdw = torch.tensor([self.W[atype] for atype in list(atypes)])  # vdw depth
        s_vdw = torch.tensor([self.S[atype] for atype in list(atypes)])  # vdw radius

        atypes_int = np.array(
            [find_gentype2num(at) for at in atypes]
        )  # string to integers
        atypes = np.eye(max(gentype2num.values()) + 1)[atypes_int]

        if samples["xyz_rec"].shape[1] != prop["xyz_rec"].shape[0]:
            print("The size of the receptors is different.")
            return self._skip_getitem(info)

        sasa = []
        sasa_lig = np.array([0.5 for _ in range(xyz_ligs.shape[1])])  # neutral value

        sasa = np.concatenate([sasa_lig, sasa_rec])
        sasa = np.expand_dims(sasa, axis=1)
        bnds = np.concatenate([bnds_lig, bnds_rec])

        # orient around ligand-COM
        center_xyz = np.mean(xyz_ligs, axis=1)
        center_xyz = np.expand_dims(center_xyz, axis=1)
        xyz = xyz - center_xyz
        xyz_ligs = xyz_ligs - center_xyz
        center_xyz[:, :, :] = 0.0

        ball_xyzs = []
        if self.ballmode == "com":
            ball_xyzs = [center_xyz]
        elif self.ballmode == "all":
            ball_xyzs = [[a[None, :] for a in xyz_lig] for xyz_lig in xyz_ligs]

        # randomize coordinate
        if self.randomize > 1e-3:
            # randxyz = 2.0*self.randomize*(0.5 - np.random.rand(self.subset_size, xyz.shape[1],3)) # -0.2 ~ 0.2
            randxyz = self.randomize * np.random.randn(
                self.subset_size, xyz.shape[1], 3
            )
            xyz = xyz + randxyz

        resfeat = [islig, aas1hot, sasa]
        resfeat_extra = []  # from res-index

        if self.use_AF:
            # Dims: islig(1) + aas1hot(21) + atypes(65) + charges(1) -> 88
            input_features = [islig, aas1hot, atypes, charges]

            # AF features (dim=1) -> 89
            input_features = self.add_extra_features(
                input_features, pname, sname, prop, samples, self.training
            )
            if input_features is None:
                return self._skip_getitem(info)

            G_atm_list, idx_ord_list = [], []
            for idx in range(self.subset_size):
                G_atm, idx_ord = self.make_atm_graphs(
                    xyz[idx],
                    ball_xyzs[idx],
                    input_features,
                    bnds,
                    xyz_ligs.shape[1],
                    atypes_int,
                    charges,
                    s_vdw,
                    w_vdw,
                )
                G_atm_list.append(G_atm)
                idx_ord_list.append(idx_ord)
        else:
            G_atm_list, idx_ord_list = [], []
            for idx in range(self.subset_size):
                G_atm, idx_ord = self.make_atm_graphs(
                    xyz[idx],
                    ball_xyzs[idx],
                    [
                        islig,
                        aas1hot,
                        atypes,
                        charges,
                    ],  # dims: islig(1) + aas1hot(33) + atypes(65) + charges(1) -> 100
                    bnds,
                    xyz_ligs.shape[1],
                    atypes_int,
                    charges,
                    s_vdw,
                    w_vdw,
                )
                G_atm_list.append(G_atm)
                idx_ord_list.append(idx_ord)

        info["islig"] = torch.tensor(islig).float()

        # Reorder by where atm idx in G_atm are
        rsds_ord_list = [r2a[idx_ord] for idx_ord in idx_ord_list]
        G_res_list, r2amap_list = [], []
        for idx in range(self.subset_size):
            G_res, r2amap = self.make_res_graph(
                xyz[idx],
                center_xyz[idx],
                resfeat,
                repsatm_idx,
                rsds_ord_list[idx],
                resfeat_extra,
            )
            G_res_list.append(G_res)
            r2amap_list.append(r2amap)

        # store which indices go to ligand atms
        ligidx_list = []
        for idx in range(self.subset_size):
            ligidx = np.zeros((len(idx_ord_list[idx]), xyz_ligs.shape[1]))
            for i in range(xyz_ligs.shape[1]):
                ligidx[i, i] = 1.0
            ligidx_list.append(torch.tensor(ligidx).float())
        info["ligidx"] = ligidx_list

        info["fnat"] = torch.tensor(fnat).float()
        info["rosetta_e"] = torch.tensor(np.array(rosetta_e)).float()
        info["lddt"] = torch.tensor(lddt).float()
        info["r2amap"] = [torch.tensor(r2amap).float() for r2amap in r2amap_list]
        info["r2a"] = torch.tensor(r2a1hot).float()
        info["repsatm_idx"] = torch.tensor(repsatm_idx).float()
        return G_atm_list, G_res_list, info

    def per_ligand_features(self, samples, pindex):
        xyz_ligs = samples["xyz"][pindex].reshape(
            self.subset_size, -1, 3
        )  # shape: (subset_size, atom_num, 3)
        xyz_recs = samples["xyz_rec"][pindex].reshape(self.subset_size, -1, 3)
        lddt = samples["lddt"][pindex].reshape(
            self.subset_size, -1
        )  # shape: (subset_size, num_atom_ligand)
        fnat = samples["fnat"][pindex].reshape(
            self.subset_size, -1
        )  # shape: (subset_size, 1)

        atypes_lig = samples["atypes_lig"][0]
        bnds_lig = samples["bnds_lig"][0]
        charges_lig = samples["charge_lig"][0]
        aas_lig = samples["aas_lig"][0]

        if "repsatm_lig" in samples:
            repsatm_lig = samples["repsatm_lig"][0]
        else:
            repsatm_lig = 0
        return (
            xyz_ligs,
            xyz_recs,
            lddt,
            fnat,
            atypes_lig,
            bnds_lig,
            charges_lig,
            aas_lig,
            repsatm_lig,
        )

    def make_atm_graphs(
        self, xyz, ball_xyzs, obt_fs, bnds, nlig, atypes_int, charges, s_vdw, w_vdw
    ):
        kd = cKDTree(xyz)
        indices = []
        for ball_xyz in ball_xyzs:
            kd_ca = cKDTree(ball_xyz)
            indices += kd_ca.query_ball_tree(kd, self.ball_radius)[0]
        indices = np.unique(indices)

        # Make sure ligand atms are ALL INCLUDED
        idx_ord = [i for i in indices if i < nlig]
        idx_ord += [i for i in indices if i >= nlig]
        old_idx_map = {new_i: old_i for new_i, old_i in enumerate(idx_ord)}

        # Concatenate all one-body-features
        obt = []
        for f in obt_fs:
            if len(f) > 0:
                obt.append(f[idx_ord])
        obt = np.concatenate(obt, axis=-1)

        xyz_old = xyz
        bnds_old = bnds
        xyz = xyz[idx_ord]
        bnds = [bnd for bnd in bnds if (bnd[0] in idx_ord) and (bnd[1] in idx_ord)]
        bnds_bin = np.zeros((len(xyz), len(xyz)))
        for i, j in bnds:
            k, l = idx_ord.index(i), idx_ord.index(j)
            bnds_bin[k, l] = bnds_bin[l, k] = 1
        for i in range(len(xyz)):
            bnds_bin[i, i] = 1  # self

        # Concatenate coord & centralize xyz to ca.
        xyz = torch.tensor(xyz).float()

        # for G_atm
        u, v = self.dist_fn_atm(xyz[None,])  # Num_edges

        # Edge feature: distance
        w = torch.sqrt(torch.sum((xyz[v] - xyz[u]) ** 2, axis=-1) + 1e-6)[..., None]
        w1hot = self.distance_feature(self.distance_feat, w, 0.5, 5.0)
        bnds_bin = torch.tensor(
            bnds_bin[v, u]
        ).float()  # replace first bin (0.0~0.5 Ang) to bond info
        w1hot[:, 0] = bnds_bin  # Shape: (Num_edges, 2)

        # Other edge features
        if self.extra_edgefeat:
            uv = np.array(torch.concatenate([u[None,], v[None,]]).T)
            edge_hbond = torch.tensor(
                self.is_hbond(uv, bnds_old, atypes_int, xyz_old, old_idx_map),
                dtype=torch.int,
            )
            edge_pol_apol = torch.tensor(
                self.is_polar_apolar(
                    uv, bnds_old, charges, atypes_int, xyz_old, old_idx_map
                ),
                dtype=torch.int,
            )
            w1hot = torch.concatenate(
                [w1hot, edge_hbond[:, None], edge_pol_apol], dim=-1
            )

        # Other node features
        S_vdw = [s_vdw[i] for i in idx_ord]
        W_vdw = [w_vdw[i] for i in idx_ord]

        ## Construct graphs
        # G_atm: graph for all atms
        G_atm = dgl.graph((u, v))
        G_atm.ndata["0"] = torch.tensor(obt).float()
        G_atm.ndata["x"] = xyz[:, None, :]
        G_atm.ndata["vdw_sig"] = torch.tensor(S_vdw).float()
        G_atm.ndata["vdw_dep"] = torch.tensor(W_vdw).float()
        G_atm.edata["rel_pos"] = xyz[v] - xyz[u]
        G_atm.edata["0"] = w1hot

        return G_atm, idx_ord

    def get_all_energy(self, pname):
        samples_ros = np.load(self.rosdata_dir / (pname + ".npz"), allow_pickle=True)
        return samples_ros

    def select_energy(self, samples_ros, sname):
        rosetta_e = []
        for i in range(len(sname)):
            name = str(sname[i][0] + ".pdb")
            index = np.where(samples_ros["pdb"] == name)[0]
            if len(index) == 0:
                return None
            rosetta_e.append(
                [
                    samples_ros["vdw"][index][0],
                    samples_ros["solvation"][index][0],
                    samples_ros["Coulomb"][index][0],
                    samples_ros["Hbond"][index][0],
                ]
            )
        return rosetta_e

    def get_a_sample(self, pname, index, samples_ros):
        samples = np.load(self.datadir / (pname + ".lig.npz"), allow_pickle=True)
        pindices = list(range(len(samples["name"])))
        names = samples["name"]
        e_names = samples_ros["pdb"]
        xsorted = np.argsort(e_names)
        re_idx = xsorted[np.searchsorted(e_names[xsorted], names)]
        vdw = samples_ros["vdw"][re_idx]
        # Filtering repulsion
        if self.sample_mode == "random":
            pindex = np.random.choice(
                pindices, size=self.subset_size, replace=False, p=self.upsample(vdw)
            )
        elif self.sample_mode == "serial":
            pindex = [index % len(pindices)]
        return samples, pindex

    def receptor_features(self, prop):
        charges_rec = prop["charge_rec"]
        atypes_rec = prop["atypes_rec"]
        aas_rec = prop["aas_rec"]
        repsatm_idx = prop[
            "repsatm_idx"
        ]  # representative atm idx for each residue (e.g. CA); receptor only
        r2a = (
            np.array(prop["residue_idx"], dtype=int) + 1
        )  # Add ligand as the first residue
        reschains = prop["reschains"]
        sasa_rec = 0
        if "sasa_rec" in prop:
            sasa_rec = prop["sasa_rec"]
        return charges_rec, atypes_rec, aas_rec, repsatm_idx, r2a, sasa_rec, reschains

    def make_res_graph(
        self, xyz, center_xyz, obt_fs, repsatm_idx, rsds_in_Gatm, extra_fs
    ):
        xyz_reps = xyz[repsatm_idx]
        xyz_reps = torch.tensor(xyz_reps - center_xyz).float()

        # Grabbing a < dist neighbor
        kd = cKDTree(xyz_reps)
        kd_ca = cKDTree(center_xyz)
        indices = kd_ca.query_ball_tree(kd, 1000.0)[
            0
        ]  # any huge number to cover entire protein
        reps_idx = [repsatm_idx[i] for i in indices]

        r2amap = np.zeros(len(rsds_in_Gatm), dtype=int)
        for i, rsd in enumerate(rsds_in_Gatm):
            if rsd in indices:
                r2amap[i] = indices.index(rsd)
            else:
                sys.exit("unexpected resno %d" % (rsd))
        r2amap = np.eye(max(indices) + 1)[r2amap]  # one_hot

        obt = []
        for f in obt_fs:
            if len(f) > 0:
                obt.append(f[reps_idx])

        for f in extra_fs:
            if len(f) > 0:
                obt.append(f[indices][:, None])  # should match the size

        obt = np.concatenate(obt, axis=-1)

        u, v = self.dist_fn_res(xyz_reps[None,])
        D = torch.sqrt(torch.sum((xyz_reps[v] - xyz_reps[u]) ** 2, axis=-1) + 1e-6)[
            ..., None
        ]
        D1hot = self.distance_feature(self.distance_feat, D, 1.0, 10.0)

        G_res = dgl.graph((u, v))
        G_res.ndata["0"] = torch.tensor(obt).float()
        G_res.ndata["x"] = xyz_reps[:, None, :]
        G_res.edata["rel_pos"] = xyz_reps[v] - xyz_reps[u]
        G_res.edata["0"] = D1hot

        return G_res, r2amap

    def get_nearest_protein_atom_index(self, xyz, ball_xyzs, distance=5):
        kd = cKDTree(xyz)
        indices = []
        for ball_xyz in ball_xyzs:
            kd_ca = cKDTree(ball_xyz)
            indices += kd_ca.query_ball_tree(kd, distance)[0]
        indices = np.unique(indices)
        return indices

    def add_extra_features(
        self, input_features, pname, sname, prop, samples, training: bool = False
    ):
        """
        Using AF feature as extra features.
        """
        af_feature_path = Path(self.AF_plddt_path)

        # try:
        if training and ("native_dock" in sname or "near_native.native" in sname):
            af_feature = np.ones([samples["xyz_rec"].shape[1], 1])
        elif af_feature_path is None:
            af_feature = np.ones([samples["xyz_rec"].shape[1], 1])
        else:
            af_feature = np.load(af_feature_path.joinpath(f"{pname}_conf.npy"))
            af_feature = af_feature[prop["residue_idx"]]
        lig_feature = np.zeros(
            [samples["xyz"].shape[1], af_feature.shape[-1]]
        )  # samples['xyz'].shape[1]: ligand atom num
        af_feature = np.concatenate([lig_feature, af_feature], axis=0)

        input_features.append(af_feature)
        # except:
        #     return None

        return input_features

    def distance_feature(
        self, mode: str, d: Tensor, binsize: float = 0.5, maxd: float = 5.0
    ):
        if mode == "1hot":
            b = (d / binsize).long()
            nbin = int(maxd / binsize)
            b[b >= nbin] = nbin - 1
            d1hot = torch.eye(nbin)[b].float()
            feat = d1hot.squeeze()
        elif mode == "std":  # sigmoid
            d0 = 0.5 * (maxd - binsize)  # center
            m = 5.0 / d0  # slope
            feat = 1.0 / (1.0 + torch.exp(-m * (d - d0)))
            feat = feat.repeat(1, 2)  # hacky!! make as 2-dim
        return feat

    def get_dist_neighbors(self, X, mode="topk", top_k=16, dcut=4.5, eps=1e-6):
        """Pairwise euclidean distances"""
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = torch.sqrt(torch.sum(dX**2, 3) + eps)

        if mode in ["topk", "mink"]:
            D_neighbors, E_idx = torch.topk(D, top_k + 1, dim=-1, largest=False)
            # exclude self-connection
            D_neighbor = D_neighbors[:, :, 1:]
            E_idx = E_idx[:, :, 1:]
            u = (
                torch.tensor(np.arange(E_idx.shape[1]))[:, None]
                .repeat(1, E_idx.shape[2])
                .reshape(-1)
            )
            v = E_idx[0,].reshape(-1)

            # append more than k that are within dcut
            if mode == "mink":
                nprv = len(u)
                mask = torch.where(torch.tril(D) < 1.0e-6, 100.0, 1.0)
                _, uD, vD = torch.where((mask * D) < dcut)
                uv = np.array(list(zip(u, v)) + list(zip(uD, vD)))
                uv = np.unique(uv, axis=0)
                u = [a for a, b in uv]
                v = [b for a, b in uv]
                # print("nedge:",dcut,nprv,len(uD),len(u))

        elif mode == "distT":
            mask = torch.where(torch.tril(D) < 1.0e-6, 100.0, 1.0)
            _, u, v = torch.where((mask * D) < dcut)

        elif mode == "dist":
            _, u, v = torch.where(D < dcut)

        return u, v

    def is_hbond(
        self,
        edges: np.ndarray,
        bnds: np.ndarray,
        atypes_int: list,
        xyz: np.ndarray,
        old_idx_map: dict,
    ) -> np.ndarray:
        edges = np.sort(edges, axis=1)
        src_indices = np.vectorize(old_idx_map.get)(edges[:, 0])
        dst_indices = np.vectorize(old_idx_map.get)(edges[:, 1])
        edges = np.concatenate([src_indices[:, None], dst_indices[:, None]], axis=1)

        # Filter out self-edges and covalant bond
        result = np.logical_and(
            src_indices != dst_indices,
            np.invert((edges[:, None] == bnds).all(-1).any(-1)),
        )
        polar_hydrogen = [15, 16]  # HO, NH
        h_acceptor = [
            20,
            24,
            31,
            32,
            34,
            35,
            36,
            38,
            39,
            40,
            44,
        ]  # Nad, Ngu1, Ohx, Oet, Oad, Oat(35), Ofu, OG2, OG3, OG31(40), SG2
        src_atypes, dst_atypes = atypes_int[src_indices], atypes_int[dst_indices]
        src_xyz, dst_xyz = xyz[src_indices], xyz[dst_indices]

        # Atom atypes
        is_donor_acceptor = np.logical_and(
            np.isin(src_atypes, polar_hydrogen), np.isin(dst_atypes, h_acceptor)
        )
        is_acceptor_donor = np.logical_and(
            np.isin(src_atypes, h_acceptor), np.isin(dst_atypes, polar_hydrogen)
        )
        result *= np.logical_or(is_donor_acceptor, is_acceptor_donor)

        # Distance
        distances = np.linalg.norm(src_xyz - dst_xyz, axis=1)
        result *= distances <= 2.5

        # Acceptor and Donor conditions
        angle_mask = np.ones_like(result)
        for i in range(len(result)):
            if result[i] is False:
                angle_mask[i] = False
                continue
            if is_donor_acceptor[i]:
                dst = dst_indices[i]
                if (bnds[:, 0] == dst).sum():
                    neigh_index = bnds[bnds[:, 0] == dst, 1]
                    neigh_xyz = xyz[neigh_index].mean(axis=0)
                else:
                    neigh_index = bnds[bnds[:, 1] == dst, 0]
                    neigh_xyz = xyz[neigh_index].mean(axis=0)
                v1 = src_xyz[i] - dst_xyz[i]
                v2 = neigh_xyz - dst_xyz[i]
                cos_angle = cosine_angle(v1, v2)
                if not (cos_angle < np.radians(180) and cos_angle > np.radians(120)):
                    angle_mask[i] = False
                src = src_indices[i]
                if (bnds[:, 0] == src).sum():
                    neigh_index = bnds[bnds[:, 0] == src, 1]
                    neigh_xyz = xyz[neigh_index].mean(axis=0)
                else:
                    neigh_index = bnds[bnds[:, 1] == src, 0]
                    neigh_xyz = xyz[neigh_index].mean(axis=0)
                v1 = dst_xyz[i] - src_xyz[i]
                v2 = neigh_xyz - src_xyz[i]
                cos_angle = cosine_angle(v1, v2)
                if not (cos_angle < np.radians(180) and cos_angle > np.radians(150)):
                    angle_mask[i] = False
            elif is_acceptor_donor[i]:
                src = src_indices[i]
                if (bnds[:, 0] == src).sum():
                    neigh_index = bnds[bnds[:, 0] == src, 1]
                    neigh_xyz = xyz[neigh_index].mean(axis=0)
                else:
                    neigh_index = bnds[bnds[:, 1] == src, 0]
                    neigh_xyz = xyz[neigh_index].mean(axis=0)
                v1 = dst_xyz[i] - src_xyz[i]
                v2 = neigh_xyz - src_xyz[i]
                cos_angle = cosine_angle(v1, v2)
                if not (cos_angle < np.radians(180) and cos_angle > np.radians(120)):
                    angle_mask[i] = False
                dst = dst_indices[i]
                if (bnds[:, 0] == dst).sum():
                    neigh_index = bnds[bnds[:, 0] == dst, 1]
                    neigh_xyz = xyz[neigh_index].mean(axis=0)
                else:
                    neigh_index = bnds[bnds[:, 1] == dst, 0]
                    neigh_xyz = xyz[neigh_index].mean(axis=0)
                v1 = src_xyz[i] - dst_xyz[i]
                v2 = neigh_xyz - dst_xyz[i]
                cos_angle = cosine_angle(v1, v2)
                if not (cos_angle < np.radians(180) and cos_angle > np.radians(150)):
                    angle_mask[i] = False
        result *= angle_mask
        return result

    def is_polar_apolar(
        self,
        edge: np.ndarray,
        bnds: np.ndarray,
        charges: np.ndarray,
        atypes_int: list,
        xyz: np.ndarray,
        old_idx_map: dict,
    ) -> Tuple[bool, bool]:
        """
        Returns:
            is_polar_apolar, is_apolar_apolar
        """

        def set_dist_cut(sa, da):
            hydrogens = [15, 16, 17]
            if sa in hydrogens and da in hydrogens:
                dist_pol, dist_apol = 4.0, 4.0
            elif sa in hydrogens or da in hydrogens:
                dist_pol, dist_apol = 4.5, 4.5
            else:
                dist_pol, dist_apol = 5.0, 5.0
            return dist_pol, dist_apol

        edge = np.sort(edge, axis=1)
        src = np.vectorize(old_idx_map.get)(edge[:, 0])
        dst = np.vectorize(old_idx_map.get)(edge[:, 1])
        src_atype, dst_atype = atypes_int[src], atypes_int[dst]
        dist_cut = np.array(list(map(set_dist_cut, src_atype, dst_atype)))
        dist_pol, dist_apol = dist_cut[:, 0], dist_cut[:, 1]
        result = src != dst
        result *= np.invert((edge[:, None] == bnds).all(-1).any(-1))
        result = np.concatenate([result[:, None], result[:, None]], axis=1)
        src_charge, dst_charge = charges[src], charges[dst]
        src_xyz, dst_xyz = xyz[src], xyz[dst]
        # Apolar-apolar
        cond = ((abs(src_charge) < 0.15) & (abs(dst_charge) < 0.15)).squeeze()
        result[:, 1] = result[:, 1] * cond
        cond = np.linalg.norm(src_xyz - dst_xyz, axis=1) < dist_apol
        result[:, 1] = result[:, 1] * cond
        # Polar-apolar
        cond = (
            (abs(src_charge) > 0.3) & (abs(dst_charge) < 0.15)
            | (abs(src_charge) < 0.15) & (abs(dst_charge) > 0.3)
        ).squeeze()
        result[:, 0] = result[:, 0] * cond
        cond = np.linalg.norm(src_xyz - dst_xyz, axis=1) < dist_pol
        result[:, 0] = result[:, 0] * cond
        return result

    @property
    def vdw_radius_depth(self):
        with open(self.atom_property) as file:
            W, S = {}, {}
            for line in file:
                if (
                    not line.strip()
                    or line.strip().startswith("#")
                    or line.strip().startswith("NAME")
                ):
                    continue
                fields = line.split()
                try:
                    atom_name = fields[0]
                    if fields[3] != "":
                        w = float(fields[3])
                    if fields[2] != "":
                        s = float(fields[2])
                    W[atom_name] = w
                    S[atom_name] = s
                except ValueError:
                    print(f"Parsing error: {line}")
        return W, S

    @staticmethod
    def _skip_getitem(info):
        return False, False, info


def load_dataset_energy(
    set_params: EasyDict,
    split_path: Path,
    generator_params: EasyDict,
    setsuffix: str = "Clean_MT2",
):
    # Datasets
    train_set = EnergyDataset(
        np.load(split_path.joinpath(f"train{setsuffix}.npy")),
        training=True,
        **set_params,
    )

    valid_params = deepcopy(set_params)
    valid_params.randomize = 0.0
    val_set = EnergyDataset(
        np.load(split_path.joinpath(f"valid{setsuffix}.npy")),
        training=True,
        **valid_params,
    )
    # DataLoaders
    train_generator = data.DataLoader(
        train_set,
        worker_init_fn=lambda _: np.random.seed(),
        shuffle=False,
        sampler=DfAFSampler(
            np.load(split_path.joinpath(f"train_weight{setsuffix}.npy"))
        ),
        **generator_params,
    )

    valid_generator = data.DataLoader(
        val_set,
        worker_init_fn=lambda _: np.random.seed(),
        shuffle=True,
        **generator_params,
    )

    return train_generator, valid_generator

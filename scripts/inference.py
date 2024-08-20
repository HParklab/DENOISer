import sys
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from dgl import DGLGraph
import pandas as pd
from scipy.spatial.distance import cdist
from pathlib import Path
from easydict import EasyDict
from copy import deepcopy
import time

from denoiser.arguments import set_arguments
from denoiser.dataset import N_AATYPE, collate, LocalDataset
from denoiser.models import PIScoreNet, EnergyNet
from denoiser.utils.train_utils import count_parameters, split_energy


SCRIPTPATH = Path(os.path.abspath(__file__)).parent
NRESFEATURE = N_AATYPE + 2
DEVICE = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")


def set_each_params(local_args):
    train_params = {
        "base_learning_rate": local_args.lr,
        "w_lossG": local_args.w_lossG,
        "w_lossL": local_args.w_lossG,
        "w_lossHbond": local_args.w_lossHbond,
        "w_lossPolApol": local_args.w_lossPolApol,
        "w_lossApolApol": local_args.w_lossApolApol,
        "w_lossDist": local_args.w_lossDist,
        "w_lossR": local_args.w_lossR,
        "w_reg": local_args.w_reg,
        "clip_grad": 1.0,
    }

    model_params = {
        "drop_out": local_args.drop_out,
        "num_layers": local_args.num_layers,
        "dim_channels": (32, 32, 32),
        "inter_hidden_channels": local_args.inter_hidden_channels,
        "projection_dims": local_args.projection_dims,
        "l0_in_features": local_args.l0_in_features,
        "threshold_distance_diff": local_args.threshold_distance_diff,
        "abs_min_diff": local_args.abs_min_diff,
        "dist_bins": local_args.dist_bins,
        "backbone": local_args.backbone,
    }

    energy_model_params = {
        "encoder_edge_features": 2,
        "drop_out": local_args.e_drop_out,
        "encoder_num_layers": local_args.e_num_layers,
        "encoder_num_channels": 32,
        "projection_dims": local_args.e_projection_dims,
        "input_l0_feature": local_args.e_l0_in_features,
        "energy_num_channels": 32,
        "backbone": local_args.e_backbone,
    }

    data_params = {
        "root_dir": None,
        "ball_radius": local_args.ball_radius,
        "ballmode": "all",
        "upsample": None,
        "sasa_method": "sasa",
        "sample_mode": "serial",
        "randomize": 0.0,
        "edgemode": local_args.edge_mode,
        "edgedist": local_args.edge_dist,
        "distance_feat": "std",
        "use_AF": "only_conf",
        "nsamples_per_p": 1,
        "subset_size": 1,
        "masking_coeff": local_args.masking_coeff,
        "distance_map_dist": local_args.distance_map_dist,
        "scaling_apol_pol": local_args.scaling_apol_pol,
        "scaling_all": local_args.scaling_all,
        "atom_property": SCRIPTPATH / "atom_properties_f.txt",
    }

    generator_params = {
        "num_workers": local_args.ncpu,
        "pin_memory": True,
        "collate_fn": collate,
        "batch_size": 1,
    }

    return (
        EasyDict(train_params),
        EasyDict(model_params),
        EasyDict(energy_model_params),
        EasyDict(data_params),
        EasyDict(generator_params),
    )


def load_model(model_params, energy_model_params):
    model = PIScoreNet(
        num_layers=model_params.num_layers,
        l0_in_features=(NRESFEATURE, model_params.l0_in_features),  # N_AATYPE=33
        l1_in_features=(0, 0),
        dim_channels=model_params.dim_channels,
        inter_hidden_channels=model_params.inter_hidden_channels,
        drop_out=model_params.drop_out,
        dist_clamp=model_params.threshold_distance_diff,
        dist_bins=model_params.dist_bins,
        nsqueeze=model_params.projection_dims,
        backbone=model_params.backbone,
    )

    energy_model = EnergyNet(
        encoder_num_channels=energy_model_params.encoder_num_channels,
        input_l0_feature=energy_model_params.input_l0_feature,
        encoder_num_layers=energy_model_params.encoder_num_layers,
        energy_num_channels=energy_model_params.energy_num_channels,
        drop_out=energy_model_params.drop_out,
        nsqueeze=energy_model_params.projection_dims,
        backbone=energy_model_params.backbone,
    )

    model.to(DEVICE)
    energy_model.to(DEVICE)
    print("nparams of NL network: ", count_parameters(model))
    print("nparams of BE network: ", count_parameters(energy_model))

    if os.path.exists(f"{SCRIPTPATH}/../model/local.pkl" % ()):
        print("Loading a checkpoint")
        model_checkpoint = torch.load(
            f"{SCRIPTPATH}/../model/local.pkl",
            map_location=DEVICE,
        )
        energy_model_checkpoint = torch.load(
            f"{SCRIPTPATH}/../model/local_energy.pkl",
            map_location=DEVICE,
        )

        model.load_state_dict(model_checkpoint["model_state_dict"])
        energy_model.load_state_dict(energy_model_checkpoint["model_state_dict"])
    else:
        sys.exit("no model found")

    return model, energy_model


def add_final_score(df: pd.DataFrame, self_docking_mode: bool = False):
    def min_max_scaling(x):
        return (x - x.min()) / (x.max() - x.min())

    df["pred_scaled"] = df.groupby("pname")["pred_fnat"].transform(min_max_scaling)
    df["energy_i"] = -df["energy"]
    df["energy_scaled"] = df.groupby("pname")["energy_i"].transform(min_max_scaling)
    if self_docking_mode:
        df["consensus_score"] = df["pred_scaled"] * 0.1 + df["energy_scaled"] * 0.9
    else:
        df["consensus_score"] = (df["pred_scaled"] + df["energy_scaled"]) / 2
    return df


def clash_penalty(G_atm: DGLGraph, penalty_weight: float = 0.01) -> float:
    n_lig = (G_atm.ndata["0"][:, 0] == 1).sum()
    vdw_sig = G_atm.ndata["vdw_sig"]
    vdw_sig = vdw_sig[:n_lig][:, None] + vdw_sig[n_lig:][None,]

    xyz = G_atm.ndata["x"].squeeze()
    xyz_lig, xyz_prop = xyz[:n_lig], xyz[n_lig:]
    pair_dist = torch.tensor(cdist(xyz_lig, xyz_prop))
    p = torch.where(pair_dist < vdw_sig, vdw_sig - pair_dist, 0.0)
    torch.diagonal(p, 0).zero_()
    return (p.square().sum() * penalty_weight).item()


def enumerate_an_epoch(
    model: nn.Module,
    energy_model: nn.Module,
    generator: DataLoader,
    self_docking_mode: bool = False,
    no_gald: bool = False,
) -> None:
    model.eval()
    energy_model.eval()
    result = {"pname": [], "sname": [], "fnat": [], "pred_fnat": [], "energy": []}
    for G_atm, G_res, G_high, info in generator:
        # Get prediction and target value
        if not G_atm:
            print("skip %s %s" % (info["pname"], info["sname"]))
            if info["sname"] == "end":
                break
            continue
        if G_atm.num_edges() > 25000:
            print(
                f"There are too many contacts in the protein-ligand complex ({info['sname'][0]}). Check the PDB file."
            )
            continue
        if no_gald:
            penalty = clash_penalty(G_atm)
        else:
            penalty = 0

        for key in info:
            if key in [
                "hbond_masks",
                "polar_apolar_masks",
                "apolar_apolar_masks",
                "distance_masks",
                "xtal_hbond_masks",
                "xtal_polar_apolar_masks",
                "xtal_apolar_apolar_masks",
                "xtal_distance_masks",
                "dist_rec_indices",
            ]:
                info[key] = [mask.to(DEVICE) for mask in info[key]]
            else:
                if isinstance(info[key], list):
                    pass
                else:
                    info[key] = info[key].to(DEVICE)

        fnat = info["fnat"]
        if isinstance(G_atm, tuple) or isinstance(G_res, tuple):
            print(G_atm)
            print(G_res)
        G_atm_old = deepcopy(G_atm)
        _, _, _, _, pred_fnat, _ = model(
            G_atm.to(DEVICE), G_res.to(DEVICE), G_high.to(DEVICE), info
        )
        pred_fnat = pred_fnat.to(DEVICE)

        pred_vdw, pred_elec, pred_solv = energy_model(G_atm_old.to(DEVICE))
        pred_vdw, pred_elec, pred_solv = pred_vdw * 20, pred_elec * 5, pred_solv * 10
        sum_pred = float(pred_vdw + pred_elec + pred_solv)
        result["pname"].append(info["pname"][0])
        result["sname"].append(info["sname"][0])
        result["fnat"].append(fnat.item())
        result["pred_fnat"].append(pred_fnat.item() - penalty)
        result["energy"].append(sum_pred)
    result = pd.DataFrame(result)
    result = add_final_score(result, self_docking_mode)
    result = result.sort_values(by="consensus_score", ascending=False)
    for idx, row in result.iterrows():
        print(
            "%s %s : true/pred: complex_lDDT %6.4f/%6.4f  energy: %6.4f | pred_scaled: %6.4f  energy_scaled: %6.4f | consensus_score: %6.6f"
            % (
                row["pname"],
                row["sname"],
                row["fnat"],
                row["pred_fnat"],
                row["energy"],
                row["pred_scaled"],
                row["energy_scaled"],
                row["consensus_score"],
            )
        )


def main():
    local_args = set_arguments()
    train_params, model_params, energy_model_params, data_params, generator_params = (
        set_each_params(local_args)
    )
    model, energy_model = load_model(model_params, energy_model_params)
    print("\n" + "-" * 40)
    print("Device:", DEVICE)
    print("Num.threads:", torch.get_num_threads(), "\n")
    print("Warning: --AF is not given. plDDT will be set to ones")
    print("-" * 40, "\n")

    data_params.root_dir = Path(local_args.input_features)
    plddt_path = Path(local_args.AF) if local_args.AF is not None else local_args.AF
    data_params.AF_plddt_path = plddt_path
    data_params.ros_dir = local_args.energy
    self_docking_mode = local_args.self_docking
    no_gald = local_args.no_gald
    test_files = list(
        [t.split(".")[0] for t in os.listdir(data_params.root_dir) if "lig" in t]
    )

    stime = time.time()
    for target in test_files:
        try:
            ndata = len(
                np.load(data_params.root_dir.joinpath(f"{target}.lig.npz"))["name"]
            )
            if ndata == 0:
                continue

            data_params.nsamples_per_p = ndata
            targets = [target]

            generator = DataLoader(
                LocalDataset(targets, **data_params),
                worker_init_fn=lambda _: np.random.seed(),
                **generator_params,
            )
            print(target)
            print(len(generator))

            with torch.no_grad():
                enumerate_an_epoch(
                    model, energy_model, generator, self_docking_mode, no_gald
                )

        except Exception as e:
            print(f"Raise Error in {target}. Skip")
            if local_args.debug:
                print(e)
                break
            continue

    print("Done! Run time:", time.time() - stime, " seconds")


if __name__ == "__main__":
    main()

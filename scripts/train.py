import sys
import os
import wandb
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from easydict import EasyDict
from typing import List

# from denoiser.arguments import set_arguments
from denoiser.arguments import set_arguments_train
from denoiser.dataset import load_dataset, collate_training
from denoiser.models import PIScoreNet
from denoiser.utils.train_utils import (
    cosine_schedule,
    upsample_auxil,
    count_parameters,
    set_random_seed,
    pairwise_2d,
)
from denoiser.utils.chem_utils import distogram, N_AATYPE


PATH = Path(os.path.abspath(__file__)).parent
NRESFEATURE = N_AATYPE + 2  # is_lig, sasa
VALID_REPEAT = 3
DEVICE = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")


def set_each_params(local_args):
    split_path = Path(local_args.data_path)

    train_params = {
        "model_name": local_args.model_name,
        "base_learning_rate": local_args.lr,
        "max_epochs": 100,
        "w_lossG": local_args.w_lossG,
        "w_lossL": local_args.w_lossL,
        "w_lossHbond": local_args.w_lossHbond,  # Interaction difference map
        "w_lossPolApol": local_args.w_lossPolApol,  # Interaction difference map
        "w_lossApolApol": local_args.w_lossApolApol,  # Interaction difference map
        "w_lossDist": local_args.w_lossDist,  # distance difference map
        "w_lossR": local_args.w_lossR,  # ranking loss; KL divergence unit
        "w_reg": local_args.w_reg,
        "setsuffix": "v6",
        "clip_grad": 1.0,
        "fix_random_seed": local_args.fix_random_seed,
        "scheduling": local_args.scheduling,
        "init0iter": local_args.init0iter,
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

    data_params = {
        "root_dir": Path(local_args.input_features),
        "AF_plddt_path": Path(local_args.AF),
        "ballmode": "all",
        "upsample": upsample_auxil,
        "sasa_method": "sasa",
        "randomize": local_args.randomize,
        "edgemode": local_args.edge_mode,
        "edgek": (0, 0),
        "edgedist": local_args.edge_dist,
        "distance_feat": "std",
        "use_AF": "only_conf",
        "nsamples_per_p": 1,
        "subset_size": local_args.subset_size,
        "masking_coeff": local_args.masking_coeff,
        "distance_map_dist": local_args.distance_map_dist,
        "scaling_apol_pol": local_args.scaling_apol_pol,
        "scaling_all": local_args.scaling_all,
        "extra_edgefeat": local_args.extra_edgefeat,
        "atom_property": PATH / "atom_properties_f.txt",
    }

    generator_params = {
        "num_workers": 0,
        "pin_memory": True,
        "collate_fn": collate_training,
        "batch_size": 1,
    }

    return (
        split_path,
        EasyDict(train_params),
        EasyDict(model_params),
        EasyDict(data_params),
        EasyDict(generator_params),
    )


def load_model(train_params, model_params, data_params):
    model_name = train_params.model_name
    base_learning_rate = train_params.base_learning_rate
    extra_edgefeat = data_params.extra_edgefeat

    if extra_edgefeat:
        edge_features = (2, 5, 2)
    else:
        edge_features = (2, 2, 2)

    model = PIScoreNet(
        num_layers=model_params.num_layers,
        l0_in_features=(NRESFEATURE, model_params.l0_in_features),  # N_AATYPE=33
        l1_in_features=(0, 0, 0),
        dim_channels=model_params.dim_channels,
        edge_features=edge_features,
        inter_hidden_channels=model_params.inter_hidden_channels,
        drop_out=model_params.drop_out,
        dist_clamp=model_params.threshold_distance_diff,
        dist_bins=model_params.dist_bins,
        nsqueeze=model_params.projection_dims,
        backbone=model_params.backbone,
    )

    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_learning_rate)
    print("nparams: ", count_parameters(model))

    epoch = 0
    train_loss = {
        "total": [],
        "global": [],
        "local": [],
        "rank": [],
        "distance_map": [],
        "hbond_map": [],
        "polar_apolar_map": [],
        "apolar_apolar_map": [],
        "reg": [],
    }
    valid_loss = {
        "total": [],
        "global": [],
        "local": [],
        "rank": [],
        "distance_map": [],
        "hbond_map": [],
        "polar_apolar_map": [],
        "apolar_apolar_map": [],
        "reg": [],
    }

    if not PATH.joinpath("models", model_name).exists():
        print("Creating a new dir at", str(PATH / "models" / model_name))
        PATH.joinpath("models", model_name).mkdir(parents=True, exist_ok=True)

    return epoch, model, optimizer, train_loss, valid_loss


def save_scatter_plot(epoch: int, stdout_text_list: List[str]) -> None:
    if epoch % 10 == 0:
        with open("temp_train_log.txt", "w") as f:
            f.writelines(stdout_text_list)
        fnat_list = []
        pred_list = []

        with open("temp_train_log.txt") as f:
            for line in f:
                if "Epoch" in line and "TRAIN" in line:
                    idx_temp = line.split("fnat/pred")[-1].split("|")[0].split("/")
                    fnat, pred = [float(temp) for temp in idx_temp]
                    fnat_list.append(fnat)
                    pred_list.append(pred)

        result = np.array([fnat_list, pred_list])
        x, y = result[0, :], result[1, :]
        fig, ax = plt.subplots()
        ax.scatter(x, y, s=1, alpha=0.7)
        ax.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10), color="r")
        wandb.log(
            {f"scatter_plot_epoch{epoch}(x: fnat, y: predicted)": wandb.Image(fig)}
        )


def calc_losses(model_prediction, info, model_params):
    threshold_distance_diff = model_params["threshold_distance_diff"]
    abs_min_diff = model_params["abs_min_diff"]
    dist_bins = model_params["dist_bins"]

    (
        hbond_embed,
        polar_apolar_embed,
        apolar_apolar_embed,
        distance_embed,
        pred_fnat,
        pred_lddt,
    ) = model_prediction
    LossG = torch.nn.SmoothL1Loss()
    LossL = torch.nn.SmoothL1Loss()
    LossHbond = torch.nn.SmoothL1Loss()
    LossPolarApolar = torch.nn.SmoothL1Loss()
    LossApolarApolar = torch.nn.SmoothL1Loss()
    LossDist = torch.nn.NLLLoss()
    LossRanking = torch.nn.MarginRankingLoss()
    fnat = info["fnat"]
    lddt = info["lddt"][None, :]
    pred_fnat = pred_fnat.to(DEVICE)
    pred_lddt = pred_lddt.to(DEVICE)
    loss1 = LossG(pred_fnat, fnat.float())
    loss2 = LossL(pred_lddt, lddt.float())
    loss3, loss4, loss5, loss6 = 0.0, 0.0, 0.0, 0.0

    if lddt.size() != pred_lddt.size():
        return [None] * 7

    for i in range(len(info["hbond_masks"])):
        inter_diff = info["hbond_masks"][i] - info["xtal_hbond_masks"][i]
        mask_idx = info["hbond_masks"][i].nonzero(as_tuple=True)
        if len(mask_idx[0]) != 0:
            loss3 += LossHbond(hbond_embed[i][mask_idx], inter_diff[mask_idx])
        else:
            loss3 += torch.tensor(0.0).to(DEVICE)
        inter_diff = info["polar_apolar_masks"][i] - info["xtal_polar_apolar_masks"][i]
        mask_idx = info["polar_apolar_masks"][i].nonzero(as_tuple=True)
        if len(mask_idx[0]) != 0:
            loss4 += LossPolarApolar(
                polar_apolar_embed[i][mask_idx], inter_diff[mask_idx]
            )
        else:
            loss4 += torch.tensor(0.0).to(DEVICE)
        inter_diff = (
            info["apolar_apolar_masks"][i] - info["xtal_apolar_apolar_masks"][i]
        )
        mask_idx = info["apolar_apolar_masks"][i].nonzero(as_tuple=True)
        if len(mask_idx[0]) != 0:
            loss5 += LossApolarApolar(
                apolar_apolar_embed[i][mask_idx], inter_diff[mask_idx]
            )
        else:
            loss5 += torch.tensor(0.0).to(DEVICE)

        distance_diff_masks = (
            info["distance_masks"][i] - info["xtal_distance_masks"][i]
        ).contiguous()
        distance_diff_masks = distogram(
            distance_diff_masks,
            -threshold_distance_diff,
            threshold_distance_diff,
            dist_bins,
            abs_min_diff,
        )
        loss6 += LossDist(
            distance_embed[i].unsqueeze(0).permute(0, 3, 1, 2).contiguous(),
            distance_diff_masks.unsqueeze(0),
        )

    pairwise_pred = pairwise_2d(pred_fnat)
    a, b = pairwise_pred[0, :], pairwise_pred[1, :]
    pairwise_true = pairwise_2d(fnat)
    c = torch.where(pairwise_true[0, :] > pairwise_true[1, :], 1, -1)
    lossR = LossRanking(a, b, c)
    return loss1, loss2, loss3, loss4, loss5, loss6, lossR


def enumerate_an_epoch(
    model,
    optimizer,
    generator,
    temp_loss,
    train_params,
    model_params,
    is_training=True,
    header="",
    epoch=0,
    wandb=False,
):
    if temp_loss == {}:
        temp_loss = {
            "total": [],
            "global": [],
            "local": [],
            "rank": [],
            "distance_map": [],
            "hbond_map": [],
            "polar_apolar_map": [],
            "apolar_apolar_map": [],
            "reg": [],
        }

    b_count = 0
    w_reg = train_params["w_reg"]
    scheduling = train_params["scheduling"]
    init0iter = train_params["init0iter"]

    stdout_text_list = []

    for i, (G_atm, G_res, G_high, info) in enumerate(generator):
        # Get prediction and target value
        if not G_atm:
            print("skip %s %s" % (info["pname"], info["sname"]))
            continue

        for key in info.keys():
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

        if isinstance(G_atm, tuple) or isinstance(G_res, tuple):
            print(G_atm)
            print(G_res)

        model_prediction = model(
            G_atm.to(DEVICE), G_res.to(DEVICE), G_high.to(DEVICE), info
        )
        fnat = info["fnat"]
        pred_fnat = model_prediction[-2]
        loss1, loss2, loss3, loss4, loss5, loss6, lossR = calc_losses(
            model_prediction, info, model_params
        )
        if loss1 is None:
            print("skip %s %s" % (info["pname"], info["sname"]))
            continue

        suffix = " : fnat/pred"
        for a, b in zip(fnat, pred_fnat):
            suffix += " %6.3f/%6.3f | " % (float(a), float(b))
        suffix += " %6.3f %1d" % (float(lossR), len(G_atm.batch_num_nodes()))

        if scheduling:
            sheduled_w = cosine_schedule(300, 20)
            sheduled_w[:init0iter] = 1.0
            sheduled_w = sheduled_w[epoch]
            loss = (
                train_params["w_lossG"] * loss1 * (1 - sheduled_w)
                + train_params["w_lossL"] * loss2 * (1 - sheduled_w)
                + train_params["w_lossHbond"] * loss3 * sheduled_w
                + train_params["w_lossPolApol"] * loss4 * sheduled_w
                + train_params["w_lossApolApol"] * loss5 * sheduled_w
                + train_params["w_lossDist"] * loss6 * sheduled_w
                + train_params["w_lossR"] * lossR * (1 - sheduled_w)
            )
        else:
            loss = (
                train_params["w_lossG"] * loss1
                + train_params["w_lossL"] * loss2
                + train_params["w_lossHbond"] * loss3
                + train_params["w_lossPolApol"] * loss4
                + train_params["w_lossApolApol"] * loss5
                + train_params["w_lossDist"] * loss6
                + train_params["w_lossR"] * lossR
            )

        if is_training:
            l2_reg = torch.tensor(0.0).to(DEVICE)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss = loss + w_reg * l2_reg
            if not np.isnan(loss.cpu().detach().numpy()):
                loss.backward(retain_graph=True)
            else:
                print("nan loss encountered", pred_fnat, fnat)
            temp_loss["reg"].append(l2_reg.cpu().detach().numpy())

            if train_params["clip_grad"] > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_params["clip_grad"]
                )
            optimizer.step()
            optimizer.zero_grad()

        b_count += 1
        temp_loss["total"].append(loss.cpu().detach().numpy())
        temp_loss["global"].append(loss1.cpu().detach().numpy())
        temp_loss["local"].append(loss2.cpu().detach().numpy())
        temp_loss["rank"].append(lossR.cpu().detach().numpy())
        temp_loss["distance_map"].append(loss6.cpu().detach().numpy())
        temp_loss["hbond_map"].append(loss3.cpu().detach().numpy())
        temp_loss["polar_apolar_map"].append(loss4.cpu().detach().numpy())
        temp_loss["apolar_apolar_map"].append(loss5.cpu().detach().numpy())

        if header != "":
            if scheduling:
                stdout_text = (
                    "\r%s, Batch: [%2d/%2d], loss: total %6.4f global %6.4f rank %6.4f dist %6.4f hbond %6.4f pol-apol %6.4f apol-apol %6.4f : %s\n"
                    % (
                        header,
                        b_count,
                        len(generator),
                        temp_loss["total"][-1],
                        train_params["w_lossG"]
                        * temp_loss["global"][-1]
                        * (1 - sheduled_w),
                        train_params["w_lossR"]
                        * temp_loss["rank"][-1]
                        * (1 - sheduled_w),
                        train_params["w_lossDist"]
                        * temp_loss["distance_map"][-1]
                        * sheduled_w,
                        train_params["w_lossHbond"]
                        * temp_loss["hbond_map"][-1]
                        * sheduled_w,
                        train_params["w_lossPolApol"]
                        * temp_loss["polar_apolar_map"][-1]
                        * sheduled_w,
                        train_params["w_lossApolApol"]
                        * temp_loss["apolar_apolar_map"][-1]
                        * sheduled_w,
                        suffix,
                    )
                )
            else:
                stdout_text = (
                    "\r%s, Batch: [%2d/%2d], loss: total %6.4f global %6.4f rank %6.4f dist %6.4f hbond %6.4f pol-apol %6.4f apol-apol %6.4f : %s\n"
                    % (
                        header,
                        b_count,
                        len(generator),
                        temp_loss["total"][-1],
                        train_params["w_lossG"] * temp_loss["global"][-1],
                        train_params["w_lossR"] * temp_loss["rank"][-1],
                        train_params["w_lossDist"] * temp_loss["distance_map"][-1],
                        train_params["w_lossHbond"] * temp_loss["hbond_map"][-1],
                        train_params["w_lossPolApol"]
                        * temp_loss["polar_apolar_map"][-1],
                        train_params["w_lossApolApol"]
                        * temp_loss["apolar_apolar_map"][-1],
                        suffix,
                    )
                )
            sys.stdout.write(stdout_text)
            if is_training and wandb:
                stdout_text_list.append(stdout_text + "\n")

    if is_training and wandb:
        save_scatter_plot(epoch, stdout_text_list)

    return temp_loss


def wandb_initialize(
    project_name: str, model_name: str, config: dict, resume: bool = False
) -> None:
    wandb.init(
        project=project_name,
        name=model_name,
        id=model_name,
        config=config,
        resume=resume,
    )
    wandb.run.save()


def wandb_log(loss_dict: dict, epoch: int, prefix: str = "train") -> None:
    wandb.log(
        {
            prefix + "_loss(fnat)": np.array(loss_dict["global"]).mean(),
            prefix + "_loss(rank)": np.array(loss_dict["rank"]).mean(),
            prefix + "_loss(dist)": np.array(loss_dict["distance_map"]).mean(),
            prefix + "_loss(hbond)": np.array(loss_dict["hbond_map"]).mean(),
            prefix + "_loss(pol-apol)": np.array(loss_dict["polar_apolar_map"]).mean(),
            prefix + "_loss(apol-apol)": np.array(
                loss_dict["apolar_apolar_map"]
            ).mean(),
        },
        step=epoch,
    )


def model_save(
    epoch, model, optimizer, train_loss, valid_loss, model_name, output
) -> None:
    save_path = PATH / f"models/{model_name}/{output}.pkl"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "valid_loss": valid_loss,
        },
        save_path,
    )


def main():
    local_args = set_arguments_train()
    split_path, train_params, model_params, data_params, generator_params = (
        set_each_params(local_args)
    )
    # use_wandb = local_args.wandb
    use_wandb = False

    if train_params.fix_random_seed:
        set_random_seed()

    decay = 0.99
    max_epochs = train_params.max_epochs
    model_name = train_params.model_name
    base_learning_rate = train_params.base_learning_rate

    if use_wandb:
        resume = False
        if os.path.exists(PATH / f"models/{model_name}/best.pkl"):
            resume = True
        wandb_initialize("DfAF", model_name, vars(local_args), resume)
    else:
        generator_params["num_workers"] = 0

    # Load the model
    print("load model")
    start_epoch, model, optimizer, train_loss, valid_loss = load_model(
        train_params, model_params, data_params
    )

    # Load datasets & make generators
    generators = load_dataset(
        data_params, split_path, generator_params, setsuffix=".auxil"
    )
    train_generator, valid_generator = generators

    # Training
    for epoch in range(start_epoch, max_epochs):
        model.train()
        lr = base_learning_rate * np.power(decay, epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.zero_grad()

        temp_lossQA = {}

        header = "TRAIN Epoch(%s): [%2d/%2d] QA" % (model_name, epoch, max_epochs)
        temp_lossQA = enumerate_an_epoch(
            model,
            optimizer,
            train_generator,
            temp_lossQA,
            train_params,
            model_params,
            is_training=True,
            header=header,
            epoch=epoch,
            wandb=use_wandb,
        )

        for key in [
            "total",
            "global",
            "rank",
            "local",
            "distance_map",
            "hbond_map",
            "polar_apolar_map",
            "apolar_apolar_map",
            "reg",
        ]:
            train_loss[key].append(np.array(temp_lossQA[key]).mean())

        if use_wandb:
            wandb_log(temp_lossQA, epoch, "train")

        # Validation
        header = "VALID Epoch(%s): [%2d/%2d] QA" % (model_name, epoch, max_epochs)
        with torch.no_grad():
            model.eval()
            temp_lossQA = {}
            for i in range(VALID_REPEAT):  # repeat multiple times for stable numbers
                temp_lossQA = enumerate_an_epoch(
                    model,
                    optimizer,
                    valid_generator,
                    temp_lossQA,
                    train_params,
                    model_params,
                    is_training=False,
                    header=header,
                    epoch=epoch,
                    wandb=use_wandb,
                )

            for key in [
                "total",
                "global",
                "rank",
                "local",
                "distance_map",
                "hbond_map",
                "polar_apolar_map",
                "apolar_apolar_map",
            ]:
                valid_loss[key].append(np.array(temp_lossQA[key]).mean())

            if use_wandb:
                wandb_log(temp_lossQA, epoch, "valid")

        # Save best model and current model
        if (
            epoch == 0
            or (np.min([np.mean(vl) for vl in valid_loss["global"]]))
            == valid_loss["global"][-1]
        ):
            model_save(
                epoch, model, optimizer, train_loss, valid_loss, model_name, "best"
            )

        model_save(epoch, model, optimizer, train_loss, valid_loss, model_name, "model")


if __name__ == "__main__":
    main()

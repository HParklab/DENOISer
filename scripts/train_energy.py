#!/usr/bin/env python
import sys
import os
import wandb
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from easydict import EasyDict
from collections import defaultdict

from denoiser.arguments import set_arguments_train
from denoiser.dataset import collate_energy, load_dataset_energy
from denoiser.models import EnergyNet
from denoiser.utils.train_utils import (
    filter_clash,
    count_parameters,
    set_random_seed,
    pairwise_2d,
)


PATH = Path(os.path.abspath(__file__)).parent
VALID_REPEAT = 3
DEVICE = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")


def set_each_params(local_args):
    split_path = Path(local_args.data_path)

    train_params = {
        "model_name": local_args.model_name,
        "base_learning_rate": local_args.e_lr,
        "max_epochs": 100,
        "w_loss_vdw": local_args.e_w_loss_vdw,
        "w_loss_elec": local_args.e_w_loss_elec,
        "w_loss_solv": local_args.e_w_loss_solv,
        "w_loss_rank": local_args.e_w_loss_rank,
        "w_reg": local_args.w_reg,
        "setsuffix": "v6",
        "clip_grad": 1.0,  # set < 0 if don't want
        "fix_random_seed": local_args.fix_random_seed,
    }

    model_params = {
        "drop_out": local_args.e_drop_out,
        "encoder_edge_features": 2,
        "encoder_num_layers": local_args.e_num_layers,
        "encoder_num_channels": 32,
        "input_l0_feature": local_args.e_l0_in_features,
        "energy_num_channels": 32,
        "backbone": local_args.e_backbone,
        "projection_dims": local_args.e_projection_dims,
    }

    data_params = {
        "root_dir": Path(local_args.input_features),
        "AF_plddt_path": Path(local_args.AF),
        "ball_radius": local_args.ball_radius,
        "ballmode": "all",
        "upsample": filter_clash,
        "sasa_method": "sasa",
        "randomize": 0.2,
        "edgemode": local_args.edge_mode,
        "edgek": (0, 0),
        "edgedist": local_args.edge_dist,
        "distance_feat": "std",
        "aa_as_het": True,
        "use_AF": "only_conf",
        "nsamples_per_p": 1,
        "subset_size": local_args.subset_size,
        "ros_dir": Path(local_args.energy_path),
        "atom_property": PATH / "atom_properties_f.txt",
    }

    generator_params = {
        "num_workers": 0,
        "pin_memory": True,
        "collate_fn": collate_energy,
        "batch_size": 1,
    }

    return (
        split_path,
        EasyDict(train_params),
        EasyDict(model_params),
        EasyDict(data_params),
        EasyDict(generator_params),
    )


def load_model(train_params, model_params):
    model_name = train_params.model_name
    base_learning_rate = train_params.base_learning_rate

    model = EnergyNet(
        encoder_num_channels=model_params.encoder_num_channels,
        input_l0_feature=model_params.input_l0_feature,
        encoder_num_layers=model_params.encoder_num_layers,
        energy_num_channels=model_params.energy_num_channels,
        drop_out=model_params.drop_out,
        nsqueeze=model_params.projection_dims,
        backbone=model_params.backbone,
    )

    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_learning_rate)
    print("nparams: ", count_parameters(model))

    epoch = 0
    train_loss = {
        "total": [],
        "vdw": [],
        "elec": [],
        "solv": [],
        "rank": [],
        "reg": [],
    }
    valid_loss = {
        "total": [],
        "vdw": [],
        "elec": [],
        "solv": [],
        "rank": [],
        "reg": [],
    }

    if not PATH.joinpath("models", model_name).exists():
        print("Creating a new dir at", str(PATH / "models" / model_name))
        os.makedirs(PATH / "models" / model_name)

    return epoch, model, optimizer, train_loss, valid_loss


def save_scatter_plot(true_list, pred_list, target, epoch):
    result = np.array([true_list, pred_list])
    x, y = result[0, :], result[1, :]
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=1, alpha=0.7)
    ax.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10), color="r")
    wandb.log(
        {f"scatter_plot_{target}_epoch{epoch}(x: true, y: predicted)": wandb.Image(fig)}
    )


def split_energy(info, scaling: bool = True):
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


def calc_losses(model_prediction, info):
    """Return losses for vdw, elec, solv"""
    Loss_vdw = torch.nn.SmoothL1Loss(beta=0.1)
    Loss_elec = torch.nn.SmoothL1Loss(beta=0.1)
    Loss_solv = torch.nn.SmoothL1Loss(beta=0.1)
    Loss_rank = torch.nn.MarginRankingLoss()

    pred_vdw, pred_elec, pred_solv = [t.to(DEVICE) for t in model_prediction]
    pred_sum = pred_vdw + pred_elec + pred_solv
    vdw, solv, elec = split_energy(info, scaling=True)
    true_sum = vdw + solv + elec
    loss1 = Loss_vdw(pred_vdw, vdw.float())
    loss2 = Loss_elec(pred_elec, elec.float())
    loss3 = Loss_solv(pred_solv, solv.float())
    loss4 = 0.0
    for i, (pred, true) in enumerate(
        zip([pred_vdw, pred_solv, pred_elec, pred_sum], [vdw, solv, elec, true_sum])
    ):
        pairwise_pred = pairwise_2d(pred)
        a, b = pairwise_pred[0, :], pairwise_pred[1, :]
        pairwise_true = pairwise_2d(true)
        c = torch.where(pairwise_true[0, :] > pairwise_true[1, :], 1, -1)
        if i != 3:
            loss4 += Loss_rank(a, b, c)
        else:
            loss4 += Loss_rank(a, b, c) * 2
    return loss1, loss2, loss3, loss4, (vdw, solv, elec)


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
            "vdw": [],
            "elec": [],
            "solv": [],
            "rank": [],
            "reg": [],
        }

    b_count = 0
    w_reg = train_params["w_reg"]
    temp_true, temp_pred = defaultdict(list), defaultdict(list)

    for i, (G_atm, G_res, info) in enumerate(generator):
        # Get prediction and target value
        if not G_atm:
            print("skip %s %s" % (info["pname"], info["sname"]))
            continue

        for key in info.keys():
            if isinstance(info[key], list):
                pass
            else:
                info[key] = info[key].to(DEVICE)

        if isinstance(G_atm, tuple) or isinstance(G_res, tuple):
            print(G_atm)
            print(G_res)

        model_prediction = model(G_atm.to(DEVICE))
        # vdw, solv, coulomb, hbond, total = split_energy(info)
        loss1, loss2, loss3, loss4, true_values = calc_losses(model_prediction, info)
        vdw, solv, elec = true_values

        if loss1 is None:
            print("skip %s %s" % (info["pname"], info["sname"]))
            continue

        suffix = "\n"
        for n, t, p in zip(
            ["vdw", "elec", "solv"], [vdw, elec, solv], model_prediction
        ):
            suffix += n + " true/pred "
            for a, b in zip(t, p.detach()):
                suffix += " %6.3f/%6.3f | " % (float(a), float(b))
            suffix += "\n"
        loss = (
            train_params["w_loss_vdw"] * loss1
            + train_params["w_loss_elec"] * loss2
            + train_params["w_loss_solv"] * loss3
            + train_params["w_loss_rank"] * loss4
        )

        if is_training:
            l2_reg = torch.tensor(0.0).to(DEVICE)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss = loss + w_reg * l2_reg
            if not np.isnan(loss.cpu().detach().numpy()):
                loss.backward(retain_graph=True)
            else:
                print("nan loss encountered")
            temp_loss["reg"].append(l2_reg.cpu().detach().numpy())

            if train_params["clip_grad"] > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_params["clip_grad"]
                )
            optimizer.step()
            optimizer.zero_grad()

        b_count += 1
        temp_loss["total"].append(loss.cpu().detach().numpy())
        temp_loss["vdw"].append(loss1.cpu().detach().numpy())
        temp_loss["elec"].append(loss2.cpu().detach().numpy())
        temp_loss["solv"].append(loss3.cpu().detach().numpy())
        temp_loss["rank"].append(loss4.cpu().detach().numpy())

        if header != "":
            stdout_text = (
                "\r%s, Batch: [%2d/%2d], loss: total %6.4f vdw %6.4f elec %6.4f solv %6.4f rank %6.4f: %s\n"
                % (
                    header,
                    b_count,
                    len(generator),
                    temp_loss["total"][-1],
                    train_params["w_loss_vdw"] * temp_loss["vdw"][-1],
                    train_params["w_loss_elec"] * temp_loss["elec"][-1],
                    train_params["w_loss_solv"] * temp_loss["solv"][-1],
                    train_params["w_loss_rank"] * temp_loss["rank"][-1],
                    suffix,
                )
            )
            sys.stdout.write(stdout_text)
            if is_training and wandb and epoch % 10 == 0:
                for t, true, pred in zip(
                    ["vdw", "elec", "rank"], [vdw, elec, solv], model_prediction
                ):
                    temp_true[t] += true.tolist()
                    temp_pred[t] += pred.detach().tolist()

    if is_training and wandb and epoch % 10 == 0:
        for t in temp_true.keys():
            save_scatter_plot(temp_true[t], temp_pred[t], t, epoch)

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
            prefix + "_loss(vdw)": np.array(loss_dict["vdw"]).mean(),
            prefix + "_loss(elec)": np.array(loss_dict["elec"]).mean(),
            prefix + "_loss(solv)": np.array(loss_dict["solv"]).mean(),
            prefix + "_loss(rank)": np.array(loss_dict["rank"]).mean(),
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
        train_params, model_params
    )

    # Load datasets & make generators
    generators = load_dataset_energy(
        data_params, split_path, generator_params, setsuffix=".MT_all"
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

        # end nalt
        for key in ["vdw", "elec", "solv", "rank", "reg"]:
            train_loss[key].append(np.array(temp_lossQA[key]).mean())

        # summ up total loss
        totalloss = np.array(temp_lossQA["total"]).mean()
        train_loss["total"].append(totalloss)

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
                    wandb=use_wandb,
                )

            for key in ["vdw", "elec", "solv", "rank", "reg"]:
                valid_loss[key].append(np.array(temp_lossQA[key]).mean())

            totalloss = np.array(temp_lossQA["total"]).mean()
            valid_loss["total"].append(totalloss)

            if use_wandb:
                wandb_log(temp_lossQA, epoch, "valid")

        # Save best model and current model
        if (
            epoch == 0
            or (np.min([np.mean(vl) for vl in valid_loss["total"]]))
            == valid_loss["total"][-1]
        ):
            model_save(
                epoch, model, optimizer, train_loss, valid_loss, model_name, "best"
            )

        model_save(epoch, model, optimizer, train_loss, valid_loss, model_name, "model")


if __name__ == "__main__":
    main()

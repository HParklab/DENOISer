import torch
from torch import nn
from torch.nn import functional as F
from dgl import mean_nodes, sum_nodes
from .se3_transformer import SE3Transformer, Fiber
from .egnn import EGNN


class ProjectionModule(nn.Module):
    def __init__(
        self,
        nsqueeze=15,
        l0_in_features=(35, 65 + 28 + 2),
        num_channels=(32, 32),
        drop_out=0.1,
    ):
        super().__init__()
        self.linear00_atm = nn.Linear(21, nsqueeze)
        self.linear01_atm = nn.Linear(65, nsqueeze)
        self.linear00_atm2 = nn.Linear(nsqueeze, nsqueeze)
        self.linear01_atm2 = nn.Linear(nsqueeze, nsqueeze)
        self.drop = nn.Dropout(drop_out)

        self.linear1_atm = nn.Linear(l0_in_features[1], num_channels[1])

    def forward(self, G_atm):
        # 0: is_lig; 1~33: AAtype; 34~98: atomtype; 99: charge, 100~: other features
        # atom type
        h00_atm = F.elu(self.linear00_atm(G_atm.ndata["0"][:, 1:22]))
        h00_atm = self.drop(h00_atm)
        h00_atm = self.linear00_atm2(h00_atm)

        # aa type & etc
        h01_atm = F.elu(self.linear01_atm(G_atm.ndata["0"][:, 22:87]))
        h01_atm = self.drop(h01_atm)
        h01_atm = self.linear01_atm2(h01_atm)

        proj_input = torch.cat(
            [G_atm.ndata["0"][:, :1], h00_atm, h01_atm, G_atm.ndata["0"][:, 87:]], dim=1
        )

        h_atm = self.drop(proj_input)
        h_atm = F.elu(self.linear1_atm(h_atm))
        return h_atm


class LinearModule(nn.Module):
    def __init__(self, num_channels=(64, 32, 32), drop_out=0.1, out_dims=1):
        super().__init__()
        linear_block = []
        linear_block.append(nn.Linear(num_channels[0], num_channels[1]))
        linear_block.append(nn.GELU())
        linear_block.append(nn.Dropout(drop_out))
        linear_block.append(nn.Linear(num_channels[1], num_channels[-1]))
        linear_block.append(nn.GELU())
        linear_block.append(nn.Dropout(drop_out))
        linear_block.append(nn.Linear(num_channels[-1], out_dims))
        self.linear_block = nn.ModuleList(linear_block)

    def forward(self, g):
        for idx, layer in enumerate(self.linear_block):
            if idx == 0:
                out = layer(g)
            else:
                out = layer(out)
        return out


class ResidueGraph(nn.Module):
    def __init__(
        self,
        l0_in_features=(35, 65 + 28 + 2),
        num_channels=(32, 32),
        num_layers=(2, 2, 2),
        edge_features=(2, 2, 2),
        drop_out=0.1,
        backbone="se3_transformer",
    ):
        super().__init__()
        # Linear projection layers for each
        self.linear1_res = nn.Linear(l0_in_features[0], l0_in_features[0])
        self.linear2_res = nn.Linear(l0_in_features[0], num_channels[0])
        self.drop = nn.Dropout(drop_out)
        self.backbone = backbone

        if backbone == "se3_transformer":
            fiber_in = Fiber({0: num_channels[0]})
            self.se3res = SE3Transformer(
                num_layers=num_layers[0],
                num_heads=4,
                channels_div=4,
                fiber_in=fiber_in,
                fiber_hidden=Fiber(
                    {0: num_channels[0], 1: num_channels[0], 2: num_channels[0]}
                ),
                fiber_out=Fiber({0: num_channels[0]}),
                fiber_edge=Fiber({0: edge_features[0]}),
            )
        elif backbone == "egnn":
            self.egnn = EGNN(
                in_node_nf=num_channels[0],
                hidden_nf=num_channels[0],
                out_node_nf=num_channels[0],
                n_layers=num_layers[0],
                in_edge_nf=edge_features[0],
            )
        else:
            raise NotImplementedError

    def forward(self, G_res):
        # Squeeze dim
        h_res = F.elu(self.linear1_res(G_res.ndata["0"].squeeze()))
        h_res = self.drop(h_res)
        h_res = self.linear2_res(h_res)

        if self.backbone == "se3_transformer":
            h_res = h_res.unsqueeze(2)
            node_features = {"0": h_res}
            edge_features = {"0": G_res.edata["0"][:, :, None].float()}

            h_res = self.se3res(G_res, node_features, edge_features)
            h_res = h_res["0"].squeeze(2)
        else:
            # print(G_res.edata["0"].float().shape)
            h_res, _ = self.egnn(
                h_res,
                G_res.ndata["x"].squeeze(1),
                G_res.edges(),
                G_res.edata["0"].float(),
            )
        return h_res


class AtomGraph(nn.Module):
    def __init__(
        self,
        l1_in_features,
        num_channels,
        num_layers,
        edge_features,
        backbone="se3_transformer",
    ):
        super().__init__()
        self.backbone = backbone

        if backbone == "se3_transformer":
            fiber_in = (
                Fiber({0: num_channels[1] + num_channels[0]})
                if l1_in_features[1] == 0
                else Fiber({0: num_channels[1] + num_channels[0], 1: l1_in_features[1]})
            )
            self.se3atm = SE3Transformer(
                num_layers=num_layers[1],
                num_heads=4,
                channels_div=4,
                fiber_in=fiber_in,
                fiber_hidden=Fiber(
                    {0: num_channels[1], 1: num_channels[1], 2: num_channels[1]}
                ),
                fiber_out=Fiber({0: num_channels[1]}),
                fiber_edge=Fiber({0: edge_features[1]}),
            )
        elif backbone == "egnn":
            self.egnn = EGNN(
                in_node_nf=num_channels[1] + num_channels[0],
                hidden_nf=num_channels[1],
                out_node_nf=num_channels[1],
                n_layers=num_layers[1],
                in_edge_nf=edge_features[1],
            )
        else:
            raise NotImplementedError

    def forward(self, G_atm, residue_feature, atom_feature):
        h_atm = torch.cat((residue_feature, atom_feature), axis=1)
        if self.backbone == "se3_transformer":
            node_features = {"0": h_atm[:, :, None].float()}
            edge_features = {"0": G_atm.edata["0"][:, :, None].float()}

            h_atm = self.se3atm(G_atm, node_features, edge_features)["0"].squeeze(2)
        else:
            h_atm, _ = self.egnn(
                h_atm,
                G_atm.ndata["x"].squeeze(1),
                G_atm.edges(),
                G_atm.edata["0"].float(),
            )
        return h_atm


class HighResAtomGraph(nn.Module):
    def __init__(
        self,
        l1_in_features,
        num_channels,
        num_layers,
        edge_features,
        backbone="se3_transformer",
    ):
        super().__init__()
        self.backbone = backbone

        if backbone == "se3_transformer":
            fiber_in = (
                Fiber({0: num_channels[2]})
                if l1_in_features[2] == 0
                else Fiber({0: num_channels[2], 1: l1_in_features[2]})
            )
            self.se3atm = SE3Transformer(
                num_layers=num_layers[2],
                num_heads=4,
                channels_div=4,
                fiber_in=fiber_in,
                fiber_hidden=Fiber(
                    {0: num_channels[2], 1: num_channels[2], 2: num_channels[2]}
                ),
                fiber_out=Fiber({0: num_channels[2]}),
                fiber_edge=Fiber({0: edge_features[2]}),
            )
        elif backbone == "egnn":
            self.egnn = EGNN(
                in_node_nf=num_channels[2],
                hidden_nf=num_channels[2],
                out_node_nf=num_channels[2],
                n_layers=num_layers[2],
                in_edge_nf=edge_features[2],
            )
        else:
            raise NotImplementedError

    def forward(self, G_high_atm, atom_feature):
        h_atm = atom_feature
        if self.backbone == "se3_transformer":
            node_features = {"0": h_atm[:, :, None].float()}
            edge_features = {"0": G_high_atm.edata["0"][:, :, None].float()}

            h_atm = self.se3atm(G_high_atm, node_features, edge_features)["0"].squeeze(
                2
            )
        else:
            h_atm, _ = self.egnn(
                h_atm,
                G_high_atm.ndata["x"].squeeze(1),
                G_high_atm.edges(),
                G_high_atm.edata["0"].float(),
            )
        return h_atm


class PIScoreNet(nn.Module):
    """For AF features & rank 1 tensor feature"""

    def __init__(
        self,
        num_layers=(2, 2, 2),
        l0_in_features=(35, 65 + 28 + 2),
        l1_in_features=(0, 0, 0),  # l1 not necessary
        num_degrees=2,
        num_channels=(32, 32, 32),
        inter_hidden_channels=(32, 32),
        edge_features=(2, 2, 2),  # normalized distance (+bnd, optional)
        pooling="avg",
        modeltype="simple",
        drop_out=0.1,
        dist_bins=20,
        nsqueeze=15,
        backbone="se3_transformer",
        **kwargs,
    ):
        super().__init__()

        # Build the network
        self.num_layers = num_layers
        self.l1_in_features = l1_in_features
        self.num_channels = num_channels
        self.edge_features = edge_features
        self.num_degrees = num_degrees
        self.pooling = pooling
        self.modeltype = modeltype
        self.inter_hidden_channels = inter_hidden_channels
        self.drop = nn.Dropout(drop_out)
        self.dist_bins = dist_bins

        # Residue graph
        self.residue_graph = ResidueGraph(
            l0_in_features, num_channels, num_layers, edge_features, drop_out, backbone
        )
        # pooling residue dim to atm
        self.linear_res = nn.Linear(num_channels[0] + num_channels[1], num_channels[0])

        # Atom graph
        self.projection_atm = ProjectionModule(
            nsqueeze, l0_in_features, num_channels, drop_out
        )
        self.atom_graph = AtomGraph(
            l1_in_features, num_channels, num_layers, edge_features, backbone
        )

        # High resolution atom graph
        self.highres_atom_graph = HighResAtomGraph(
            l1_in_features, num_channels, num_layers, edge_features, backbone
        )

        # Finalize
        self.h_bond_MLP = LinearModule(
            num_channels=(
                num_channels[0] * 2,
                inter_hidden_channels[0],
                inter_hidden_channels[1],
            ),
            drop_out=drop_out,
        )
        self.polar_apolar_MLP = LinearModule(
            num_channels=(
                num_channels[0] * 2,
                inter_hidden_channels[0],
                inter_hidden_channels[1],
            ),
            drop_out=drop_out,
        )
        self.apolar_apolar_MLP = LinearModule(
            num_channels=(
                num_channels[0] * 2,
                inter_hidden_channels[0],
                inter_hidden_channels[1],
            ),
            drop_out=drop_out,
        )
        self.distance_MLP = LinearModule(
            num_channels=(
                num_channels[0] * 2,
                inter_hidden_channels[0],
                inter_hidden_channels[0],
            ),
            drop_out=drop_out,
            out_dims=dist_bins,
        )

        self.PAblock = LinearModule((num_channels[0], 32, 32), drop_out)
        self.PAblock_high = LinearModule((num_channels[2], 32, 32), drop_out)
        self.final_weights_fnat = nn.Sequential(
            nn.Linear(64, 16), nn.GELU(), nn.Linear(16, 5)
        )
        self.final_weights_lddt = nn.Sequential(
            nn.Linear(64, 16), nn.GELU(), nn.Linear(16, 5)
        )

    def _mask_split(self, tensor, indices):
        unique = torch.unique(indices)
        return [tensor[indices == i] for i in unique]

    def _get_final_weights(self, complex_embeddings, ligand_masks):
        complex_context = []
        for idx, complex_embedding in enumerate(complex_embeddings):
            ligand_mask = ligand_masks[idx].sum(axis=1).type(torch.bool)
            ligand_embedding = complex_embedding[ligand_mask].mean(axis=0)
            receptor_embedding = complex_embedding[~ligand_mask].mean(axis=0)
            complex_context.append(torch.cat([ligand_embedding, receptor_embedding]))
        complex_context = torch.stack(complex_context)
        final_weights_fnat = torch.softmax(
            self.final_weights_fnat(complex_context), dim=1
        )
        final_weights_lddt = torch.softmax(
            self.final_weights_lddt(complex_context), dim=1
        )
        device = final_weights_fnat.device
        final_weights_fnat = torch.cat(
            (
                final_weights_fnat,
                torch.full((final_weights_fnat.shape[0], 1), 0.1, device=device),
            ),
            dim=1,
        )
        final_weights_lddt = torch.cat(
            (
                final_weights_lddt,
                torch.full((final_weights_lddt.shape[0], 1), 0.1, device=device),
            ),
            dim=1,
        )
        return final_weights_fnat, final_weights_lddt

    def calc_ligand_embedding(self, g, info, highres=False):
        if highres:
            idx = info["high_ligidx"]
            g_emb = self.PAblock_high(g)
        else:
            idx = info["ligidx"]
            g_emb = self.PAblock(g)
        g_emb = torch.matmul(idx.T, g_emb)
        g_emb_split = []
        lsum = 0
        for l in info["nligatms"]:
            acc = g_emb[lsum : lsum + l, :].T.squeeze(0)
            g_emb_split.append(acc)
            lsum += l
        g_emb_split = torch.stack(g_emb_split)
        return g_emb_split

    def calc_difference_matrices(self, complex_embedding, ligand_masks, info, idx):
        ligand_mask = ligand_masks[idx].sum(axis=1).type(torch.bool)
        ligand_embedding = complex_embedding[ligand_mask]
        receptor_embedding = complex_embedding[~ligand_mask]
        each_lig = ligand_embedding.unsqueeze(1).repeat(1, len(receptor_embedding), 1)
        each_rec = receptor_embedding.unsqueeze(0).repeat(len(ligand_embedding), 1, 1)
        complex_tensor = torch.cat(
            (each_lig, each_rec), dim=-1
        )  # [N_lig, N_rec, hidden_dims * 2]
        dist_complex_tensor = torch.cat(
            (
                each_lig[:, info["dist_rec_indices"][idx], :],
                each_rec[:, info["dist_rec_indices"][idx], :],
            ),
            dim=-1,
        )

        # Hydrogen bonding
        hbond_embed = self.h_bond_MLP(complex_tensor).squeeze(-1)  # [N_lig, N_rec]
        # Apolar-apolar interaction
        polar_apolar_embed = self.polar_apolar_MLP(complex_tensor).squeeze(-1)
        # Polar-apolar interaction
        apolar_apolar_embed = self.apolar_apolar_MLP(complex_tensor).squeeze(-1)
        # Distance map
        distance_embed = self.distance_MLP(dist_complex_tensor)

        return hbond_embed, polar_apolar_embed, apolar_apolar_embed, distance_embed

    def masked_difference_matrices(
        self, hbond_embed, polar_apolar_embed, apolar_apolar_embed, info, idx
    ):
        hbond_embed_masked = torch.where(
            info["hbond_masks"][idx] != 0, hbond_embed, torch.zeros_like(hbond_embed)
        )
        polar_apolar_embed_masked = torch.where(
            info["polar_apolar_masks"][idx] != 0,
            polar_apolar_embed,
            torch.zeros_like(polar_apolar_embed),
        )
        apolar_apolar_embed_masked = torch.where(
            info["apolar_apolar_masks"][idx] != 0,
            apolar_apolar_embed,
            torch.zeros_like(apolar_apolar_embed),
        )
        return hbond_embed_masked, polar_apolar_embed_masked, apolar_apolar_embed_masked

    def calc_global_lddt(
        self,
        final_weights_fnat,
        final_weights_lddt,
        hbond_embed_masked,
        polar_apolar_embed_masked,
        apolar_apolar_embed_masked,
        distance_embed,
        g_emb_split,
        g_high_emb_split,
        idx,
    ):
        hbond_embed_masked = torch.abs(hbond_embed_masked).detach()  # [N_lig, N_rec]
        polar_apolar_embed_masked = torch.abs(
            polar_apolar_embed_masked
        ).detach()  # [N_lig, N_rec]
        apolar_apolar_embed_masked = torch.abs(
            apolar_apolar_embed_masked
        ).detach()  # [N_lig, N_rec]
        distance_embed = torch.abs(
            distance_embed.detach().argmax(dim=-1) - (self.dist_bins / 2 - 0.5)
        )  # [N_lig, N_rec(in 5A)]

        hbond_embed_masked = hbond_embed_masked.sum(dim=1, keepdim=True) / (
            (hbond_embed_masked != 0).sum(dim=1, keepdim=True) + 1e-15
        )
        polar_apolar_embed_masked = (
            polar_apolar_embed_masked.sum(dim=1, keepdim=True)
            / ((polar_apolar_embed_masked != 0).sum(dim=1, keepdim=True) + 1e-15)
        ) * 1.5
        apolar_apolar_embed_masked = (
            apolar_apolar_embed_masked.sum(dim=1, keepdim=True)
            / ((apolar_apolar_embed_masked != 0).sum(dim=1, keepdim=True) + 1e-15)
        ) * 1.0
        distance_embed = (
            distance_embed.sum(dim=1, keepdim=True) / distance_embed.shape[1]
        ) / 7.5

        all_emb = torch.cat(
            [
                g_emb_split[idx].unsqueeze(1),
                hbond_embed_masked,
                polar_apolar_embed_masked,
                apolar_apolar_embed_masked,
                distance_embed,
                g_high_emb_split[idx].unsqueeze(1),
            ],
            dim=1,
        )
        lddt = 1 - all_emb.mm(final_weights_lddt[idx][:, None])
        fnat = 1 - all_emb.mm(final_weights_fnat[idx][:, None]).mean()
        return fnat, lddt

    def forward(self, G_atm, G_res, G_high, info):
        # Process G_res
        h_res = self.residue_graph(G_res)
        h_resA = torch.matmul(info["r2a"], h_res.clone())  # broadcast to atoms

        # Process G_atm
        # 1. First condense dimension by importance
        h_atm = self.projection_atm(G_atm)
        # 2. se3; feed in h_res as input to G_atm
        h_atm = self.atom_graph(G_atm, h_resA, h_atm)

        # Process G_highatm
        h_high_atm = self.projection_atm(G_high)
        h_high_atm = self.highres_atom_graph(G_high, h_high_atm)

        # Graph embedding and ligand embedding
        g = h_atm.clone()  # [N_all_atom, 32]
        g_high = h_high_atm.clone()
        g_emb_split = self.calc_ligand_embedding(g, info)
        g_high_emb_split = self.calc_ligand_embedding(g_high, info, True)

        # Split embeddings to each complex
        indices = torch.repeat_interleave(G_atm.batch_num_nodes())
        complex_embeddings = self._mask_split(g, indices)
        nligatms = info["nligatms"][0]  # Number of ligand atoms
        ligand_masks = [
            t[:, nligatms * i : nligatms * (i + 1)]
            for i, t in enumerate(self._mask_split(info["ligidx"], indices))
        ]

        # Graph pooling for weights
        final_weights_fnat, final_weights_lddt = self._get_final_weights(
            complex_embeddings, ligand_masks
        )

        fnats, lddts = [], []
        hbond_embeds, polar_apolar_embeds, apolar_apolar_embeds, distance_embeds = (
            [],
            [],
            [],
            [],
        )
        for idx, complex_embedding in enumerate(complex_embeddings):
            # Difference matrices
            hbond_embed, polar_apolar_embed, apolar_apolar_embed, distance_embed = (
                self.calc_difference_matrices(
                    complex_embedding, ligand_masks, info, idx
                )
            )
            hbond_embeds.append(hbond_embed)
            polar_apolar_embeds.append(polar_apolar_embed)
            apolar_apolar_embeds.append(apolar_apolar_embed)
            distance_embeds.append(F.log_softmax(distance_embed, dim=-1))

            # Masks for interaction types
            (
                hbond_embed_masked,
                polar_apolar_embed_masked,
                apolar_apolar_embed_masked,
            ) = self.masked_difference_matrices(
                hbond_embed, polar_apolar_embed, apolar_apolar_embed, info, idx
            )

            # Global lDDT
            fnat, lddt = self.calc_global_lddt(
                final_weights_fnat,
                final_weights_lddt,
                hbond_embed_masked,
                polar_apolar_embed_masked,
                apolar_apolar_embed_masked,
                distance_embed,
                g_emb_split,
                g_high_emb_split,
                idx,
            )
            lddts.append(lddt)
            fnats.append(fnat)

        lddts = torch.stack(lddts, dim=0).reshape(1, -1)
        lddts = torch.clamp(lddts, min=-3.0, max=3.0)
        fnats = torch.stack(fnats)
        fnats = torch.clamp(fnats, min=-3.0, max=3.0)
        return (
            hbond_embeds,
            polar_apolar_embeds,
            apolar_apolar_embeds,
            distance_embeds,
            fnats,
            lddts,
        )


class GlobalPIScoreNet(nn.Module):
    """
    Network for predicting global position and orientation of the ligand.
    It perform to classify the binary label whether target or not.
    """

    def __init__(
        self,
        num_layers=(2, 2),
        l0_in_features=(35, 65 + 28 + 2),
        l1_in_features=(0, 0),  # l1 not necessary
        num_channels=(32, 32),
        edge_features=(2, 2),  # normalized distance (+bnd, optional)
        pooling="avg",
        drop_out=0.1,
        nsqueeze=15,
        n_class=9,
        backbone="se3_transformer",
        **kwargs,
    ):
        super().__init__()

        # Build the network
        self.num_layers = num_layers
        self.l1_in_features = l1_in_features
        self.num_channels = num_channels
        self.edge_features = edge_features
        self.pooling = pooling
        self.drop = nn.Dropout(drop_out)

        ## Residue graph
        self.residue_graph = ResidueGraph(
            l0_in_features, num_channels, num_layers, edge_features, drop_out, backbone
        )

        ## Atom graph
        self.projection_atm = ProjectionModule(
            nsqueeze, l0_in_features, num_channels, drop_out
        )
        self.atom_graph = AtomGraph(
            l1_in_features, num_channels, num_layers, edge_features, backbone
        )

        # Finalize
        # self.classify = nn.Sequential(nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 1), nn.Sigmoid())
        self.classify = nn.Sequential(
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(16, n_class),
        )

    def forward(self, G_atm, G_res, info):
        # Process Gres
        h_res = self.residue_graph(G_res)
        h_resA = torch.matmul(info["r2a"], h_res.clone())  # broadcast to atoms

        # Process Gatm
        # 1. First condense dimension by importance
        h_atm = self.projection_atm(G_atm)
        # 2. se3; feed in h_res as input to G_atm
        h_atm = self.atom_graph(G_atm, h_resA, h_atm)

        with G_atm.local_scope():
            G_atm.ndata["0"] = h_atm
            if self.pooling == "avg":
                pred_targets = mean_nodes(G_atm, "0")
            elif self.pooling == "sum":
                pred_targets = sum_nodes(G_atm, "0")
            else:
                raise NotImplementedError()
            pred_targets = self.classify(pred_targets)
        return pred_targets

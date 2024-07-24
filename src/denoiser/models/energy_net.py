import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from .egnn import EGNN
from .se3_transformer import SE3Transformer, Fiber


class ProjectionModule(nn.Module):
    def __init__(
        self,
        nsqueeze=15,
        l0_in_features=(1 + 20 + 20 + 1 + 1),
        num_channels=32,
        drop_out=0.1,
    ):
        super().__init__()
        self.linear00_atm = nn.Linear(21, nsqueeze)
        self.linear01_atm = nn.Linear(65, nsqueeze)
        self.linear00_atm2 = nn.Linear(nsqueeze, nsqueeze)
        self.linear01_atm2 = nn.Linear(nsqueeze, nsqueeze)
        self.drop = nn.Dropout(drop_out)

        self.linear1_atm = nn.Linear(l0_in_features, num_channels)

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


class AtomGraph(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_layers: int,
        edge_features: int,
        backbone: str = "egnn",
    ):
        super().__init__()
        self.backbone = backbone

        if backbone == "se3_transformer":
            fiber_in = Fiber({0: num_channels})
            self.se3atm = SE3Transformer(
                num_layers=num_layers,
                num_heads=4,
                channels_div=4,
                fiber_in=fiber_in,
                fiber_hidden=Fiber({0: num_channels, 1: num_channels, 2: num_channels}),
                fiber_out=Fiber({0: num_channels}),
                fiber_edge=Fiber({0: edge_features}),
            )
        elif backbone == "egnn":
            self.egnn = EGNN(
                in_node_nf=num_channels,
                hidden_nf=num_channels,
                out_node_nf=num_channels,
                n_layers=num_layers,
                in_edge_nf=edge_features,
            )
        else:
            raise NotImplementedError

    def forward(self, G_atm, h_atm):
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


class vdWModule(nn.Module):
    def __init__(self, channel: int = 32):
        super().__init__()
        self.channel = channel
        self.depth_factor = torch.nn.parameter.Parameter(torch.zeros((2 * channel, 1)))
        self.radius_factor = torch.nn.parameter.Parameter(torch.zeros((2 * channel, 1)))
        torch.nn.init.xavier_uniform_(self.depth_factor, gain=0.1)
        torch.nn.init.xavier_uniform_(self.radius_factor, gain=0.1)
        with torch.no_grad():
            self.depth_factor += 1.0
            self.radius_factor += 1.0

    def forward(self, X, embs, w0, s0, paired_mask):
        device = X.device
        B = X.shape[0]
        V = X.unsqueeze(1) - X.unsqueeze(2)  # displacement vector
        D = (V * V + 1e-6).sum(dim=-1).sqrt()

        N = embs.shape[1]
        emb_X = embs.unsqueeze(1).repeat_interleave(N, dim=1)
        emb_XT = emb_X.transpose(1, 2)
        emb_ij = torch.cat([emb_X, emb_XT], dim=-1)

        # B x N x N
        # cap: sigma 0.4~1.2; depth 0.5~1.5
        s0_ij = (s0.unsqueeze(0) + s0.unsqueeze(1)).unsqueeze(0)
        s0_ij = s0_ij.repeat(B, 1, 1, 1)
        s = s0_ij * (
            0.8 * torch.sigmoid(torch.matmul(emb_ij, self.radius_factor).squeeze())
            + 0.4
        )

        repl = 5.0 * torch.exp(-0.3 * D * D * D) * paired_mask

        D = D - s
        attr = (
            (
                torch.exp(-1.0 * (D - 0.3) * (D - 0.3))
                + torch.exp(-3.0 * D * D)
                + torch.exp(-10.0 * (D * D))
            )
            / 3.0
            * paired_mask
        )

        w0_ij = (w0.unsqueeze(0) * w0.unsqueeze(1) + 1e-6).sqrt().unsqueeze(0)
        w0_ij = w0_ij.repeat(B, 1, 1, 1)
        w = w0_ij * (
            torch.sigmoid(torch.matmul(emb_ij, self.depth_factor)).squeeze() + 0.5
        )  # 0.5~1.5

        Etotal = -w * attr + repl.unsqueeze(0)  # per-pair decomposed
        Etotal = Etotal.sum()
        nan_mask = torch.isnan(Etotal)
        Etotal = torch.where(nan_mask, torch.tensor(1e-6).to(device), Etotal)
        return Etotal


class ElectrostaticsModule(nn.Module):
    def __init__(
        self,
        channel: int = 32,
        screening_order: int = 0,  # 0: point-charge 1:dipole-X 2: quadrupole-X
        allow_dipole: bool = True,
        allow_quadropole: bool = False,
    ):
        super().__init__()
        self.channel = channel
        self.conv = 332.07156

        initparams = torch.zeros((2 * channel + 1, 1))  # atom1 emb + atom2 emb + 1/d
        initparams[-1, :] = 1.0  # "DDD"
        self.screening_filter = nn.parameter.Parameter(initparams)
        torch.nn.init.xavier_normal_(self.screening_filter, gain=0.1)

    def forward(self, X, embs, qs, Ps, paired_mask):
        device = X.device
        B, N = X.shape[0], X.shape[1]

        V = X.unsqueeze(1) - X.unsqueeze(2)  # displacement vector
        D = (V * V + 1e-6).sum(dim=-1).sqrt()

        V = torch.nn.functional.normalize(V + 1e-6, dim=-1)
        invD = 1.0 / (D + 1e-6)

        Ps = Ps.repeat(B, 1, 1)
        OqP = torch.einsum("bijk,bik->bij", V, Ps)  # dot product;
        OPP = torch.einsum("bik,bjk->bij", Ps, Ps)  # dot product;

        # Coulombic
        qs = qs.squeeze()  # to vector
        Q12 = qs.outer(qs).repeat((B, 1, 1))

        emb_X = embs.unsqueeze(1).repeat_interleave(N, dim=1)
        emb_XT = emb_X.transpose(1, 2)
        emb_ij = torch.cat([emb_X, emb_XT, invD.unsqueeze(-1)], dim=-1)

        screening_factor = torch.matmul(emb_ij, self.screening_filter).squeeze(
            -1
        )  # B x N x N
        E_qq = Q12 * invD * screening_factor

        invD3 = invD * invD * invD
        E_qP = OqP * invD * invD3  # parameter-free yet; decays as 1/r4
        E_PP = OPP * invD3 * invD3  # parameter-free; decays as 1/r6

        # per-pair decomposed values
        Etotal = self.conv * paired_mask * (E_qq + E_qP + E_PP)
        Etotal = Etotal.sum() * 0.5
        nan_mask = torch.isnan(Etotal)
        Etotal = torch.where(nan_mask, torch.tensor(1e-6).to(device), Etotal)
        return Etotal


class SolvationModule(nn.Module):
    def __init__(self, channel: int = 32, calc_pair: bool = False):
        super().__init__()
        self.channel = channel
        self.conv = 332.07156
        self.calc_pair = calc_pair

        # initparamsR = torch.zeros(channel) + 1.0 #atom1 emb
        # initparamsE = torch.zeros(channel) + 4.0 #atom1 emb
        self.Born_factor = nn.parameter.Parameter(torch.randn(channel) * 0.1)
        self.die_factor = nn.parameter.Parameter(torch.randn(channel) * 0.1)
        with torch.no_grad():
            self.Born_factor += 1.0
            self.die_factor += 4.0

        initparams1 = (
            torch.randn((2 * channel + 1, 1)) * 0.1
        )  # atom1 emb + atom2 emb + 1/d
        initparams1[-1, :] = 1.0  # "DDD"
        self.screening_filter = nn.parameter.Parameter(initparams1)

    def forward(self, X, embs, qs, paired_mask):
        device = X.device
        B, N = X.shape[0], X.shape[1]

        V = X.unsqueeze(1) - X.unsqueeze(2)  # displacement vector
        D = (V * V + 1e-6).sum(dim=-1).sqrt()

        V = torch.nn.functional.normalize(V + 1e-6, dim=-1)

        invD = 1.0 / (D + 1e-6)

        atomic_die = torch.matmul(embs, self.die_factor) + 1e-6
        R = torch.matmul(embs, self.Born_factor) + 1.0  # cap to 1 Ang

        E_self = -(1 - 1.0 / atomic_die) * qs / (R + 1e-6)

        E_pair = torch.zeros((B, N, N), requires_grad=True).to(device)

        if self.calc_pair:
            emb_X = embs.unsqueeze(0).repeat(B, 1, 1, 1)
            emb_XT = emb_X.transpose(1, 2)
            emb_ij = torch.cat([emb_X, emb_XT, D.unsqueeze(-1)], dim=-1)
            screening_factor = torch.matmul(emb_ij, self.screening_filter).squeeze(
                -1
            )  # B x N x N
            E_pair = invD * screening_factor

        # per-pair decomposed values
        Etotal = (
            self.conv
            * (E_self.sum(dim=1) + 0.5 * E_pair.sum(dim=(1, 2))).squeeze()
            * 0.01
        )
        nan_mask = torch.isnan(Etotal)
        Etotal = torch.where(nan_mask, torch.tensor(1e-6).to(device), Etotal)
        return Etotal


class EnergyNet(nn.Module):
    def __init__(
        self,
        encoder_edge_features: int = 2,
        drop_out: float = 0.2,
        encoder_num_layers: int = 6,
        encoder_num_channels: int = 32,
        nsqueeze: int = 15,
        input_l0_feature: int = 42,
        energy_num_channels: int = 8,
        backbone: str = "egnn",
    ):
        super().__init__()

        # Atom graph
        self.projection_module = ProjectionModule(
            nsqueeze, input_l0_feature, encoder_num_channels, drop_out
        )
        self.atom_module = AtomGraph(
            encoder_num_channels, encoder_num_layers, encoder_edge_features, backbone
        )

        # Energy prediction model
        self.vdw_model = vdWModule(energy_num_channels)
        self.elec_model = ElectrostaticsModule(energy_num_channels)
        self.solv_model = SolvationModule(energy_num_channels)

    def forward(self, G_atm):
        all_chainidx = G_atm.ndata["0"][:, 0].detach().clone()
        h_atm = self.projection_module(G_atm)
        h_atm = self.atom_module(G_atm, h_atm)

        E_vdw, E_elec, E_solv = [], [], []
        with G_atm.local_scope():
            G_atm.ndata["0"] = h_atm
            num_nodes = G_atm.batch_num_nodes()
            n = 0
            for i, g in enumerate(dgl.unbatch(G_atm)):
                X = g.ndata["x"].squeeze(1)[None,]
                embs = g.ndata["0"].unsqueeze(0)
                w0, s0 = g.ndata["vdw_dep"], g.ndata["vdw_sig"]
                w0, s0 = w0.unsqueeze(0), s0.unsqueeze(0)
                qs = g.ndata["0"][:, -2].unsqueeze(0)  # Charge
                Ps = torch.zeros(X.shape[0], X.shape[1], 3).to(
                    g.device
                )  # Dipole orientation (it should be corrected)
                chainidx = all_chainidx[n : n + num_nodes[i]].unsqueeze(0)
                paired_mask = (
                    (chainidx != chainidx.transpose(0, 1)).float().unsqueeze(0)
                )
                n += num_nodes[i]

                E_vdw.append(self.vdw_model(X, embs, w0, s0, paired_mask))
                E_elec.append(self.elec_model(X, embs, qs, Ps, paired_mask))
                E_solv.append(self.solv_model(X, embs, qs, paired_mask))
        E_vdw, E_elec, E_solv = (
            torch.stack(E_vdw),
            torch.stack(E_elec),
            torch.stack(E_solv),
        )
        return E_vdw, E_elec, E_solv

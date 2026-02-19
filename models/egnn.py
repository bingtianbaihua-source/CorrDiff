import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum
from models.common import GaussianSmearing, MLP, batch_hybrid_edge_connection, NONLINEARITIES


class EnBaseLayer(nn.Module):
    def __init__(self, hidden_dim, edge_feat_dim, num_r_gaussian, update_x=True, act_fn='silu', norm=False):
        super().__init__()
        self.r_min = 0.
        self.r_max = 10.
        self.hidden_dim = hidden_dim
        self.num_r_gaussian = num_r_gaussian
        self.edge_feat_dim = edge_feat_dim
        self.update_x = update_x
        self.act_fn = act_fn
        self.norm = norm
        if num_r_gaussian > 1:
            self.distance_expansion = GaussianSmearing(self.r_min, self.r_max, num_gaussians=num_r_gaussian)
        self.edge_mlp = MLP(2 * hidden_dim + edge_feat_dim + num_r_gaussian, hidden_dim, hidden_dim,
                            num_layer=2, norm=norm, act_fn=act_fn, act_last=True)
        self.edge_inf = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        if self.update_x:
            # self.x_mlp = MLP(hidden_dim, 1, hidden_dim, num_layer=2, norm=norm, act_fn=act_fn)
            x_mlp = [nn.Linear(hidden_dim, hidden_dim), NONLINEARITIES[act_fn]]
            layer = nn.Linear(hidden_dim, 1, bias=False)
            torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
            x_mlp.append(layer)
            x_mlp.append(nn.Tanh())
            self.x_mlp = nn.Sequential(*x_mlp)

        self.node_mlp = MLP(2 * hidden_dim, hidden_dim, hidden_dim, num_layer=2, norm=norm, act_fn=act_fn)

    def forward(self, h, x, edge_index, mask_ligand, edge_attr=None, fix_x=False):
        src, dst = edge_index
        hi, hj = h[dst], h[src]
        # \phi_e in Eq(3)
        rel_x = x[dst] - x[src]
        d_sq = torch.sum(rel_x ** 2, -1, keepdim=True)
        if self.num_r_gaussian > 1:
            d_feat = self.distance_expansion(torch.sqrt(d_sq + 1e-8))
        else:
            d_feat = d_sq
        if edge_attr is not None:
            edge_feat = torch.cat([d_feat, edge_attr], -1)
        else:
            edge_feat = d_sq

        mij = self.edge_mlp(torch.cat([hi, hj, edge_feat], -1))
        eij = self.edge_inf(mij)
        mi = scatter_sum(mij * eij, dst, dim=0, dim_size=h.shape[0])

        # h update in Eq(6)
        h = h + self.node_mlp(torch.cat([mi, h], -1))
        if self.update_x and not fix_x:
            # x update in Eq(4)
            xi, xj = x[dst], x[src]
            # (xi - xj) / (\|xi - xj\| + C) to make it more stable
            delta_x = scatter_sum((xi - xj) / (torch.sqrt(d_sq + 1e-8) + 1) * self.x_mlp(mij), dst, dim=0)
            x = x + delta_x * mask_ligand[:, None]  # only ligand positions will be updated

        return h, x


class EGNN(nn.Module):
    def __init__(self, num_layers, hidden_dim, edge_feat_dim, num_r_gaussian, k=32, cutoff=10.0, cutoff_mode='knn',
                 update_x=True, act_fn='silu', norm=False):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.edge_feat_dim = edge_feat_dim
        self.num_r_gaussian = num_r_gaussian
        self.update_x = update_x
        self.act_fn = act_fn
        self.norm = norm
        self.k = k
        self.cutoff = cutoff
        self.cutoff_mode = cutoff_mode
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=num_r_gaussian)
        self.net = self._build_network()

    def _build_network(self):
        # Equivariant layers
        layers = []
        for l_idx in range(self.num_layers):
            layer = EnBaseLayer(self.hidden_dim, self.edge_feat_dim, self.num_r_gaussian,
                                update_x=self.update_x, act_fn=self.act_fn, norm=self.norm)
            layers.append(layer)
        return nn.ModuleList(layers)

    # todo: refactor
    def _connect_edge(self, x, mask_ligand, batch):
        # if self.cutoff_mode == 'radius':
        #     edge_index = radius_graph(x, r=self.r, batch=batch, flow='source_to_target')
        if self.cutoff_mode == 'knn':
            from torch_geometric.nn import knn_graph as pyg_knn_graph
            edge_index = pyg_knn_graph(x, k=self.k, batch=batch, flow='source_to_target')
        elif self.cutoff_mode == 'hybrid':
            edge_index = batch_hybrid_edge_connection(
                x, k=self.k, mask_ligand=mask_ligand, batch=batch, add_p_index=True)
        else:
            raise ValueError(f'Not supported cutoff mode: {self.cutoff_mode}')
        return edge_index

    # todo: refactor
    @staticmethod
    def _build_edge_type(edge_index, mask_ligand):
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1
        n_dst = mask_ligand[dst] == 1
        edge_type[n_src & n_dst] = 0
        edge_type[n_src & ~n_dst] = 1
        edge_type[~n_src & n_dst] = 2
        edge_type[~n_src & ~n_dst] = 3
        edge_type = F.one_hot(edge_type, num_classes=4)
        return edge_type

    def forward(self, h, x, mask_ligand, batch, return_all=False, fix_x=False):
        all_x = [x]
        all_h = [h]
        for l_idx, layer in enumerate(self.net):
            edge_index = self._connect_edge(x, mask_ligand, batch)
            edge_type = self._build_edge_type(edge_index, mask_ligand)
            h, x = layer(h, x, edge_index, mask_ligand, edge_attr=edge_type, fix_x=fix_x)
            all_x.append(x)
            all_h.append(h)
        outputs = {'x': x, 'h': h}
        if return_all:
            outputs.update({'all_x': all_x, 'all_h': all_h})
        return outputs


class EGNNGraphEncoder(nn.Module):
    """
    Lightweight graph-level encoder built on top of EGNN.

    This wrapper is intentionally minimal so downstream modules (e.g. disentangled
    VAE heads) can obtain a graph embedding via scatter pooling.
    """

    def __init__(
        self,
        *,
        node_feat_dim: int,
        hidden_dim: int,
        num_layers: int,
        edge_feat_dim: int,
        num_r_gaussian: int,
        k: int = 32,
        cutoff: float = 10.0,
        cutoff_mode: str = "knn",
        update_x: bool = False,
        act_fn: str = "silu",
        norm: bool = False,
        pool: str = "sum",
    ):
        super().__init__()
        if pool not in ("sum", "mean"):
            raise ValueError(f"pool must be 'sum' or 'mean', got: {pool}")
        self.pool = pool
        self.node_emb = nn.Linear(node_feat_dim, hidden_dim)
        self.egnn = EGNN(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            edge_feat_dim=edge_feat_dim,
            num_r_gaussian=num_r_gaussian,
            k=k,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            update_x=update_x,
            act_fn=act_fn,
            norm=norm,
        )

    def forward(self, node_feat, pos, mask_ligand, batch, *, return_node=False, fix_x=True):
        h0 = self.node_emb(node_feat)
        out = self.egnn(h0, pos, mask_ligand, batch, return_all=False, fix_x=fix_x)
        if self.pool == "sum":
            graph_h = scatter_sum(out["h"], batch, dim=0)
        else:
            graph_h = scatter_sum(out["h"], batch, dim=0) / (
                scatter_sum(torch.ones_like(batch, dtype=out["h"].dtype), batch, dim=0).clamp_min(1.0).unsqueeze(-1)
            )
        outputs = {"graph_h": graph_h, "pos": out["x"]}
        if return_node:
            outputs["node_h"] = out["h"]
        return outputs


class CorrelationMatrixModule(nn.Module):
    """
    Predict a per-sample symmetric correlation matrix C in (0, 1) with diagonal set to 1.

    Intended usage: pool (graph-level) embeddings from ligand/molecule and protein/pocket subgraphs,
    concatenate them, and map to the upper-triangular entries of C.
    """

    def __init__(
        self,
        *,
        mol_dim: int,
        pocket_dim: int,
        n_props: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        norm: bool = False,
        act_fn: str = "silu",
    ):
        super().__init__()
        if n_props <= 0:
            raise ValueError(f"n_props must be > 0, got: {n_props}")
        self.n_props = int(n_props)
        out_dim = self.n_props * (self.n_props + 1) // 2
        self.mlp = MLP(
            mol_dim + pocket_dim,
            out_dim,
            hidden_dim,
            num_layer=num_layers,
            norm=norm,
            act_fn=act_fn,
            act_last=False,
        )

    def forward(self, mol_pooled: torch.Tensor, pocket_pooled: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mol_pooled: (B, D_m)
            pocket_pooled: (B, D_p)
        Returns:
            C: (B, P, P) symmetric, in (0, 1) with diag==1
        """
        if mol_pooled.dim() != 2 or pocket_pooled.dim() != 2:
            raise ValueError("mol_pooled and pocket_pooled must be 2D tensors (B, D)")
        if mol_pooled.shape[0] != pocket_pooled.shape[0]:
            raise ValueError("mol_pooled and pocket_pooled must have same batch size")

        bsz = mol_pooled.shape[0]
        vec = torch.sigmoid(self.mlp(torch.cat([mol_pooled, pocket_pooled], dim=-1)))
        c = vec.new_zeros((bsz, self.n_props, self.n_props))
        triu = torch.triu_indices(self.n_props, self.n_props, offset=0, device=vec.device)
        c[:, triu[0], triu[1]] = vec
        c = c + c.transpose(1, 2) - torch.diag_embed(torch.diagonal(c, dim1=1, dim2=2))
        diag = torch.arange(self.n_props, device=vec.device)
        c[:, diag, diag] = 1.0
        return c

    def get_branch_mask(self, c: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Args:
            c: (P, P) or (B, P, P) correlation matrix
        Returns:
            mask: same shape as c, boolean; diagonal always True
        """
        if c.dim() not in (2, 3):
            raise ValueError(f"c must be 2D or 3D, got shape: {tuple(c.shape)}")
        mask = c >= float(threshold)
        if c.dim() == 2:
            diag = torch.arange(c.shape[0], device=c.device)
            mask[diag, diag] = True
        else:
            diag = torch.arange(c.shape[1], device=c.device)
            mask[:, diag, diag] = True
        return mask

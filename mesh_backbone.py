"""
================================================================================
GraphCast-style GNN Backbone for CFM Heat Wave Forecasting
================================================================================

Supports two modes via the `deterministic` flag:

  DETERMINISTIC (GraphCast):
    - Input: [x_t, x_tm1, x_tm2, spatial_c] (no flow state)
    - t fixed at 0.5 (embedding becomes a learned bias)
    - Predicts residual: y - x_t
    - Single forward pass at inference

  PROBABILISTIC (GenCast/CFM):
    - Input: [x_flow, x_t, x_tm1, x_tm2, spatial_c]
    - t is the flow time in [0, 1]
    - Predicts velocity field v_t
    - ODE integration at inference

The encoder-processor-decoder backbone is IDENTICAL in both modes.
Only the input channel count differs (no flow channel in deterministic mode).

CHANGES:
  - MeshProcessor: FiLM conditioning applied per message-passing round
  - Mesh2GridDecoder: FiLM conditioning before output_mlp
  - MeshFlowNet: Padding logic removed (GNN operates on 1D node arrays)
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from icosahedral_mesh import IcosahedralMesh, grid_to_flat, flat_to_grid


# =============================================================================
# BUILDING BLOCKS
# =============================================================================

class ContinuousTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        time = time * 1000.0
        device = time.device
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=device) * -(math.log(10000) / (half - 1))
        )
        args = time[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, residual=True, dropout=0.0):
        super().__init__()
        self.residual = residual and (in_dim == out_dim)
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        # LayerNorm(1) kills the output (normalizes a scalar to 0).
        # Skip it for 1-d outputs (i.e. the final prediction head).
        self.norm = nn.LayerNorm(out_dim) if out_dim > 1 else nn.Identity()

    def forward(self, x):
        out = self.net(x)
        if self.residual:
            out = out + x
        return self.norm(out)


class PeriodicLonConv2d(nn.Module):
    """2D convolution with circular padding in longitude and normal padding in latitude."""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.lon_padding = int(padding)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=(self.lon_padding, 0),
        )

    def forward(self, x):
        if self.lon_padding > 0:
            x = F.pad(x, (self.lon_padding, self.lon_padding, 0, 0), mode="circular")
        return self.conv(x)


class InteractionNetwork(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, bipartite=False, sender_dim=None, dropout=0.0):
        super().__init__()
        s_dim = sender_dim if sender_dim is not None else node_dim
        self.bipartite = bipartite
        self.edge_mlp = MLP(edge_dim + s_dim + node_dim, hidden_dim, edge_dim, num_layers=2, dropout=dropout)
        # NOTE: residual inside MLP is disabled (in_dim != out_dim) since we concat
        # node+agg. We handle the residual EXPLICITLY below instead.
        self.node_mlp = MLP(node_dim + edge_dim, hidden_dim, node_dim, num_layers=2, residual=False, dropout=dropout)

    def forward(self, sender_features, receiver_features, edge_index, edge_attr):
        src_idx, dst_idx = edge_index[0], edge_index[1]
        h_src = sender_features[src_idx]
        h_dst = receiver_features[dst_idx]

        updated_edges = self.edge_mlp(torch.cat([edge_attr, h_src, h_dst], dim=-1))

        # FIX: Force scatter_add_ to float32 to avoid bf16 accumulation errors
        # With ~42 messages per node, bf16 rounding kills gradient signal
        num_recv = receiver_features.shape[0]
        with torch.amp.autocast('cuda', enabled=False):
            updated_edges_f32 = updated_edges.float()
            agg = torch.zeros(num_recv, updated_edges_f32.shape[-1],
                              device=receiver_features.device, dtype=torch.float32)
            agg.scatter_add_(0, dst_idx.unsqueeze(-1).expand_as(updated_edges_f32), updated_edges_f32)
            agg = agg.to(receiver_features.dtype)

        updated_receiver = self.node_mlp(torch.cat([receiver_features, agg], dim=-1))

        # FIX: Explicit residual connection (GraphCast-style)
        if receiver_features.shape[-1] == updated_receiver.shape[-1]:
            updated_receiver = updated_receiver + receiver_features

        return updated_receiver, updated_edges


# =============================================================================
# ENCODER / PROCESSOR / DECODER
# =============================================================================

class Grid2MeshEncoder(nn.Module):
    def __init__(self, grid_input_dim, mesh_pos_dim, latent_dim, edge_dim,
                 hidden_dim, time_dim, cond_dim, dropout=0.0):
        super().__init__()
        self.grid_encoder = MLP(grid_input_dim, hidden_dim, latent_dim, num_layers=2, residual=False, dropout=dropout)
        self.mesh_encoder = MLP(mesh_pos_dim, hidden_dim, latent_dim, num_layers=2, residual=False, dropout=dropout)
        self.edge_encoder = MLP(edge_dim, hidden_dim, latent_dim, num_layers=1, residual=False)
        self.time_mlp = nn.Sequential(
            ContinuousTimeEmbedding(latent_dim),
            nn.Linear(latent_dim, latent_dim * 2), nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim * 2),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, latent_dim * 2), nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim * 2),
        )
        self.interaction = InteractionNetwork(
            node_dim=latent_dim, edge_dim=latent_dim, hidden_dim=hidden_dim,
            bipartite=True, sender_dim=latent_dim, dropout=dropout,
        )

    def forward(self, grid_features, mesh_pos_features, g2m_edge_index, g2m_edge_attr, t, cond):
        B = grid_features.shape[0]
        grid_h = self.grid_encoder(grid_features)
        mesh_h = self.mesh_encoder(mesh_pos_features).unsqueeze(0).expand(B, -1, -1)
        edge_h = self.edge_encoder(g2m_edge_attr)

        t_emb = self.time_mlp(t)
        c_emb = self.cond_mlp(cond)
        scale, shift = (t_emb + c_emb).chunk(2, dim=-1)
        grid_h = grid_h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        mesh_out = []
        for b in range(B):
            m_h, _ = self.interaction(grid_h[b], mesh_h[b], g2m_edge_index, edge_h)
            mesh_out.append(m_h)
        return torch.stack(mesh_out, dim=0)


class MeshProcessor(nn.Module):
    """
    Message-passing processor with per-round FiLM conditioning.

    Each round applies time/cond modulation via scale+shift (FiLM)
    so the processor can differentiate flow-matching timesteps.
    Without this, the processor is blind to t after the encoder.
    """
    def __init__(self, latent_dim, edge_dim, hidden_dim, num_rounds=8,
                 time_dim=None, dropout=0.0):
        super().__init__()
        self.edge_encoder = MLP(edge_dim, hidden_dim, latent_dim, num_layers=1, residual=False)
        self.rounds = nn.ModuleList([
            InteractionNetwork(node_dim=latent_dim, edge_dim=latent_dim, hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(num_rounds)
        ])
        # FiLM: one (scale, shift) projection per round
        if time_dim is not None:
            self.time_film = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(time_dim, latent_dim * 2), nn.GELU(),
                    nn.Linear(latent_dim * 2, latent_dim * 2),
                )
                for _ in range(num_rounds)
            ])
        else:
            self.time_film = None

    def forward(self, mesh_features, mesh_edge_index, mesh_edge_attr, t_emb=None):
        B = mesh_features.shape[0]
        edge_h = self.edge_encoder(mesh_edge_attr)
        all_h = []
        for b in range(B):
            h, e = mesh_features[b], edge_h.clone()
            for i, rnd in enumerate(self.rounds):
                e_in = e
                h, e = rnd(h, h, mesh_edge_index, e)
                e = e + e_in
                # Per-round FiLM conditioning
                if self.time_film is not None and t_emb is not None:
                    film_out = self.time_film[i](t_emb[b])  # (latent_dim * 2,)
                    scale, shift = film_out.chunk(2, dim=-1)  # each (latent_dim,)
                    h = h * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)
            all_h.append(h)
        return torch.stack(all_h, dim=0)


class Mesh2GridDecoder(nn.Module):
    """
    Decoder with FiLM conditioning before the output projection.
    Ensures the time signal stays strong as latent features project
    back to grid space.
    """
    def __init__(self, latent_dim, edge_dim, hidden_dim, output_dim,
                 time_dim=None, dropout=0.0):
        super().__init__()
        self.edge_encoder = MLP(edge_dim, hidden_dim, latent_dim, num_layers=1, residual=False)
        self.grid_proj = MLP(latent_dim, hidden_dim, latent_dim, num_layers=1, residual=True, dropout=dropout)
        self.interaction = InteractionNetwork(
            node_dim=latent_dim, edge_dim=latent_dim, hidden_dim=hidden_dim,
            bipartite=True, sender_dim=latent_dim, dropout=dropout,
        )
        # FiLM before output_mlp
        if time_dim is not None:
            self.time_film = nn.Sequential(
                nn.Linear(time_dim, latent_dim * 2), nn.GELU(),
                nn.Linear(latent_dim * 2, latent_dim * 2),
            )
        else:
            self.time_film = None
        self.output_mlp = MLP(latent_dim, hidden_dim, output_dim, num_layers=2, residual=False, dropout=dropout)

    def forward(self, mesh_features, grid_skip, m2g_edge_index, m2g_edge_attr, t_emb=None):
        B = mesh_features.shape[0]
        grid_h = self.grid_proj(grid_skip)
        edge_h = self.edge_encoder(m2g_edge_attr)
        grid_out = []
        for b in range(B):
            g_h, _ = self.interaction(mesh_features[b], grid_h[b], m2g_edge_index, edge_h)
            grid_out.append(g_h)
        out = torch.stack(grid_out, dim=0)
        # FiLM before output projection
        if self.time_film is not None and t_emb is not None:
            film_out = self.time_film(t_emb)  # (B, latent_dim*2)
            scale, shift = film_out.chunk(2, dim=-1)  # each (B, latent_dim)
            out = out * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.output_mlp(out)


# =============================================================================
# GLOBAL CONTEXT ENCODER
# =============================================================================

class GlobalContextForMesh(nn.Module):
    def __init__(self, in_channels, encoder_dim, output_dim, time_dim):
        super().__init__()
        self.time_proj = nn.Sequential(nn.Linear(time_dim, encoder_dim * 4), nn.SiLU())
        self.blocks = nn.Sequential(
            self._block(in_channels, encoder_dim),
            self._block(encoder_dim, encoder_dim * 2),
            self._block(encoder_dim * 2, encoder_dim * 4),
        )
        self.proj = nn.Conv2d(encoder_dim * 4, output_dim, kernel_size=1)
        self.norm = nn.GroupNorm(8, output_dim)

    def _block(self, in_c, out_c):
        return nn.Sequential(
            PeriodicLonConv2d(in_c, out_c, 3, padding=1),
            nn.GroupNorm(min(8, out_c), out_c), nn.SiLU(),
        )

    def forward(self, global_fields, time_emb, mesh_lat, mesh_lon):
        B = global_fields.shape[0]
        x = self.blocks(global_fields)
        x = x + self.time_proj(time_emb)[:, :, None, None]
        x = self.norm(self.proj(x))

        # ERA5 global grid: lat 90N->90S (top->bottom), lon 0E->359E (left->right).
        # Duplicate lon=0 at lon=360 so bilinear mesh-node sampling is periodic
        # across the global seam. The conv blocks above use circular lon padding.
        x = torch.cat([x, x[..., :1]], dim=-1)
        norm_lat = -(mesh_lat / 90.0)
        mesh_lon_360 = (mesh_lon + 360.0) % 360.0
        norm_lon = (mesh_lon_360 / 180.0) - 1.0

        coords = torch.stack([
            torch.from_numpy(norm_lon).float().to(x.device),
            torch.from_numpy(norm_lat).float().to(x.device),
        ], dim=-1).unsqueeze(0).unsqueeze(2).expand(B, -1, -1, -1)

        return F.grid_sample(x, coords, mode='bilinear',
                             padding_mode='border', align_corners=True
                             ).squeeze(-1).permute(0, 2, 1)


# =============================================================================
# FULL MODEL
# =============================================================================

class MeshFlowNet(nn.Module):
    """
    GraphCast-style mesh GNN backbone.

    deterministic=False (GenCast/CFM):
      Input:  [x_flow, x_t, x_tm1, x_tm2, spatial_c]
      Output: velocity field v_t
      t:      flow time in [0, 1]

    deterministic=True (GraphCast):
      Input:  [x_t, x_tm1, x_tm2, spatial_c]
      Output: direct target field, or residual over persistence when enabled
      t:      fixed at 0.5 internally (acts as learned bias)
    """
    def __init__(
        self,
        img_channels=1,
        spatial_cond_channels=15,
        condition_dim=5,
        latent_dim=256,
        hidden_dim=512,
        num_processor_rounds=8,
        mesh=None,
        image_size=(621, 1405),
        num_global_channels=5,
        global_encoder_dim=64,
        deterministic=False,
        dropout=0.0,
        input_mode="standard",
        predict_persistence_residual=False,
        multi_lead_tube=False,
        prediction_leads=(15,),
        tube_temporal_heads=4,
        tube_loss_weights=(0.80, 0.10, 0.10),
        gradient_loss_weight=0.0,
        enable_exceedance_head=False,
        exceedance_initial_logit=-2.9444389791664403,
    ):
        super().__init__()
        self.img_channels = img_channels
        self.image_size = image_size
        self.use_global = num_global_channels > 0
        self.deterministic = deterministic
        self.input_mode = input_mode
        self.predict_persistence_residual = bool(predict_persistence_residual)
        self.multi_lead_tube = bool(multi_lead_tube)
        self.prediction_leads = tuple(int(x) for x in prediction_leads)
        self.tube_num_leads = len(self.prediction_leads)
        self.center_lead = 15 if 15 in self.prediction_leads else self.prediction_leads[self.tube_num_leads // 2]
        self.tube_loss_weights = tuple(float(x) for x in tube_loss_weights)
        self.gradient_loss_weight = float(gradient_loss_weight)
        self.enable_exceedance_head = bool(enable_exceedance_head)
        self.last_exceedance_logits = None

        if deterministic:
            grid_input_dim = spatial_cond_channels
        else:
            grid_input_dim = img_channels + spatial_cond_channels

        time_dim = latent_dim * 2

        self.time_mlp = nn.Sequential(
            ContinuousTimeEmbedding(latent_dim),
            nn.Linear(latent_dim, time_dim), nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        if self.use_global:
            self.global_encoder = GlobalContextForMesh(
                in_channels=num_global_channels, encoder_dim=global_encoder_dim,
                output_dim=latent_dim, time_dim=time_dim,
            )

        self.encoder = Grid2MeshEncoder(
            grid_input_dim=grid_input_dim, mesh_pos_dim=9,
            latent_dim=latent_dim, edge_dim=3,
            hidden_dim=hidden_dim, time_dim=time_dim, cond_dim=condition_dim,
            dropout=dropout,
        )
        self.skip_proj = MLP(grid_input_dim, hidden_dim, latent_dim,
                             num_layers=2, residual=False, dropout=dropout)
        self.processor = MeshProcessor(
            latent_dim=latent_dim, edge_dim=4,
            hidden_dim=hidden_dim, num_rounds=num_processor_rounds,
            time_dim=time_dim, dropout=dropout,
        )
        self.decoder = Mesh2GridDecoder(
            latent_dim=latent_dim, edge_dim=3,
            hidden_dim=hidden_dim, output_dim=img_channels,
            time_dim=time_dim, dropout=dropout,
        )
        if self.multi_lead_tube:
            if not deterministic:
                raise ValueError("multi_lead_tube is currently implemented for deterministic mode only.")
            heads = max(1, min(int(tube_temporal_heads), int(latent_dim)))
            while latent_dim % heads != 0 and heads > 1:
                heads -= 1
            self.lead_embedding = nn.Embedding(self.tube_num_leads, latent_dim)
            self.lead_time_proj = nn.Linear(latent_dim, time_dim)
            self.tube_temporal_attn = nn.MultiheadAttention(
                embed_dim=latent_dim,
                num_heads=heads,
                dropout=dropout,
                batch_first=True,
            )
            self.tube_temporal_norm = nn.LayerNorm(latent_dim)
            self.tube_temporal_ffn = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.tube_temporal_ffn_norm = nn.LayerNorm(latent_dim)
        # Mesh-to-grid interpolation is sparse and can leave visible triangular
        # facets in raster outputs. This tiny residual CNN learns a local
        # grid-space correction after the graph decode while preserving the
        # large-scale mesh signal. The final layer starts at zero so old
        # checkpoints load safely and begin with identical behavior.
        refine_dim = max(16, min(64, latent_dim // 4))
        self.grid_refiner = nn.Sequential(
            nn.Conv2d(img_channels, refine_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(refine_dim, refine_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(refine_dim, img_channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.grid_refiner[-1].weight)
        nn.init.zeros_(self.grid_refiner[-1].bias)
        if self.enable_exceedance_head:
            self.exceedance_head = nn.Sequential(
                nn.Conv2d(img_channels, refine_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
                nn.Conv2d(refine_dim, refine_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(refine_dim, img_channels, kernel_size=3, padding=1),
            )
            nn.init.zeros_(self.exceedance_head[-1].weight)
            nn.init.constant_(self.exceedance_head[-1].bias, float(exceedance_initial_logit))
        self._mesh = mesh

    def set_mesh(self, mesh):
        self._mesh = mesh

    def forward(self, x, t, cond, global_fields=None):
        """
        Same signature for both modes.
        In deterministic mode, t is overridden to 0.5 internally.

        x: (B, C, H, W) - NO padding. Raw (621, 1405).
        """
        mesh = self._mesh
        assert mesh is not None, "Call set_mesh() before forward()"
        self.last_exceedance_logits = None

        B = x.shape[0]
        H, W = self.image_size

        if self.deterministic:
            t = torch.full((B,), 0.5, device=x.device, dtype=x.dtype)

        # Phase 3: No cropping needed. Input is already (H, W).
        grid_flat = grid_to_flat(x, mesh.grid_node_indices, (H, W))

        t_emb = self.time_mlp(t)

        if self.use_global and global_fields is not None:
            global_ctx = self.global_encoder(global_fields, t_emb, mesh.mesh_lat, mesh.mesh_lon)

        grid_skip = self.skip_proj(grid_flat)

        mesh_h = self.encoder(
            grid_flat, mesh.mesh_node_features_t,
            mesh.g2m_edge_index_t, mesh.g2m_edge_attr_t, t, cond,
        )

        if self.use_global and global_fields is not None:
            mesh_h = mesh_h + global_ctx

        # Phase 1: Pass t_emb into processor for per-round FiLM
        mesh_h = self.processor(mesh_h, mesh.mesh_edge_index_t, mesh.mesh_edge_attr_t,
                                t_emb=t_emb)

        if self.multi_lead_tube:
            lead_idx = torch.arange(self.tube_num_leads, device=x.device)
            lead_emb = self.lead_embedding(lead_idx).to(dtype=mesh_h.dtype)
            tube_h = mesh_h.unsqueeze(1) + lead_emb.view(1, self.tube_num_leads, 1, -1)

            # Temporal attention is applied across the lead dimension for each mesh node.
            tube_tokens = tube_h.permute(0, 2, 1, 3).reshape(
                B * mesh_h.shape[1], self.tube_num_leads, mesh_h.shape[-1]
            )
            attn_out, _ = self.tube_temporal_attn(tube_tokens, tube_tokens, tube_tokens, need_weights=False)
            tube_tokens = self.tube_temporal_norm(tube_tokens + attn_out)
            tube_tokens = self.tube_temporal_ffn_norm(tube_tokens + self.tube_temporal_ffn(tube_tokens))
            tube_h = tube_tokens.reshape(
                B, mesh_h.shape[1], self.tube_num_leads, mesh_h.shape[-1]
            ).permute(0, 2, 1, 3).contiguous()

            flat_mesh_h = tube_h.reshape(B * self.tube_num_leads, mesh_h.shape[1], mesh_h.shape[-1])
            flat_grid_skip = grid_skip.unsqueeze(1).expand(
                B, self.tube_num_leads, grid_skip.shape[1], grid_skip.shape[2]
            ).reshape(B * self.tube_num_leads, grid_skip.shape[1], grid_skip.shape[2])
            lead_t_emb = self.lead_time_proj(lead_emb).to(dtype=t_emb.dtype)
            flat_t_emb = (t_emb.unsqueeze(1) + lead_t_emb.unsqueeze(0)).reshape(
                B * self.tube_num_leads, t_emb.shape[-1]
            )
            grid_out = self.decoder(flat_mesh_h, flat_grid_skip,
                                    mesh.m2g_edge_index_t, mesh.m2g_edge_attr_t,
                                    t_emb=flat_t_emb)
            out = flat_to_grid(grid_out, mesh.grid_node_indices, (H, W), fill_value=0.0)
            out = out + self.grid_refiner(out)
            return out.reshape(B, self.tube_num_leads, self.img_channels, H, W).squeeze(2)

        # Phase 1: Pass t_emb into decoder for FiLM before output_mlp
        grid_out = self.decoder(mesh_h, grid_skip,
                                mesh.m2g_edge_index_t, mesh.m2g_edge_attr_t,
                                t_emb=t_emb)

        # Phase 3: Reconstruct directly to (H, W). No padding restoration.
        out = flat_to_grid(grid_out, mesh.grid_node_indices, (H, W), fill_value=0.0)
        out = out + self.grid_refiner(out)
        return out


def count_parameters(model):
    total = 0
    components = {}
    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        components[component] = components.get(component := name.split('.')[0], 0) + n
    print(f"\nParameter breakdown ({total/1e6:.1f}M total):")
    for comp, count in sorted(components.items(), key=lambda x: -x[1]):
        print(f"  {comp:25s} {count/1e6:8.1f}M ({100*count/total:5.1f}%)")
    return total

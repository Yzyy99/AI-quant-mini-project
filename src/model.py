import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoder(nn.Module):
    """Transformer encoder for variable feature count time-series windows."""

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_feedforward: int = 256,
        embed_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.seq_len = seq_len
        self.n_features = n_features
        self.d_model = d_model
        self.embed_dim = embed_dim

        self.input_projection = nn.Linear(n_features, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, embed_dim),
        )
        self.norm = nn.LayerNorm(d_model)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected x shape (B, T, F), got {tuple(x.shape)}")
        if x.shape[1] != self.seq_len or x.shape[2] != self.n_features:
            raise ValueError(
                f"expected (T, F)=({self.seq_len}, {self.n_features}), "
                f"got ({x.shape[1]}, {x.shape[2]})"
            )

        h = self.input_projection(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = h + self.pos_embedding
        h = self.encoder(h)
        return self.norm(h[:, 0])

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        cls_output = self.encode(x)
        if return_embedding:
            return cls_output
        z = self.projection_head(cls_output)
        return F.normalize(z, dim=1)


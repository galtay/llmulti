import torch
from torch import nn


class GPTClone(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_heads,
        num_layers,
        feedforward_dim,
        dropout,
        max_seq_len,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, token_ids):
        batch_size, seq_len = token_ids.shape
        positions = (
            torch.arange(0, seq_len, device=token_ids.device)
            .unsqueeze(0).expand(batch_size, seq_len)
        )
        x = self.token_embedding(token_ids) + self.position_embedding(positions)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)        
        x = self.transformer(x, mask=causal_mask, is_causal=True)
        return self.output_layer(x)
import numpy as np
import torch
from torch.nn import Module, ModuleList, Embedding, Linear, TransformerEncoderLayer, CrossEntropyLoss, LayerNorm, Dropout


class JetTransformer(Module):

    def __init__(self, hidden_dim=256, num_layers=10, num_heads=4, num_features=3, num_bins=(41, 41, 41), dropout=0.1):
        super(JetTransformer, self).__init__()
        self.num_features = num_features
        self.dropout = dropout

        # learn embedding for each bin of each feature dim
        self.feature_embeddings = ModuleList([
            Embedding(
                embedding_dim=hidden_dim,
                num_embeddings=num_bins[l]
            ) for l in range(num_features)
        ])

        # build transformer layers
        self.layers = ModuleList([
            TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
                batch_first=True,
                norm_first=True,
                dropout=dropout,
            ) for l in range(num_layers)
        ])

        # output projection and loss criterion
        self.total_bins = int(np.prod(num_bins))
        self.out_norm = LayerNorm(hidden_dim)
        self.dropout = Dropout(dropout)
        self.out_proj = Linear(hidden_dim, self.total_bins)
        self.criterion = CrossEntropyLoss()

    def forward(self, x, padding_mask):

        # construct causal mask to restrict attention to preceding elements
        seq_len = x.shape[1]
        seq_idx = torch.arange(seq_len, dtype=torch.long, device=x.device)
        causal_mask = seq_idx.view(-1, 1) < seq_idx.view(1, -1)
        padding_mask = ~padding_mask

        # project x to initial embedding
        x[x < 0] = 0
        emb = self.feature_embeddings[0](x[:, :, 0])
        for i in range(1, self.num_features):
            emb += self.feature_embeddings[i](x[:, :, i])

        # apply transformer layer
        for layer in self.layers:
            emb = layer(src=emb, src_mask=causal_mask, src_key_padding_mask=padding_mask)
        emb = self.out_norm(emb)
        emb = self.dropout(emb)

        # project final embedding to logits (not normalized with softmax)
        logits = self.out_proj(emb)
        return logits

    def loss(self, logits, true_bin):
        # ignore final logits
        logits = logits[:, :-1].reshape(-1, self.total_bins)

        # shift target bins to right
        true_bin = true_bin[:, 1:].flatten()

        loss = self.criterion(logits, true_bin)
        return loss

    def probability(self, logits, padding_mask, true_bin, perplexity=False, logarithmic=False):
        batch_size, padded_seq_len, num_bin = logits.shape
        seq_len = padding_mask.long().sum(dim=1)

        # ignore final logits
        logits = logits[:, :-1].reshape(-1, self.total_bins)
        probs = torch.softmax(logits, dim=1)

        # shift target bins to right
        true_bin = true_bin[:, 1:].flatten()

        # select probs of true bins
        sel_idx = torch.arange(probs.shape[0], dtype=torch.long, device=probs.device)
        probs = probs[sel_idx, true_bin].view(batch_size, padded_seq_len-1)
        probs[~padding_mask[:, :-1]] = 1.0

        if perplexity:
            probs = probs ** (1 / seq_len.float().view(-1, 1))

        if logarithmic:
            probs = -torch.log(probs).sum(dim=1)
        else:
            probs = probs.prod(dim=1)
        return probs

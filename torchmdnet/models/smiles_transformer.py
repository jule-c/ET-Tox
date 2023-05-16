import numpy as np
import torch
import torch.nn.functional as F
from collections import OrderedDict
from torch import nn
from torch import Tensor
from einops import rearrange
from einops import reduce
    
from torch.utils.data import Dataset
import torch
import random
from rdkit import Chem
from rdkit.Chem import AllChem

def smiles_to_int(smiles, vocab):
    vocab_size = len(vocab)
    return [vocab[char] for char in smiles]

class QuickGELU(nn.Module):
    def __init__(self):
        super(QuickGELU, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 245, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask) -> Tensor:
        # split keys, queries and values in num_heads
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = -np.inf
            energy = energy.masked_fill(mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, emb_size: int = 128, expansion: int = 4):
        super().__init__()

        #self.attention = MultiHeadAttention()
        self.attention = nn.MultiheadAttention(emb_size, num_heads=8, dropout=0.1)
        self.layernorm_1 = nn.LayerNorm(emb_size, eps=1e-5)

        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(emb_size, emb_size * expansion)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(emb_size * expansion, emb_size))
        ]))

        self.layernorm_2 = nn.LayerNorm(emb_size, eps=1e-5)

    def forward(self, x, mask):
        x_ln = self.layernorm_1(x)
        #x_ln = self.attention(x_ln, mask)
        x_ln = self.attention(x_ln, x_ln, x_ln, need_weights=False, key_padding_mask=mask)[0]
        x = x + x_ln
        x_ln = self.layernorm_2(x)
        x_mlp = self.mlp(x_ln)
        x = x + x_mlp
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, depth=8, context_length=175, hidden_size=128, output_size=128):
        super().__init__()

        self.depth = depth
        self.layers = nn.ModuleList([ResidualBlock(emb_size=hidden_size) for _ in range(depth)])

        self.vocab_size = 60
        self.context_length = context_length
        self.text_hidden_size = hidden_size
        self.text_output_size = output_size
        self.token_embedding = nn.Embedding(self.vocab_size, self.text_hidden_size, padding_idx=0)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, self.text_hidden_size))
        self.ln_final = nn.LayerNorm(self.text_hidden_size, eps=1e-5)
        self.text_projection = nn.Sequential(
            nn.Linear(self.text_hidden_size, self.text_hidden_size),
            nn.SiLU(),
            nn.Dropout(p=0.25),
            nn.Linear(self.text_hidden_size, self.text_hidden_size),
            nn.SiLU(),
            nn.Dropout(p=0.25),
            )
        self.final_projection = nn.Sequential(
            nn.Linear(self.text_hidden_size, self.text_output_size),
            nn.SiLU(),
            nn.Linear(self.text_output_size, self.text_output_size)
        )

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.text_hidden_size ** -0.5) * ((2 * self.depth) ** -0.5)
        attn_std = self.text_hidden_size ** -0.5
        fc_std = (2 * self.text_hidden_size) ** -0.5
        for block in self.layers:
            nn.init.normal_(block.attention.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attention.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def encode_text(self, text):
        #mask = (text == 0.).unsqueeze(1).unsqueeze(1).cuda()
        device = text.device
        mask = (text == 0.).to(device)
        x = self.token_embedding(text)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)
        for l in self.layers:
            x = l(x, mask)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = reduce(x, 'b n e -> b e', reduction='mean')
        #x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        x = self.text_projection(x)
        # output = torch.matmul(output[grid_pos == seq_end].view(-1, output.size(-1)), self.text_projection.to(device=token_emb.device))
        x = self.final_projection(x)
        return x

    def forward(self, smiles):
        x = self.encode_text(smiles)
        return x


import torch
import torch.nn as nn
import numpy as np

class Head(nn.Module):
    def __init__(self, n_embd: int, head_size: int, context: int, mask: bool = True, dropout_p: float = .1):
        super().__init__()
        self.q, self.k, self.v = [nn.Linear(n_embd, head_size, bias=False) for _ in range(3)]
        self.mask = mask
        if mask:
            self.masked_matrix = torch.tril(torch.ones(context, context))
        self.drop = nn.Dropout(dropout_p)

    def forward(self, x):
        T, C = x.shape[1:]
        w = self.q(x) @ self.k(x).transpose(-2, -1) / C ** .5
        if self.mask:
            w = w.masked_fill(self.masked_matrix[:T, :T] == 0, float("-inf"))
        out = nn.Softmax(-1)(w) 
        return self.drop(out) @ self.v(x)

class MultiHead(nn.Module):
    def __init__(self, n_embd: int, n_heads: int, context: int, mask: bool = True, dropout_p: float = .1):
        super().__init__()
        assert n_embd % n_heads == 0
        head_size = n_embd // n_heads
        self.multi_head = nn.ModuleList([Head(n_embd, head_size, context, mask, dropout_p) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.drop = nn.Dropout(dropout_p)
    
    def forward(self, x):
        x = torch.cat([h(x) for h in self.multi_head], axis=-1)
        return self.drop(self.proj(x))


class Block(nn.Module):
    def __init__(self, n_embd: int, context: int, n_heads: int = 6, mask: bool = True, dropout_p: float = .1):
        super().__init__()
        self.self_attention = MultiHead(n_embd, n_heads, context, mask, dropout_p)
        self.layn1 = nn.LayerNorm(n_embd)
        self.ff = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout_p))
        self.layn2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.self_attention(self.layn1(x))
        return x + self.ff(self.layn2(x))


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, context: int, n_embd: int, n_heads: int = 6, n_blocks: int = 8, dropout_p: float = .1):
        super().__init__()
        self.pos_embd = nn.Embedding(context, n_embd)
        self.out_embd = nn.Embedding(vocab_size, n_embd)
        self.pos = torch.arange(0, context)
        self.block = nn.Sequential(*[Block(n_embd, context, n_heads, dropout_p=dropout_p) for _ in range(n_blocks)])
        self.layN = nn.LayerNorm(n_embd)
        self.lin = nn.Linear(n_embd, vocab_size)
        self.eval()
    
    def forward(self, x):
        T = x.shape[1]
        x = self.out_embd(x) + self.pos_embd(self.pos[:T])
        x = self.block(x)
        return self.lin(self.layN(x))

import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import cross_entropy
from torch.distributions import Categorical
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-episodes", help="Total number of episodes", type=int, default=100)
parser.add_argument("-state_dict", help="Path to state dict", type=str, default="state_dict")
parser.add_argument("-data_path", help="Path to data", type=str, default="data.txt")
parser.add_argument("--train", help="Train the model", action=argparse.BooleanOptionalAction)
args = parser.parse_args()

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


with open(args.data_path, 'r') as f:
    txt = f.read()

unique_chars = set(txt)
vocab_size = len(unique_chars)
print(f"Vocab size: {vocab_size}")
stoi = {c: i for i, c in enumerate(unique_chars)}
itos = {i: c for i, c in enumerate(unique_chars)}
data = [stoi[c] for c in txt]
''.join([itos[i] for i in data[:100]]) ## Check

def get_batch(data, batch_size: int = 32, context: int = 32):
    start_pos = np.random.choice(len(data) - context, batch_size)
    return torch.tensor([data[s: s + context] for s in start_pos]), torch.tensor([data[s+1: s + context + 1] for s in start_pos])

CONTEXT = 32
N_EMBD = 64
BATCH_SIZE = 16
N_HEADS = 4
N_BLOCKS = 4
LEARNING_RATE = 1e-04
dropout_p = 0

model = Transformer(vocab_size, CONTEXT, N_EMBD, N_HEADS, N_BLOCKS, dropout_p)
try:
    model.load_state_dict(torch.load(args.state_dict))
except Exception as e:
    print(f"Unable to load state dict {e}")

if args.train:
    total_params = sum([p.numel() for p in model.parameters()])
    print(f"Total parameters: {total_params / 1e06}")
    optim = torch.optim.AdamW(model.parameters(), LEARNING_RATE)
    model.train()
    for e in range(args.episodes):
        x, y = get_batch(data, BATCH_SIZE, CONTEXT)
        fcst = model(x)
        loss = cross_entropy(fcst.view(BATCH_SIZE * CONTEXT, -1), y.view(-1))
        loss.backward()
        optim.step()
        optim.zero_grad()
        if e % 100 == 0:
            print(f"Episode {e} Loss {loss:.2f}")
            torch.save(model.state_dict(), args.state_dict)
    model.eval()
    
ctx = torch.tensor([[0]])
res = []
for _ in range(args.episodes):
    logits = model(ctx[:, -CONTEXT:])
    token_dist = Categorical(logits = logits[:, -1])
    new_t = token_dist.sample()[None, :]
    ctx = torch.cat([ctx, new_t], axis=1)
    res.append(new_t.item())

print(''.join([itos[c] for c in res]))

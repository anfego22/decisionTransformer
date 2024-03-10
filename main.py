import torch
from torch.nn.functional import cross_entropy
from torch.distributions import Categorical
import gym
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-episodes", help="Total number of episodes", type=int, default=100)
parser.add_argument("-state_dict", help="Path to state dict", type=str, default="state_dict")
parser.add_argument("-data_path", help="Path to data", type=str, default="data.txt")
parser.add_argument("--train", help="Train the model", action=argparse.BooleanOptionalAction)
args = parser.parse_args()

CONTEXT = 32
N_EMBD = 64
BATCH_SIZE = 16
N_HEADS = 4
N_BLOCKS = 4
LEARNING_RATE = 1e-03
dropout_p = 0

def play(env, agent):
    

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
    

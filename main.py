import torch
from torch.nn.functional import cross_entropy
from torch.distributions import Categorical
from transformer import DecisionTransformer
import numpy as np
import gym
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-episodes", help="Total number of episodes", type=int, default=100)
parser.add_argument("-state_dict", help="Path to state dict", type=str, default="state_dict")
parser.add_argument("-data_path", help="Path to data", type=str, default="data.txt")
parser.add_argument("--train", help="Train the model", action=argparse.BooleanOptionalAction)
args = parser.parse_args()

CONTEXT = 10
N_EMBD = 32
BATCH_SIZE = 16
N_HEADS = 4
N_BLOCKS = 4
LEARNING_RATE = 1e-03
ACTION_SIZE = 4
OBS_SIZE = 8
dropout_p = 0

model = DecisionTransformer(ACTION_SIZE, OBS_SIZE, N_EMBD, CONTEXT, N_BLOCKS)

def play(env, agent, context: int, target_ret: float = 100.0):
    obs, _ = env.reset()    
    done, trunc, total_reward = False, False, 0
    R, s, a, t = [[target_ret]], obs[None, :], [0], [[0]]
    while not done and not trunc:
        x = {
            'R': torch.FloatTensor(R).view(1, -1, 1),
            's': torch.FloatTensor(s).view(1, -1, obs.shape[0]),
            'a': torch.tensor(a).view(1, -1),
            't': torch.tensor(t).view(1, -1)
        }
        action = agent.act(x)
        obs, r, done, trunc, _ = env.step(action)
        total_reward += r
        R.append([R[-1][0] - r])
        s = np.concatenate([s, obs[None, :]], axis=0)
        a.append(action)
        t.append([len(R) - 1])
    rb = {'R': torch.tensor(R), 's': torch.tensor(s),
          'a': torch.tensor(a), 't': torch.tensor(t)}
    return rb, total_reward



try:
    model.load_state_dict(torch.load(args.state_dict))
except Exception as e:
    print(f"Unable to load state dict {e}")


if args.train:
    env = gym.make("LunarLander-v2")
    target_ret = 1
    total_params = sum([p.numel() for p in model.parameters()])
    print(f"Total parameters: {total_params / 1e06}")
    optim = torch.optim.AdamW(model.parameters(), LEARNING_RATE)
    rb, tr = play(env, model, CONTEXT, target_ret)
    target_ret = max(tr, target_ret)
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
else:
    env = gym.make("LunarLander-v2", render_mode="human")
    target_ret = 1
    total = 0
    for i in range(args.episodes):
        rb, tr = play(env, model, CONTEXT, target_ret)
        total += tr
        print(f"Episode returns {tr:.2f} Average return {total / (i + 1):.2f}")
    


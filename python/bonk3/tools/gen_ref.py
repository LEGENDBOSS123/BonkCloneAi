import json, numpy as np, torch
from bonk2 import config as C
from bonk2.ppo import PPOAgent
from bonk2.recurrent import RecurrentAgent
torch.manual_seed(0); np.random.seed(0)

out = {}
# feedforward
ff = PPOAgent(C.STATE_DIM, C.NUM_ACTIONS, "cpu")
obs = np.random.randn(4, C.STATE_DIM).astype(np.float32)
with torch.no_grad():
    logits = ff.actor(torch.from_numpy(obs)).numpy()
out["mlp"] = {"save": {"agent": {"actor": ff.actor.to_records()}},
              "obs": obs.tolist(), "logits": logits.tolist()}

# recurrent: 3 sequential steps so the hidden-state recursion is exercised
rc = RecurrentAgent(C.STATE_DIM, C.NUM_ACTIONS, "cpu")
h = torch.zeros(1, rc.net.hidden_size)
seq, outs = [], []
with torch.no_grad():
    for _ in range(3):
        o = torch.randn(1, C.STATE_DIM)
        lg, v, h = rc.net.step(o, h)
        seq.append(o.numpy()[0].tolist()); outs.append(lg.numpy()[0].tolist())
out["gru"] = {"save": {"agent": rc.serialize()}, "obs": seq, "logits": outs,
              "hidden": rc.net.hidden_size}
json.dump(out, open("/tmp/ref.json", "w"))
print(f"reference written: mlp {len(out['mlp']['obs'])} inputs, gru {len(seq)} steps")

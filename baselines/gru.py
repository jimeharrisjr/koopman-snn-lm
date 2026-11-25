"""Rung 3 GRU baselines. Protocol: docs/05-rung3-preregistration.md.

usage: gru.py <text8 path> <hidden> <lr0> <seed> [threads]
Appends one line to results/rung3-gru.tsv:
hidden, lr0, seed, params, valid BPC, test BPC, test SE, per-epoch monitor curve.
"""
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn

SUBSET, TRAIN_END, VALID_END, V = 10_000_000, 9_000_000, 9_500_000, 27
EPOCHS, STREAMS, T_LEN = 10, 64, 64

path, hidden, lr0, seed = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
torch.set_num_threads(int(sys.argv[5]) if len(sys.argv) > 5 else 2)
torch.manual_seed(seed)

raw = np.frombuffer(open(path, "rb").read(SUBSET), dtype=np.uint8)
chars = torch.from_numpy(np.where(raw == 32, 0, raw - 96).astype(np.int64))
assert int(chars.min()) == 0 and int(chars.max()) == 26


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(V, hidden, batch_first=True)
        self.out = nn.Linear(hidden, V)

    def forward(self, x, h):
        y, h = self.gru(nn.functional.one_hot(x, V).float(), h)
        return self.out(y), h


net = Net()
params = sum(p.numel() for p in net.parameters())
opt = torch.optim.Adam(net.parameters(), lr=lr0)


def lr_at(e):
    return lr0 if e < EPOCHS // 2 else lr0 / 4 if e < 3 * EPOCHS // 4 else lr0 / 16


@torch.no_grad()
def evaluate(start, ranges):
    """Continuous pass from a zero state at `start`; score targets in ranges."""
    net.eval()
    end = max(b for _, b in ranges)
    bits = torch.empty(end - start - 1, dtype=torch.float64)
    h = None
    for a in range(start, end - 1, 100_000):
        b = min(a + 100_000, end - 1)
        logits, h = net(chars[a:b].unsqueeze(0), h)
        lp = torch.log_softmax(logits[0].double(), -1)
        bits[a - start:b - start] = -lp.gather(1, chars[a + 1:b + 1].unsqueeze(1))[:, 0] / math.log(2)
    net.train()
    out = []
    for a, b in ranges:  # bits[i] scores target position start + i + 1
        x = bits[a - start - 1:b - start - 1]
        out.append((x.mean().item(), (x.var(unbiased=False) / len(x)).sqrt().item()))
    return out


length = TRAIN_END // STREAMS
windows = (length - 1) // T_LEN
base = torch.arange(STREAMS) * length
curve = []
t0 = time.time()
for e in range(EPOCHS):
    for g in opt.param_groups:
        g["lr"] = lr_at(e)
    h, tot = None, 0.0
    for w in range(windows):
        idx = (base + w * T_LEN).unsqueeze(1) + torch.arange(T_LEN + 1).unsqueeze(0)
        seq = chars[idx]
        logits, h = net(seq[:, :-1], h)
        h = h.detach()
        loss = nn.functional.cross_entropy(logits.reshape(-1, V), seq[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        tot += loss.item()
    mon = evaluate(8_900_000, [(TRAIN_END, VALID_END)])[0][0]
    curve.append(mon)
    print(f"[gru h{hidden} lr{lr0} s{seed}] epoch {e + 1} train {tot / windows / math.log(2):.4f} "
          f"monitor valid {mon:.4f} | {time.time() - t0:.0f} s", file=sys.stderr, flush=True)

(valid, _), (test, se) = evaluate(0, [(TRAIN_END, VALID_END), (VALID_END, SUBSET)])
print(f"[gru h{hidden} lr{lr0} s{seed}] FINAL valid {valid:.4f} test {test:.4f}", file=sys.stderr, flush=True)
with open("results/rung3-gru.tsv", "a") as f:
    f.write("\t".join(map(str, [hidden, lr0, seed, params, valid, test, se,
                                 ",".join(f"{c:.4f}" for c in curve)])) + "\n")

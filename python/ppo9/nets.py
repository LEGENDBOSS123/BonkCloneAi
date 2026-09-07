"""Net construction and the frozen-pool helpers.

One encoder class, one records format. ppo7 carried a legacy conv encoder and
a bare MLP alongside minGRU, which meant every construction site had to
dispatch on a mode flag and `policy.py` had to sniff three record layouts.
"""
from __future__ import annotations

import copy

import torch

from .config import NetConfig, Ppo9Config
from .mingru import MinGRUNet, Records


def build_actor(cfg: Ppo9Config, device: str | torch.device = "cpu") -> MinGRUNet:
    """The policy net.

    Its output vector packs BOTH heads: ``[0:num_actions]`` are joint-action
    logits and the rest are FiGAR duration logits. One net keeps the records
    format and the browser deploy simple; the split happens wherever the policy
    is read. The heads are conditionally independent given the state, so their
    log-probs and entropies simply add.
    """
    return MinGRUNet(cfg.agent_state_dim, cfg.net.hidden, cfg.actor_out_dim,
                     gru_hidden=cfg.net.gru_hidden).to(device)


def build_critic(cfg: Ppo9Config, n_atoms: int,
                 device: str | torch.device = "cpu") -> MinGRUNet:
    """The value net: one logit per outcome atom, or a single scalar."""
    out_dim = n_atoms if cfg.critic.categorical else 1
    return MinGRUNet(cfg.agent_state_dim, cfg.net.hidden, out_dim,
                     gru_hidden=cfg.net.gru_hidden).to(device)


def net_from_records(recs: Records,
                     device: str | torch.device = "cpu") -> MinGRUNet:
    """Rebuild a net from checkpoint records, architecture inferred from them.

    Pool members are rebuilt this way, so a snapshot taken at one hidden size
    still loads after the live config changes.
    """
    return MinGRUNet.from_records(recs).to(device)


def freeze(net: MinGRUNet, device: str | torch.device = "cpu") -> MinGRUNet:
    """A detached, non-trainable copy for the opponent pool.

    Pool nets stay on CPU by default even when the learner trains on GPU: they
    run many small batches, where per-dispatch overhead dominates.
    """
    frozen = copy.deepcopy(net).to(device)
    for p in frozen.parameters():
        p.requires_grad_(False)
    frozen.eval()
    return frozen


def net_config_of(net: MinGRUNet) -> NetConfig:
    """Recover the `NetConfig` a net was built with (for checkpoint checks)."""
    return NetConfig(hidden=tuple(net.hidden), gru_hidden=net.H)

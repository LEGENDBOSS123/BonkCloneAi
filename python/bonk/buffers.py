"""Preallocated numpy buffers — port of src/rl2/buffers.mjs."""

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.gamma_n = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.head = 0

    def push(self, state, action, reward, next_state, done, gamma_n):
        i = self.head
        self.states[i] = state
        self.next_states[i] = next_state
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = 1.0 if done else 0.0
        self.gamma_n[i] = gamma_n
        self.head = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n: int):
        idx = np.random.randint(0, self.size, size=n)
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.dones[idx], self.gamma_n[idx])


class ReservoirBuffer:
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.size = 0
        self.seen = 0

    def push(self, state, action):
        self.seen += 1
        if self.size < self.capacity:
            self.states[self.size] = state
            self.actions[self.size] = action
            self.size += 1
            return
        j = np.random.randint(0, self.seen)
        if j < self.capacity:
            self.states[j] = state
            self.actions[j] = action

    def sample(self, n: int):
        idx = np.random.randint(0, self.size, size=n)
        return self.states[idx], self.actions[idx]

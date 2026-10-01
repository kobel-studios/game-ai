"""Reward model + policy. Deep-TAMER style: learns a human-reward model
from (screen features + context) -> predicted reward, per action.
Per-action linear weights; SGD updates. numpy only."""
import zlib
from pathlib import Path

import numpy as np


def context_vector(text, dim):
    """Stable hashed bag-of-words -> signed vector in [-1, 1]."""
    v = np.zeros(dim, dtype=np.float32)
    for tok in text.lower().split():
        h = zlib.crc32(tok.encode("utf-8"))
        v[h % dim] += 1.0 if h & 1 else -1.0
    return v


class Brain:
    def __init__(self, n_actions, grid=(32, 32), ctx_dim=128, lr=0.05, rng=None):
        self.n_actions = n_actions
        self.grid = grid
        self.ctx_dim = ctx_dim
        self.lr = lr
        self.rng = rng or np.random.default_rng()
        self.dim = grid[0] * grid[1] + ctx_dim + 1  # +1 bias
        self.W = np.zeros((n_actions, self.dim), dtype=np.float32)

    def features(self, grid_gray, context_text):
        frame = grid_gray.astype(np.float32).ravel() / 255.0
        ctx = context_vector(context_text, self.ctx_dim)
        phi = np.empty(self.dim, dtype=np.float32)
        phi[: frame.size] = frame
        phi[frame.size : frame.size + self.ctx_dim] = ctx
        phi[-1] = 1.0
        return phi

    def predict(self, phi):
        return self.W @ phi

    def act(self, phi, epsilon, bias=None):
        scores = self.predict(phi)
        if bias is not None:
            scores = scores + bias
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions)), scores
        return int(np.argmax(scores)), scores

    def imitate(self, phi, target):
        """Behavior cloning: push the policy toward the action the human took.
        Perceptron-style update when the argmax disagrees with the human."""
        pred = int(np.argmax(self.W @ phi))
        if pred != target:
            self.W[target] += self.lr * phi
            self.W[pred] -= self.lr * phi
            return True
        return False

    def update(self, phi, action, target, weight=1.0):
        """NLMS: SGD on squared error, normalized by ||phi||^2 so updates
        stay bounded regardless of feature magnitude."""
        pred = float(self.W[action] @ phi)
        err = target - pred
        self.W[action] += self.lr * weight * err * phi / (float(phi @ phi) + 1e-6)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, W=self.W, grid=self.grid, ctx_dim=self.ctx_dim, lr=self.lr)

    @classmethod
    def load(cls, path, **kwargs):
        data = np.load(path, allow_pickle=False)
        b = cls(n_actions=kwargs["n_actions"], grid=tuple(data["grid"]),
                ctx_dim=int(data["ctx_dim"]), lr=float(data["lr"]), rng=kwargs.get("rng"))
        W = data["W"]
        n = kwargs["n_actions"]
        if W.shape[0] < n:  # new actions added since save - keep old learning
            W = np.vstack([W, np.zeros((n - W.shape[0], W.shape[1]), np.float32)])
        elif W.shape[0] > n:
            W = W[:n]
        b.W = W
        return b

"""Reinforcement Learning Policy Adapter & Episodic Memory for Project Antimatter.

Provides lightweight NumPy-based discrete policy networks, PPO trajectory rollout training,
and cosine-similarity episodic tactic retrieval for sovereign game and robotics agents.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class TrajectoryTransition:
    """Single step transition in an RL trajectory rollout."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float
    value: float


class DiscretePolicyNetwork:
    """Two-layer MLP actor-critic network implemented in pure NumPy for sub-millisecond inference."""

    def __init__(self, state_dim: int = 8, action_dim: int = 4, hidden_dim: int = 16, seed: int = 42):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        rng = np.random.default_rng(seed)

        # Shared trunk weights
        self.w1 = rng.normal(0, 0.1, size=(state_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)

        # Actor head (policy logits)
        self.w_actor = rng.normal(0, 0.1, size=(hidden_dim, action_dim))
        self.b_actor = np.zeros(action_dim)

        # Critic head (state value V(s))
        self.w_critic = rng.normal(0, 0.1, size=(hidden_dim, 1))
        self.b_critic = np.zeros(1)

    def forward(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """Compute action probability distribution and scalar state value."""
        s = np.asarray(state, dtype=np.float64).flatten()
        h = np.tanh(s @ self.w1 + self.b1)
        logits = h @ self.w_actor + self.b_actor
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)
        value = float((h @ self.w_critic + self.b_critic)[0])
        return probs, value

    def sample_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Sample discrete action from policy distribution and return (action, log_prob, value)."""
        probs, value = self.forward(state)
        action = int(np.random.choice(self.action_dim, p=probs))
        log_prob = float(np.log(max(probs[action], 1e-12)))
        return action, log_prob, value


class PPOAdapter:
    """Lightweight Proximal Policy Optimization (PPO) trainer for local reflex policies."""

    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 4,
        hidden_dim: int = 16,
        lr: float = 0.01,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
    ):
        self.policy = DiscretePolicyNetwork(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        self.lr = lr
        self.gamma = gamma
        self.clip_eps = clip_eps

    def train_step(self, transitions: List[TrajectoryTransition], epochs: int = 2) -> Dict[str, Any]:
        """Run PPO policy and value updates over a trajectory batch."""
        if not transitions:
            return {"loss": 0.0, "mean_reward": 0.0, "rollout_steps": 0}

        rewards = [t.reward for t in transitions]
        returns: List[float] = []
        g = 0.0
        for t in reversed(transitions):
            if t.done:
                g = 0.0
            g = t.reward + self.gamma * g
            returns.insert(0, g)

        returns_arr = np.asarray(returns, dtype=np.float64)
        values_arr = np.asarray([t.value for t in transitions], dtype=np.float64)
        advantages = returns_arr - values_arr
        if len(advantages) > 1 and np.std(advantages) > 1e-8:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        total_loss = 0.0
        for _ in range(epochs):
            epoch_loss = 0.0
            for idx, trans in enumerate(transitions):
                s = np.asarray(trans.state, dtype=np.float64).flatten()
                h = np.tanh(s @ self.policy.w1 + self.policy.b1)
                probs, val = self.policy.forward(s)

                new_log_prob = float(np.log(max(probs[trans.action], 1e-12)))
                ratio = np.exp(new_log_prob - trans.log_prob)
                adv = advantages[idx]

                surr1 = ratio * adv
                surr2 = np.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
                policy_loss = -min(surr1, surr2)
                value_loss = 0.5 * ((returns_arr[idx] - val) ** 2)
                epoch_loss += float(policy_loss + value_loss)

                # Gradient step on actor and critic heads
                grad_logits = -probs.copy()
                grad_logits[trans.action] += 1.0
                self.policy.w_actor += self.lr * np.outer(h, grad_logits * adv)
                self.policy.b_actor += self.lr * (grad_logits * adv)

                val_err = returns_arr[idx] - val
                self.policy.w_critic += self.lr * np.outer(h, [val_err])
                self.policy.b_critic += self.lr * np.array([val_err])

            total_loss = epoch_loss / len(transitions)

        return {
            "loss": float(total_loss),
            "mean_reward": float(np.mean(rewards)),
            "rollout_steps": len(transitions),
        }


class EpisodicMemoryStore:
    """In-memory vector store for high-reward game/robotics state-action episodes."""

    def __init__(self, max_records: int = 500):
        self.max_records = max_records
        self.records: List[Dict[str, Any]] = []

    def record_episode(
        self,
        state_vector: List[float],
        action_name: str,
        reward: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        vec = np.asarray(state_vector, dtype=np.float64)
        self.records.append(
            {
                "state": vec,
                "action": action_name,
                "reward": float(reward),
                "metadata": metadata or {},
            }
        )
        if len(self.records) > self.max_records:
            self.records.pop(0)

    def retrieve_similar_tactics(
        self,
        query_state: List[float],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Retrieve top_k highest-similarity episodic records weighted by cosine similarity."""
        if not self.records:
            return []

        q = np.asarray(query_state, dtype=np.float64)
        q_norm = np.linalg.norm(q) + 1e-12

        scored = []
        for rec in self.records:
            s = rec["state"]
            sim = float(np.dot(q, s) / (q_norm * (np.linalg.norm(s) + 1e-12)))
            scored.append(
                {
                    "action": rec["action"],
                    "reward": rec["reward"],
                    "similarity": sim,
                    "metadata": rec["metadata"],
                }
            )

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]

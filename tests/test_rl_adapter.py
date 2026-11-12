"""Tests for Reinforcement Learning Policy Adapter & Episodic Memory."""

import pytest
import numpy as np
from antimatter.context.rl_adapter import (
    DiscretePolicyNetwork,
    PPOAdapter,
    TrajectoryTransition,
    EpisodicMemoryStore,
)


def test_discrete_policy_forward():
    policy = DiscretePolicyNetwork(state_dim=8, action_dim=4, hidden_dim=16)
    state = np.array([1.0, 0.5, 0.0, 0.0, 0.8, -0.2, 0.1, 0.0])

    probs, val = policy.forward(state)
    assert len(probs) == 4
    assert np.isclose(np.sum(probs), 1.0)
    assert isinstance(val, float)

    action, log_prob, val2 = policy.sample_action(state)
    assert 0 <= action < 4
    assert log_prob <= 0.0


def test_ppo_training_step():
    adapter = PPOAdapter()
    transitions = []

    # Generate synthetic rollout
    for t in range(10):
        s = np.random.randn(8)
        act, lp, v = adapter.policy.sample_action(s)
        rew = 1.0 if act == 0 else -0.5
        s_next = s + 0.1 * np.random.randn(8)
        transitions.append(
            TrajectoryTransition(
                state=s,
                action=act,
                reward=rew,
                next_state=s_next,
                done=(t == 9),
                log_prob=lp,
                value=v,
            )
        )

    metrics = adapter.train_step(transitions, epochs=2)
    assert "loss" in metrics
    assert "mean_reward" in metrics
    assert metrics["rollout_steps"] == 10


def test_episodic_memory_retrieval():
    store = EpisodicMemoryStore(max_records=50)

    # Store a successful evasion tactic
    store.record_episode(
        state_vector=[0.5, 0.2, 0.1, 0.9, 0.9, 0.0, 1.0, 0.0],
        action_name="EVADE_RIGHT",
        reward=5.0,
        metadata={"obstacle_distance": 0.5},
    )

    # Store an irrelevant idle state
    store.record_episode(
        state_vector=[-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        action_name="IDLE",
        reward=0.1,
        metadata={"obstacle_distance": 10.0},
    )

    # Query with a state near the obstacle
    query = [0.55, 0.22, 0.08, 0.85, 0.92, 0.0, 1.0, 0.0]
    results = store.retrieve_similar_tactics(query, top_k=1)

    assert len(results) == 1
    assert results[0]["action"] == "EVADE_RIGHT"
    assert results[0]["reward"] == 5.0

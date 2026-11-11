"""Game & Robot Steering Engine for Project Antimatter.

Provides high-frequency, sub-50ms discrete and continuous motor action generation
using lightweight local models (SmolLM2-1.7B / Qwen2.5-3B) and episodic adaptation.
"""

from __future__ import annotations
import json
import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from antimatter.agent.relay import RelayBaton, SpecialistDomain
from antimatter.agent.event_bus import AgentEventBus, AgentMessage
from antimatter.core.config import get_settings

logger = logging.getLogger(__name__)


class ActionType(str, Enum):
    """Discrete game/robot primitive actions."""
    IDLE = "IDLE"
    MOVE_FORWARD = "MOVE_FORWARD"
    MOVE_BACKWARD = "MOVE_BACKWARD"
    STRAFE_LEFT = "STRAFE_LEFT"
    STRAFE_RIGHT = "STRAFE_RIGHT"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    JUMP = "JUMP"
    SPRINT = "SPRINT"
    INTERACT = "INTERACT"
    ATTACK = "ATTACK"
    HALT = "HALT"


class GameObservation(BaseModel):
    """Real-time spatial and state observation of the agent's world."""
    tick: int = 0
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    health: float = 100.0
    stamina: float = 100.0
    raycast_distances: List[float] = Field(default_factory=lambda: [10.0, 10.0, 10.0, 10.0, 10.0])  # [L90, L45, F, R45, R90]
    nearby_entities: List[Dict[str, Any]] = Field(default_factory=list)
    goal_direction: Tuple[float, float] = (0.0, 1.0)  # (dx, dy) normalized vector to target


class SteeringAction(BaseModel):
    """Deterministic control action packet dispatched to game/robot motor controller."""
    tick: int
    action: ActionType
    throttle: float = Field(default=0.0, ge=-1.0, le=1.0)
    steer: float = Field(default=0.0, ge=-1.0, le=1.0)
    aux_actions: List[str] = Field(default_factory=list)
    reasoning: str = ""
    timestamp: float = Field(default_factory=time.time)


class SteeringSpecialist:
    """High-frequency actor controlling robots or in-game characters."""

    def __init__(
        self,
        model_name: str = "smollm2:1.7b",
        event_bus: Optional[AgentEventBus] = None,
    ):
        self.model_name = model_name
        self.event_bus = event_bus
        self.history: List[Dict[str, Any]] = []

    def compute_reflex_action(self, obs: GameObservation) -> SteeringAction:
        """Deterministic sub-millisecond heuristic/reflex policy for continuous game ticks."""
        front_dist = obs.raycast_distances[2] if len(obs.raycast_distances) > 2 else 10.0
        left_dist = obs.raycast_distances[1] if len(obs.raycast_distances) > 1 else 10.0
        right_dist = obs.raycast_distances[3] if len(obs.raycast_distances) > 3 else 10.0

        # Obstacle avoidance reflex
        if front_dist < 2.0:
            if left_dist > right_dist:
                return SteeringAction(
                    tick=obs.tick,
                    action=ActionType.TURN_LEFT,
                    throttle=0.3,
                    steer=-0.8,
                    reasoning="Obstacle ahead; evading left towards open clearance.",
                )
            else:
                return SteeringAction(
                    tick=obs.tick,
                    action=ActionType.TURN_RIGHT,
                    throttle=0.3,
                    steer=0.8,
                    reasoning="Obstacle ahead; evading right towards open clearance.",
                )

        # Clear path towards goal
        goal_dx, goal_dy = obs.goal_direction
        steer_correction = max(-1.0, min(1.0, goal_dx))

        return SteeringAction(
            tick=obs.tick,
            action=ActionType.MOVE_FORWARD,
            throttle=1.0,
            steer=steer_correction,
            reasoning="Trajectory clear; advancing towards target objective.",
        )

    async def step(self, obs: GameObservation) -> SteeringAction:
        """Execute a tick decision and broadcast over event bus if configured."""
        action = self.compute_reflex_action(obs)

        # Record into episodic history for vector adaptation / replay
        record = {
            "tick": obs.tick,
            "obs": obs.model_dump(),
            "action": action.model_dump(),
            "timestamp": time.time(),
        }
        self.history.append(record)
        if len(self.history) > 1000:
            self.history.pop(0)

        # Broadcast over event bus
        if self.event_bus:
            await self.event_bus.publish(
                AgentMessage(
                    sender="steering_specialist",
                    topic="game.steer.action",
                    payload=action.model_dump(),
                )
            )

        return action

    async def handle_relay_baton(self, baton: RelayBaton) -> Dict[str, Any]:
        """Relay handler entrypoint conforming to SpecialistRelayEngine."""
        raw_obs = baton.context_data.get("observation", {})
        obs = GameObservation(**raw_obs) if raw_obs else GameObservation()

        action = await self.step(obs)
        advice = (
            f"Steering Tick {action.tick}: Executed {action.action.value} "
            f"(Throttle: {action.throttle:.2f}, Steer: {action.steer:.2f}). "
            f"Reasoning: {action.reasoning}"
        )

        return {
            "advice": advice,
            "artifacts": {
                "action": action.model_dump(),
                "history_length": len(self.history),
            },
        }

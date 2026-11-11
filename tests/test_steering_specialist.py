"""Tests for Game & Robot Steering Engine."""

import pytest
from antimatter.agent.specialists.steering import (
    SteeringSpecialist,
    GameObservation,
    ActionType,
)
from antimatter.agent.relay import (
    SpecialistRelayEngine,
    SpecialistDomain,
    RelayBaton,
)
from antimatter.brain.arbiter import VRAMArbiter
from antimatter.agent.event_bus import AgentEventBus, AgentMessage


@pytest.mark.asyncio
async def test_steering_obstacle_avoidance():
    specialist = SteeringSpecialist()

    # Observation with obstacle in front (0.8m) and more clearance on the right
    obs = GameObservation(
        tick=1,
        raycast_distances=[10.0, 1.5, 0.8, 8.0, 10.0],  # Right has 8.0m clearance
        goal_direction=(0.0, 1.0),
    )

    action = await specialist.step(obs)

    assert action.action == ActionType.TURN_RIGHT
    assert action.steer > 0.5
    assert "evading right" in action.reasoning


@pytest.mark.asyncio
async def test_steering_clear_path():
    specialist = SteeringSpecialist()

    # Observation with clear trajectory towards goal slightly to the left
    obs = GameObservation(
        tick=2,
        raycast_distances=[10.0, 10.0, 10.0, 10.0, 10.0],
        goal_direction=(-0.25, 0.95),
    )

    action = await specialist.step(obs)

    assert action.action == ActionType.MOVE_FORWARD
    assert action.throttle == 1.0
    assert action.steer == -0.25


@pytest.mark.asyncio
async def test_steering_relay_and_event_bus():
    arbiter = VRAMArbiter(total_vram_mb=8192, vram_headroom_mb=1024)
    event_bus = AgentEventBus()
    relay = SpecialistRelayEngine(arbiter=arbiter, event_bus=event_bus)

    received_actions = []

    async def on_steer(msg: AgentMessage):
        received_actions.append(msg.payload)

    event_bus.subscribe("game.steer.action", on_steer)

    steering = SteeringSpecialist(event_bus=event_bus)
    relay.register_specialist(SpecialistDomain.STEERING, steering.handle_relay_baton)

    baton = RelayBaton(
        baton_id="baton-steer-50",
        caller_id="game_director",
        domain=SpecialistDomain.STEERING,
        prompt="Steer character around obstacle",
        context_data={
            "observation": {
                "tick": 50,
                "raycast_distances": [10.0, 0.5, 0.5, 10.0, 10.0],
                "goal_direction": [0.0, 1.0],
            }
        },
    )

    result = await relay.consult(baton)

    assert result.status == "success"
    assert "Steering Tick 50:" in result.advice
    assert result.artifacts["action"]["action"] in ["TURN_RIGHT", "TURN_LEFT"]
    assert len(received_actions) == 1
    assert arbiter.allocated_vram_mb == 0

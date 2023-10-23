"""Registers the internal gym envs then loads the env plugins for module using the entry point."""
from typing import Any

from registration import (
    load_plugin_envs,
    make,
    pprint_registry,
    register,
    registry,
    spec,
)


register(
    id="Sabre-v0",
    entry_point="gym.sabre.sabre:Sabre",
    max_episode_steps=200,
    reward_threshold=195.0,
)
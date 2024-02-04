
from sabreEnv.gymSabre.gymsabre import GymSabreEnv
from sabre.sabreV9 import Sabre



from gymnasium.envs.registration import (
    load_plugin_envs,
    make,
    pprint_registry,
    register,
    spec,
)

# Sabre Envi
# ----------------------------------------

register(
    id="gymsabre-v0",
    entry_point="sabreEnv.gymSabre.gymsabre:GymSabreEnv",
)

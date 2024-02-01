
from sabreEnv.gymSabre.gymsabre import GymSabreEnv
from sabre.sabreV8 import Sabre
from utils.scenarioPPO import Scenarios
from sabreEnv.wrappers.fooWrapper import SabreActionWrapper



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

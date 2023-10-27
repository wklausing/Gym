
from sabreEnv.gymSabre.gymsabre import GymSabreEnv
from sabreEnv.sabre.fooBarSabre import FooSabre
from sabreEnv.utils.fooBarUtils import FooUtils
from sabreEnv.wrappers.fooWrapper import SabreActionWrapper


from gymnasium.envs.registration import (
    load_plugin_envs,
    make,
    pprint_registry,
    register,
    registry,
    spec,
)

# Sabre Envi
# ----------------------------------------

register(
    id="gymsabre-v0",
    entry_point="sabreEnv.gymSabre.gymsabre:GymSabreEnv",
)

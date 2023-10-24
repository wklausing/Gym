from gymnasium.envs.registration import register

register(
     id="sabre/Sabre-v0",
     entry_point="sabre.env:Sabre",
)
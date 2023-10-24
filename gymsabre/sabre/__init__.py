from gymnasium.envs.registration import register

register(
     id="Sabre-v0",
     entry_point="env.sabre:Sabre",
     max_episode_steps=300,
)
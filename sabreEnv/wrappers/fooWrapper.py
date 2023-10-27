import gymnasium as gym

# class FooWrapper:
#     #print("Foo from sabreEnv.utils")
#     pass


class SabreActionWrapper(gym.ActionWrapper):
    def action(self, action):

        print("Action Wrapper")
        # Modify the action here
        return 1
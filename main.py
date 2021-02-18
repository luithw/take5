from take5.envs.take5_env import Take5Env

sides = 5

env = Take5Env({"sides": sides})
observation = env.reset()
env.render()
for i in range (10):
  action_dict = {}
  for p in range(sides):
    action_dict["p_%i" %p] = i
  observation, reward, done, info = env.step(action_dict)
  env.render()

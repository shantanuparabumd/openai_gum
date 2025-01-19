import gym

# env_id = "AlohaEnv-v0"
# if env_id in gym.envs.registry:
#     print(f"{env_id} is registered!")
# else:
#     print(f"{env_id} is not registered.")
print("Hello")
for id in gym.envs.registry:
    print(f"{id} Present")
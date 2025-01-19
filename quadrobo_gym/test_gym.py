from gym.envs.registration import registry

print("All registered environments:")
for env_id, spec in registry.items():
    print(f"{env_id}: {spec.entry_point}")

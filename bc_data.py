import numpy as np
from oracle import Oracle
from env import AddPad
from bc import BehaviorCloner

def collect_bc_dataset(env, oracle_policy, n_episodes, filename):
    x = []
    y = []

    for _ in range(n_episodes):
        obs = env.reset()
        done = False

        while not done:
            obs_flattened = BehaviorCloner.flatten_obs(obs)
            action = oracle_policy(obs)
            obs, reward, done, info = env.step(action)

            x.append(obs_flattened)
            y.append(action)

    np.savez(f"{filename}.npz", X=x, y=y)
    print(len(x))

if __name__ == "__main__":
    oracle = Oracle()

    env = AddPad(max_digits=1)
    collect_bc_dataset(env, oracle.act, 10000, "data-1d")

    env = AddPad(max_digits=2)
    collect_bc_dataset(env, oracle.act, 40000, "data-2d")

    env = AddPad(max_digits=3)
    collect_bc_dataset(env, oracle.act, 40000, "data-3d")
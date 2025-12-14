import numpy as np
from oracle import Oracle
from env import AddPad

def collect_bc_dataset(env, oracle_policy, n_episodes, filename):
    x = []
    y = []

    for _ in range(n_episodes):
        obs = env.reset()
        done = False

        while not done:
            obs_flattened = flatten_obs(obs)
            action = oracle_policy(obs)
            obs, reward, done, info = env.step(action)

            x.append(obs_flattened)
            y.append(action)

    np.savez(f"{filename}.npz", X=x, y=y)
    print(len(x))

def flatten_obs(obs):
    def digit(n, place):
        return (n // place) % 10
    
    a = obs["A"]
    b = obs["B"]
    cursor = obs["cursor"]
    pad = obs["pad"]

    return np.asarray([
        digit(a, 100),
        digit(a, 10),
        digit(a, 1),
        digit(b, 100),
        digit(b, 10),
        digit(b, 1),
        1 if cursor == 1 else 0,
        1 if cursor == 2 else 0,
        1 if cursor == 3 else 0,
        1 if cursor == 4 else 0,
        1 if cursor == 5 else 0,
        1 if cursor == 6 else 0,
        1 if cursor == 7 else 0,
        pad[0],
        pad[1],
        pad[2],
        pad[3],
        pad[4],
        pad[5],
        pad[6]
    ], dtype=np.float32)

if __name__ == "__main__":
    oracle = Oracle()

    env = AddPad(max_digits=1)
    collect_bc_dataset(env, oracle.act, 10000, "data-1d")

    env = AddPad(max_digits=2)
    collect_bc_dataset(env, oracle.act, 40000, "data-2d")

    env = AddPad(max_digits=3)
    collect_bc_dataset(env, oracle.act, 40000, "data-3d")
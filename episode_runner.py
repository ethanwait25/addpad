from env import AddPad
from random import randint

def random_policy(obs):
    return randint(0, 13)

def run_episode(env, policy, max_steps=100):
    obs = env.reset()
    total_reward = 0.0
    done = False
    steps = 0

    while not done and steps < max_steps:
        action = policy(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

        print(f"Action: {action}, Reward: {reward:.2f}, Cursor: {obs['cursor']}")

    return {
        "return": total_reward,
        "steps": steps,
        "correct": env.is_correct(),
        "A": env.state.A,
        "B": env.state.B,
        "target": env.state.target,
        "result": env.get_current_answer(),
        "pad": env.state.pad,
        "info": info,
    }

if __name__ == "__main__":
    env = AddPad()
    result = run_episode(env, random_policy)
    print(result)
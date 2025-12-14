from env import AddPad
from random_agent import RandomAgent
from oracle import Oracle
from bc import BehaviorCloner, MLPModel, BCConfig

def run_episode(env, policy, verbose=False):
    obs = env.reset()
    total_reward = 0.0
    illegal_steps = 0
    timeout = False
    done = False
    steps = 0

    while not done:
        action = policy(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

        if info.get("illegal"):
            illegal_steps += 1

        if info.get("max_steps"):
            timeout = True

        if verbose:
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
        "illegal_steps": illegal_steps,
        "timeout": timeout
    }

def evaluate(env, policy, n_episodes=200):
    success = 0
    timeouts = 0
    illegal_steps = 0
    total_steps = 0
    total_return = 0.0

    for _ in range(n_episodes):
        ep = run_episode(env, policy, verbose=False)

        total_return += ep["return"]
        total_steps += ep["steps"]
        success += int(ep["correct"])
        timeouts += int(ep["timeout"])
        illegal_steps += ep["illegal_steps"]

    return {
        "episodes": n_episodes,
        "success_rate": success / n_episodes,
        "timeout_rate": timeouts / n_episodes,
        "illegal_rate": illegal_steps / n_episodes,
        "avg_steps": total_steps / n_episodes,
        "avg_return": total_return / n_episodes
    }


if __name__ == "__main__":
    env = AddPad()
    random = RandomAgent()
    oracle = Oracle()
    bc = BehaviorCloner.load("bc_3d.pt", 20)

    print(f"Random: {evaluate(env, random.act)}")
    print(f"Oracle: {evaluate(env, oracle.act)}")
    print(f"Behavior Cloner: {evaluate(env, bc.act)}")
    
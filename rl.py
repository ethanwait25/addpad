from itertools import chain
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader

from env import AddPad, EMPTY
from agent import Agent

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    # input: current state
    # output: actions logits
    def forward(self, x):
        return self.net(x)
  
class ValueNetwork(nn.Module):
    def __init__(self, state_size, hidden_size = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    # input: current state
    # output: estimated value
    def forward(self, x):
        return self.net(x)
    
class RLDataset(Dataset):
    def __init__(self, states, actions, old_logps, returns, advantages, cursors):
        assert states.ndim == 2, states.shape
        assert actions.ndim == 1, actions.shape
        assert old_logps.ndim == 1, old_logps.shape
        assert returns.ndim == 1, returns.shape

        self.states = torch.tensor(states, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.int64)
        self.old_logps = torch.tensor(old_logps, dtype=torch.float32)
        self.returns = torch.tensor(returns, dtype=torch.float32)
        self.advantages = torch.tensor(advantages, dtype=torch.float32)
        self.cursors = torch.tensor(cursors, dtype=torch.int64)

    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, index):
        return self.states[index], self.actions[index], self.old_logps[index], self.returns[index], self.advantages[index], self.cursors[index]

@dataclass
class PPOConfig:
    lr: float = 1e-3
    epochs: int = 1000
    env_episodes: int = 300
    gamma: float = 0.95
    batch_size: int = 256
    epsilon: float = 0.2
    policy_epochs: int = 8
    hidden_size: int = 128
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class RLAgent(Agent):
    def __init__(self, state_size = 27, action_size = 14, config = None):
        self.cfg = config or PPOConfig()
        self.device = torch.device(self.cfg.device)

        self.state_size = state_size
        self.action_size = action_size

        self.policy_net = PolicyNetwork(self.state_size, self.action_size, self.cfg.hidden_size).to(self.device)
        self.value_net  = ValueNetwork(self.state_size, self.cfg.hidden_size).to(self.device)

        self.optimizer = torch.optim.Adam(
            chain(self.policy_net.parameters(), self.value_net.parameters()),
            lr=self.cfg.lr
        )
        self.mask_by_cursor = self.make_legal_action_masks()

    @torch.no_grad()
    def act(self, obs, greedy = True):
        obs = self.flatten_obs(obs)
        state = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        logits = self.policy_net(state)

        mask = self.mask_by_cursor[obs["cursor"]]
        logits = torch.where(mask == 0.0, logits, torch.tensor(-1e9, device=self.device))

        if greedy:
            return int(torch.argmax(logits, dim=1).item())
        
        dist = Categorical(logits=logits)
        a = dist.sample()
        return int(a.item())
    
    def compute_gae(self, rewards, dones, values, last_value, gamma=0.99, lam=0.95):
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_nonterminal = 1.0 - dones[t]
            next_value = last_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
            last_gae = delta + gamma * lam * next_nonterminal * last_gae
            adv[t] = last_gae
        return adv, adv + np.array(values, dtype=np.float32)
    
    def get_action_ppo(self, obs):
        state_np = self.flatten_obs(obs)
        state = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
        
        logits = self.policy_net(state).squeeze(0)
        mask = self.mask_by_cursor[obs["cursor"]]
        logits = torch.where(mask == 0.0, logits, torch.tensor(-1e9, device=self.device))

        dist = Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)

        return int(a.item()), float(logp.item())
    
    def collect_epoch(self, env):
        states = []
        actions = []
        old_logps = []
        returns = []
        advantages = []
        cursors = []

        stats = {
            "episodes": 0,
            "success": 0,
            "timeouts": 0,
            "avg_ep_return": 0.0,
            "avg_ep_steps": 0.0,
            "illegal_moves": 0
        }

        for _ in range(self.cfg.env_episodes):
            obs = env.reset()
            done = False
            ep_states = []
            ep_actions = []
            ep_logps = []
            ep_rewards = []
            ep_dones = []
            ep_values = []
            ep_cursors = []
            ep_illegal = 0
            ep_return = 0.0

            while not done:
                state = self.flatten_obs(obs)
                cursor = obs["cursor"]

                with torch.no_grad():
                    st = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                    val = self.value_net(st).squeeze(1).item()

                action, logp = self.get_action_ppo(obs)
                obs, reward, done, info = env.step(action)

                ep_states.append(state)
                ep_actions.append(action)
                ep_logps.append(logp)
                ep_rewards.append(float(reward))
                ep_dones.append(1.0 if done else 0.0)
                ep_values.append(float(val))
                ep_cursors.append(cursor)
                ep_return += float(reward)

                if info.get("illegal"):
                    ep_illegal += 1

            last_v = 0.0
            if not ep_dones[-1]:
                with torch.no_grad():
                    state = torch.from_numpy(self.flatten_obs(obs)).float().unsqueeze(0).to(self.device)
                    last_v = self.value_net(state).squeeze(1).item()

            adv, ret = self.compute_gae(
                ep_rewards, ep_dones, ep_values, last_v,
                gamma=self.cfg.gamma, lam=0.95
            )

            states.extend(ep_states)
            actions.extend(ep_actions)
            old_logps.extend(ep_logps)
            returns.extend(ret.tolist())
            advantages.extend(adv.tolist())
            cursors.extend(ep_cursors)

            stats["episodes"] += 1
            stats["avg_ep_return"] += ep_return
            stats["avg_ep_steps"] += env.state.steps
            stats["success"] += int(env.is_correct())
            stats["timeouts"] += int(info.get("max_steps", False))
            stats["illegal_moves"] += ep_illegal

        if stats["episodes"] > 0:
            stats["avg_ep_return"] /= stats["episodes"]
            stats["avg_ep_steps"] /= stats["episodes"]

        states = np.stack(states).astype(np.float32)
        actions = np.asarray(actions, dtype=np.int64)
        old_logps = np.asarray(old_logps, dtype=np.float32)
        returns = np.asarray(returns, dtype=np.float32)
        advantages = np.asarray(advantages, dtype=np.float32)
        cursors = np.asarray(cursors, dtype=np.int64)

        # print("steps/ep:", stats["avg_ep_steps"])
        # print("avg_ep_return:", stats["avg_ep_return"])
        # print("adv std:", float(np.std(advantages)))
        # print("ret std:", float(np.std(returns)))

        return states, actions, old_logps, returns, advantages, cursors, stats
    
    def learn_ppo(self, loader):
        for _ in range(self.cfg.policy_epochs):
            for state, action, old_logp, returns, advantages, cursors in loader:
                state = state.to(self.device)
                action = action.to(self.device)
                old_logp = old_logp.to(self.device)
                returns = returns.to(self.device)
                advantages = advantages.to(self.device)
                cursors = cursors.to(self.device)

                values = self.value_net(state).squeeze(1)
                logits = self.policy_net(state)

                batch_size = logits.size(0)
                for i in range(batch_size):
                    cursor = int(cursors[i].item())
                    logits[i] = logits[i] + self.mask_by_cursor[cursor]
                
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(action)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - old_logp)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.cfg.epsilon, 1 + self.cfg.epsilon) * advantages
                policy_loss = -(torch.min(surr1, surr2)).mean()

                value_loss = F.mse_loss(values, returns)

                loss = policy_loss + self.cfg.vf_coef * value_loss - self.cfg.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    chain(self.policy_net.parameters(), self.value_net.parameters()),
                    0.5
                )
                self.optimizer.step()

    def train(self, max_digits):
        env = AddPad(max_digits=max_digits)

        for epoch in range(self.cfg.epochs):
            states, actions, old_logps, returns, advantages, cursors, stats = self.collect_epoch(env)

            adv_mean = np.mean(advantages)
            adv_std = np.std(advantages)
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)

            dataset = RLDataset(states, actions, old_logps, returns, advantages, cursors)
            loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)

            self.learn_ppo(loader)

            print(
                f"[digits={max_digits}] epoch={epoch:03d} "
                f"avg_ep_return={stats['avg_ep_return']:.3f} "
                f"avg_ep_steps={stats['avg_ep_steps']:.2f} "
                f"success_rate={stats['success']/max(1,stats['episodes']):.3f} "
                f"timeout_rate={stats['timeouts']/max(1,stats['episodes']):.3f} "
                f"illegal_rate={stats['illegal_moves']/max(1,stats['episodes']):.3f}"
            )
    
    @staticmethod
    def flatten_obs(obs):
        def digit(n, place):
            return (n // place) % 10
        
        a = obs["A"]
        b = obs["B"]
        cursor = obs["cursor"]
        pad = obs["pad"]

        features = [
            digit(a, 100) / 9.0,
            digit(a, 10) / 9.0,
            digit(a, 1) / 9.0,
            digit(b, 100) / 9.0,
            digit(b, 10) / 9.0,
            digit(b, 1) / 9.0,
            1 if cursor == 1 else 0,
            1 if cursor == 2 else 0,
            1 if cursor == 3 else 0,
            1 if cursor == 4 else 0,
            1 if cursor == 5 else 0,
            1 if cursor == 6 else 0,
            1 if cursor == 7 else 0,
        ]
        
        for val in pad:
            if val == EMPTY:
                features.extend([1.0, 0.0])  # is_empty=1, value=0
            else:
                features.extend([0.0, val / 9.0])  # is_empty=0, value normalized
        
        return np.asarray(features, dtype=np.float32)
    
    def make_legal_action_masks(self):
        OUT = {1, 3, 5, 7}
        CARRY = {2, 4, 6}

        mask_by_cursor = {}

        for cursor in range(1, 8):
            mask = torch.full((self.action_size,), -1e9, device=self.device)
            legal = set()
            if cursor in OUT:
                legal.update(range(0, 10))
            if cursor in CARRY:
                legal.add(10)
            if cursor != 7:
                legal.add(11)
            if cursor != 1:
                legal.add(12)
            legal.add(13)
            mask[list(legal)] = 0.0

            mask_by_cursor[cursor] = mask
        return mask_by_cursor
    
if __name__ == "__main__":
    agent = RLAgent()
    agent.train(max_digits=1)
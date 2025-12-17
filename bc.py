import sys
import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import Tuple
from agent import Agent

@dataclass
class BCConfig:
    n_actions: int = 14
    hidden_sizes: Tuple[int, ...] = (128, 128)
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 10
    weight_decay: float = 1e-5
    val_split: float = 0.1
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class MLPModel(nn.Module):
    def __init__(self, input_dim: int, n_actions: int, hidden_sizes: Tuple[int, ...]):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, n_actions)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class BehaviorCloner(Agent):
    def __init__(self, input_dim, config = None):
        self.cfg = config or BCConfig()
        self.model = MLPModel(input_dim, self.cfg.n_actions, self.cfg.hidden_sizes).to(self.cfg.device)

    def act(self, obs, stochastic = False):
        obs = self.flatten_obs(obs)
        
        self.model.eval()
        x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.cfg.device)
        logits = self.model(x).squeeze(0)

        if not stochastic:
            return int(torch.argmax(logits).item())
        
        probs = torch.softmax(logits, dim=0)
        a = torch.multinomial(probs, num_samples=1).item()
        return int(a)
    
    @staticmethod
    def load_npz(path):
        print(f"Loading data from {path}")
        data = np.load(path)
        X = data["X"]
        y = data["y"]
        X = X.astype(np.float32)
        y = y.astype(np.int64)
        return X, y
    
    def make_loaders(self, X, y):
        rng = np.random.default_rng(self.cfg.seed)
        idx = np.arange(len(y))
        rng.shuffle(idx)

        n_val = int(len(y) * self.cfg.val_split)
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

        train_loader = DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=self.cfg.batch_size, shuffle=False, drop_last=False)
        return train_loader, val_loader

    def train(self, X, y):
        train_loader, val_loader = self.make_loaders(X, y)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        loss_fn = nn.CrossEntropyLoss()

        best_val_acc = -1.0
        best_state = None

        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for x_i, y_i in train_loader:
                x_i = x_i.to(self.cfg.device)
                y_i = y_i.to(self.cfg.device)

                y_hat = self.model(x_i)
                loss = loss_fn(y_hat, y_i)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * x_i.size(0)
                train_correct += (y_hat.argmax(dim=1) == y_i).sum().item()
                train_total += x_i.size(0)
            
            train_loss /= max(1, train_total)
            train_acc = train_correct / max(1, train_total)

            val_loss, val_acc = self.evaluate(val_loader)

            print(f"Epoch {epoch:02d} | train loss {train_loss:.4f} acc {train_acc:.4f} "
                  f"| val loss {val_loss:.4f} acc {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return {"best_val_acc": float(best_val_acc)}
    
    @torch.no_grad
    def evaluate(self, loader):
        self.model.eval()
        loss_fn = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in loader:
            x = x.to(self.cfg.device)
            y = y.to(self.cfg.device)

            y_hat = self.model(x)
            loss = loss_fn(y_hat, y)

            total_loss += loss.item() * x.size(0)
            correct += (y_hat.argmax(dim=1) == y).sum().item()
            total += x.size(0)

        val_loss = total_loss / max(1, total)
        val_acc = correct / max(1, total)
        return val_loss, val_acc        
    
    def save(self, path):
        torch.save({"state_dict": self.model.state_dict(), "cfg": self.cfg}, path)
    
    @staticmethod
    def load(path, input_dim) -> "BehaviorCloner":
        print(f"Loading BC from {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("cfg", BCConfig())
        bc = BehaviorCloner(input_dim=input_dim, config=cfg)
        bc.model.load_state_dict(ckpt["state_dict"])
        bc.model.to(bc.cfg.device)
        return bc
    
    @staticmethod
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


def train_bc():
    X_1, y_1 = BehaviorCloner.load_npz("data-1d.npz")
    bc = BehaviorCloner(input_dim=X_1.shape[1])
    bc.train(X_1, y_1)
    bc.save("bc_1d.pt")

    X_2, y_2 = BehaviorCloner.load_npz("data-2d.npz")
    bc.train(X_2, y_2)
    bc.save("bc_2d.pt")

    X_3, y_3 = BehaviorCloner.load_npz("data-3d.npz")
    bc.train(X_3, y_3)
    bc.save("bc_3d.pt")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "t":
        train_bc()
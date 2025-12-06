"""
pipeline.py (Submission-ready, GPU-default)
-------------------------------------------
Quick Deep RL project for <=6 hours scope:
- Environment: CartPole-v1
- Algorithms: REINFORCE (baseline) vs PPO-Clip (main)
- PPO hyperparameter ablation: clip_eps in {0.1, 0.2}
- Seeds: {0, 1, 2}
- Saves:
    1) quick_ppo_project_results.csv  (summary table)
    2) runs/curves_<algo>_seedX_clipY.npz  (per-run curves)
    3) (optional) runs/plot_summary.png

Key features:
- tqdm progress bars
- shows learning rate (lr) in postfix
- shows recent average return
- GPU default if available (auto fallback to CPU)

Usage examples:
1) Fast sanity check:
   python pipeline.py --fast
2) Full run + summary plot:
   python pipeline.py --plot
3) Force CPU:
   python pipeline.py --device cpu
"""

import os
import time
import random
import argparse
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm, trange


# ============================================================
# 0) Utilities
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def moving_average(x: List[float], window: int = 20) -> np.ndarray:
    if len(x) == 0:
        return np.array([])
    w = min(window, len(x))
    out = []
    for i in range(len(x)):
        s = max(0, i - w + 1)
        out.append(np.mean(x[s:i + 1]))
    return np.array(out, dtype=np.float32)


# ============================================================
# 1) Models
# ============================================================
class MLPPolicy(nn.Module):
    """Discrete action policy network."""
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.logits = nn.Linear(hidden, act_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        return self.logits(h)

    def dist(self, obs: torch.Tensor):
        logits = self(obs)
        return torch.distributions.Categorical(logits=logits)


class MLPValue(nn.Module):
    """State-value function."""
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ============================================================
# 2) REINFORCE
# ============================================================
@dataclass
class ReinforceCfg:
    env_id: str = "CartPole-v1"
    seed: int = 0
    total_episodes: int = 600
    gamma: float = 0.99
    lr: float = 1e-3
    hidden: int = 128
    
    # GPU but not exist than CPU
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    log_window: int = 20


def train_reinforce(cfg: ReinforceCfg) -> Dict[str, Any]:
    import gymnasium as gym

    set_seed(cfg.seed)
    env = gym.make(cfg.env_id)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    device = torch.device(cfg.device)

    policy = MLPPolicy(obs_dim, act_dim, cfg.hidden).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    episode_returns: List[float] = []

    pbar = trange(
        cfg.total_episodes,
        desc=f"REINFORCE {cfg.env_id} seed={cfg.seed}",
        leave=False
    )

    for ep in pbar:
        obs, _ = env.reset(seed=cfg.seed + ep)
        done = False
        logps = []
        rewards = []

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            dist = policy.dist(obs_t)
            action = dist.sample()

            logps.append(dist.log_prob(action))

            next_obs, reward, terminated, truncated, _ = env.step(int(action.item()))
            done = terminated or truncated

            rewards.append(float(reward))
            obs = next_obs

        # Compute returns G_t
        G = 0.0
        returns = []
        for r in reversed(rewards):
            G = r + cfg.gamma * G
            returns.append(G)
        returns.reverse()

        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

        # Normalize returns for stability
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        logps_t = torch.stack(logps)
        loss = -(logps_t * returns_t).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        ep_ret = float(sum(rewards))
        episode_returns.append(ep_ret)

        lr_now = opt.param_groups[0]["lr"]
        recent = np.mean(episode_returns[-cfg.log_window:])

        pbar.set_postfix({
            "ep_ret": f"{ep_ret:.1f}",
            f"recent{cfg.log_window}": f"{recent:.1f}",
            "lr": f"{lr_now:.2e}"
        })

    env.close()

    result = {
        "algo": "REINFORCE",
        "seed": cfg.seed,
        "clip_eps": None,
        "ent_coef": None,
        "returns": episode_returns,
        "mean_last100": float(np.mean(episode_returns[-100:])) if len(episode_returns) >= 100 else float(np.mean(episode_returns)),
        "cfg": asdict(cfg)
    }

    # after training loop, before return result
    ensure_dir("models")
    torch.save(policy.state_dict(), f"models/reinforce_seed{cfg.seed}.pt")

    return result


# 3) PPO-Clip

@dataclass
class PPOCfg:
    env_id: str = "CartPole-v1"
    seed: int = 0

    total_steps: int = 100_000
    rollout_len: int = 1024
    update_epochs: int = 5
    minibatch: int = 256

    gamma: float = 0.99
    gae_lambda: float = 0.95

    clip_eps: float = 0.2
    lr: float = 3e-4

    ent_coef: float = 0.0
    vf_coef: float = 0.5

    hidden: int = 128
    # GPU default if available
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_window: int = 20


def compute_gae(rewards, dones, values, next_value, gamma, lam):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        nextv = next_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * nextv * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    ret = adv + values
    return adv, ret


def train_ppo(cfg: PPOCfg) -> Dict[str, Any]:
    import gymnasium as gym

    set_seed(cfg.seed)
    env = gym.make(cfg.env_id)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    device = torch.device(cfg.device)

    policy = MLPPolicy(obs_dim, act_dim, cfg.hidden).to(device)
    value_fn = MLPValue(obs_dim, cfg.hidden).to(device)

    opt = torch.optim.Adam(
        list(policy.parameters()) + list(value_fn.parameters()),
        lr=cfg.lr
    )

    # Buffers
    obs_buf = np.zeros((cfg.rollout_len, obs_dim), dtype=np.float32)
    act_buf = np.zeros((cfg.rollout_len,), dtype=np.int64)
    logp_buf = np.zeros((cfg.rollout_len,), dtype=np.float32)
    rew_buf = np.zeros((cfg.rollout_len,), dtype=np.float32)
    done_buf = np.zeros((cfg.rollout_len,), dtype=np.float32)
    val_buf = np.zeros((cfg.rollout_len,), dtype=np.float32)

    obs, _ = env.reset(seed=cfg.seed)

    episode_returns: List[float] = []
    ep_ret = 0.0
    steps = 0

    num_updates = (cfg.total_steps + cfg.rollout_len - 1) // cfg.rollout_len
    pbar = trange(
        num_updates,
        desc=f"PPO {cfg.env_id} seed={cfg.seed} clip={cfg.clip_eps}",
        leave=False
    )

    for _u in pbar:
        # ================
        # Rollout collection
        # ================
        for t in range(cfg.rollout_len):
            obs_buf[t] = obs

            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                dist = policy.dist(obs_t)
                action = dist.sample()
                logp = dist.log_prob(action)
                v = value_fn(obs_t)

            act = int(action.item())
            next_obs, reward, terminated, truncated, _ = env.step(act)
            done = float(terminated or truncated)

            act_buf[t] = act
            logp_buf[t] = float(logp.item())
            rew_buf[t] = float(reward)
            done_buf[t] = done
            val_buf[t] = float(v.item())

            obs = next_obs
            ep_ret += float(reward)
            steps += 1

            if done:
                episode_returns.append(ep_ret)
                ep_ret = 0.0
                obs, _ = env.reset()

            if steps >= cfg.total_steps:
                # truncate rollout if we reached budget
                if t < cfg.rollout_len - 1:
                    obs_buf[t + 1:] = obs_buf[t]
                    act_buf[t + 1:] = act_buf[t]
                    logp_buf[t + 1:] = logp_buf[t]
                    rew_buf[t + 1:] = 0.0
                    done_buf[t + 1:] = 1.0
                    val_buf[t + 1:] = val_buf[t]
                break

        # Bootstrap value
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            next_value = float(value_fn(obs_t).item())

        adv, ret = compute_gae(
            rew_buf, done_buf, val_buf,
            next_value, cfg.gamma, cfg.gae_lambda
        )
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # To torch
        obs_b = torch.tensor(obs_buf, dtype=torch.float32, device=device)
        act_b = torch.tensor(act_buf, dtype=torch.int64, device=device)
        oldlogp_b = torch.tensor(logp_buf, dtype=torch.float32, device=device)
        adv_b = torch.tensor(adv, dtype=torch.float32, device=device)
        ret_b = torch.tensor(ret, dtype=torch.float32, device=device)

        # ================
        # PPO Update
        # ================
        idxs = np.arange(cfg.rollout_len)
        for _ in range(cfg.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, cfg.rollout_len, cfg.minibatch):
                mb = idxs[start:start + cfg.minibatch]

                dist = policy.dist(obs_b[mb])
                logp = dist.log_prob(act_b[mb])
                entropy = dist.entropy().mean()

                v = value_fn(obs_b[mb])

                ratio = torch.exp(logp - oldlogp_b[mb])
                surr1 = ratio * adv_b[mb]
                surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_b[mb]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(v, ret_b[mb])

                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

                opt.zero_grad()
                loss.backward()
                opt.step()

        # tqdm postfix
        lr_now = opt.param_groups[0]["lr"]
        recent = np.mean(episode_returns[-cfg.log_window:]) if len(episode_returns) > 0 else 0.0

        pbar.set_postfix({
            "steps": f"{min(steps, cfg.total_steps)}/{cfg.total_steps}",
            f"recent{cfg.log_window}": f"{recent:.1f}",
            "lr": f"{lr_now:.2e}",
        })

        

        if steps >= cfg.total_steps:
            break

    env.close()

    tail = episode_returns[-50:] if len(episode_returns) >= 50 else episode_returns

    result = {
        "algo": "PPO",
        "seed": cfg.seed,
        "clip_eps": cfg.clip_eps,
        "ent_coef": cfg.ent_coef,
        "returns": episode_returns,
        "mean_last50": float(np.mean(tail)) if len(tail) > 0 else 0.0,
        "cfg": asdict(cfg)
    }

    # after training loop, before return result
    ensure_dir("models")
    torch.save(policy.state_dict(), f"models/ppo_seed{cfg.seed}_clip{cfg.clip_eps}.pt")

    return result



# 4) Experiment Orchestrator

@dataclass
class ProjectCfg:
    env_id: str = "CartPole-v1"
    seeds: List[int] = None
    clip_list: List[float] = None

    # REINFORCE
    reinforce_episodes: int = 600
    reinforce_lr: float = 1e-3

    # PPO
    ppo_steps: int = 100_000
    ppo_lr: float = 3e-4
    rollout_len: int = 1024
    update_epochs: int = 5
    minibatch: int = 256
    gae_lambda: float = 0.95
    ent_coef: float = 0.0

    hidden: int = 128

    # GPU default if available
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Output
    out_dir: str = "runs"
    summary_csv: str = "quick_ppo_project_results.csv"

    # Fast test mode
    fast: bool = False


def run_quick_project(proj: ProjectCfg, save_plots: bool = False):
    import pandas as pd

    ensure_dir(proj.out_dir)

    seeds = proj.seeds if proj.seeds is not None else [0, 1, 2]
    clip_list = proj.clip_list if proj.clip_list is not None else [0.1, 0.2]

    # Fast mode reduces budgets
    if proj.fast:
        reinforce_episodes = 200
        ppo_steps = 20_000
        rollout_len = 512
        update_epochs = 3
    else:
        reinforce_episodes = proj.reinforce_episodes
        ppo_steps = proj.ppo_steps
        rollout_len = proj.rollout_len
        update_epochs = proj.update_epochs

    rows = []
    curve_files = []

    print("=== START QUICK PROJECT ===")
    print(f"Env: {proj.env_id}")
    print(f"Seeds: {seeds}")
    print(f"PPO clip list: {clip_list}")
    print(f"Using device: {proj.device} (cuda_available={torch.cuda.is_available()})")
    if torch.cuda.is_available() and proj.device.startswith("cuda"):
        try:
            print(f"GPU name: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
    print(f"Fast mode: {proj.fast}")
    print("====================================")

    # ----------------
    # 1) REINFORCE
    # ----------------
    for s in tqdm(seeds, desc="REINFORCE seeds"):
        cfg = ReinforceCfg(
            env_id=proj.env_id,
            seed=s,
            total_episodes=reinforce_episodes,
            lr=proj.reinforce_lr,
            hidden=proj.hidden,
            device=proj.device
        )
        t0 = time.time()
        res = train_reinforce(cfg)
        t1 = time.time()

        returns = res["returns"]
        curve_path = os.path.join(proj.out_dir, f"curves_REINFORCE_seed{s}.npz")
        np.savez(curve_path, returns=np.array(returns, dtype=np.float32))
        curve_files.append(curve_path)

        rows.append({
            "algo": "REINFORCE",
            "seed": s,
            "clip_eps": None,
            "ent_coef": None,
            "mean_return_tail": res["mean_last100"],
            "episodes_or_steps": reinforce_episodes,
            "elapsed_sec": round(t1 - t0, 2),
            "device": proj.device
        })

    # ----------------
    # 2) PPO (clip ablation)
    # ----------------
    for clip in tqdm(clip_list, desc="PPO clips"):
        for s in tqdm(seeds, desc=f"PPO seeds (clip={clip})", leave=False):
            cfg = PPOCfg(
                env_id=proj.env_id,
                seed=s,
                total_steps=ppo_steps,
                rollout_len=rollout_len,
                update_epochs=update_epochs,
                minibatch=proj.minibatch,
                gamma=0.99,
                gae_lambda=proj.gae_lambda,
                clip_eps=clip,
                lr=proj.ppo_lr,
                ent_coef=proj.ent_coef,
                vf_coef=0.5,
                hidden=proj.hidden,
                device=proj.device
            )
            t0 = time.time()
            res = train_ppo(cfg)
            t1 = time.time()

            returns = res["returns"]
            curve_path = os.path.join(proj.out_dir, f"curves_PPO_seed{s}_clip{clip}.npz")
            np.savez(curve_path, returns=np.array(returns, dtype=np.float32))
            curve_files.append(curve_path)

            rows.append({
                "algo": "PPO",
                "seed": s,
                "clip_eps": clip,
                "ent_coef": proj.ent_coef,
                "mean_return_tail": res["mean_last50"],
                "episodes_or_steps": ppo_steps,
                "elapsed_sec": round(t1 - t0, 2),
                "device": proj.device
            })

    df = pd.DataFrame(rows)
    df.to_csv(proj.summary_csv, index=False)

    print("\n=== SUMMARY TABLE ===")
    print(df)
    print(f"\nSaved summary CSV: {proj.summary_csv}")
    print(f"Saved curves: {len(curve_files)} files in '{proj.out_dir}/'")

    # Optional plotting
    if save_plots:
        try:
            plot_summary(df, proj.out_dir)
        except Exception as e:
            print(f"[Plot skipped] {e}")

    print("=== END QUICK PROJECT ===")



# make Plot

def plot_summary(df, out_dir: str):
    """
    Creates a lightweight summary plot:
    - Mean±std of tail performance over seeds
    - REINFORCE vs PPO clip ablation
    """
    import matplotlib.pyplot as plt

    ensure_dir(out_dir)

    reinforce = df[df["algo"] == "REINFORCE"].copy()
    ppo = df[df["algo"] == "PPO"].copy()

    r_mean = reinforce["mean_return_tail"].mean() if len(reinforce) else 0.0
    r_std = reinforce["mean_return_tail"].std(ddof=0) if len(reinforce) else 0.0

    ppo_grp = ppo.groupby("clip_eps")["mean_return_tail"].agg(["mean", "std"]).reset_index()

    plt.figure()
    xs, ys, es = [], [], []

    xs.append(-1)
    ys.append(r_mean)
    es.append(r_std)

    for _, row in ppo_grp.iterrows():
        xs.append(float(row["clip_eps"]))
        ys.append(float(row["mean"]))
        es.append(float(row["std"]) if not np.isnan(row["std"]) else 0.0)

    plt.errorbar(xs, ys, yerr=es, fmt='o')
    plt.xticks(xs, ["REINFORCE"] + [f"PPO clip={c}" for c in ppo_grp["clip_eps"].tolist()])
    plt.ylabel("Mean Return (tail)")
    plt.title("Quick RL Project Summary (mean ± std over seeds)")
    plt.tight_layout()

    out_path = os.path.join(out_dir, "plot_summary.png")
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved plot: {out_path}")



# 6) CLI

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="CartPole-v1")
    # GPU default in CLI too
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--fast", action="store_true", help="Very quick sanity run")
    p.add_argument("--plot", action="store_true", help="Save a summary PNG plot")
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--clips", type=str, default="0.1,0.2")
    return p.parse_args()


def main():
    args = parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip() != ""]
    clips = [float(x.strip()) for x in args.clips.split(",") if x.strip() != ""]

    # Auto fallback if CUDA not available
    chosen = args.device
    if chosen == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA not available. Falling back to CPU.")
        chosen = "cpu"

    proj = ProjectCfg(
        env_id=args.env,
        seeds=seeds,
        clip_list=clips,
        device=chosen,
        fast=args.fast
    )

    run_quick_project(proj, save_plots=args.plot)


if __name__ == "__main__":
    main()




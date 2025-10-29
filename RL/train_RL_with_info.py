#!/usr/bin/env python3

import argparse
import os
import json
import zipfile
from pathlib import Path
from typing import Any, Dict
import numpy as np

import torch as th
import torch.optim as optim
from stable_baselines3 import PPO

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv


from swarm.utils.env_factory import make_env
from swarm.validator.task_gen import random_task

# Import from centralized constants
from swarm.constants import SIM_DT, HORIZON_SEC, SAFE_META_FILENAME

def information_save(model: PPO, save_stem: str) -> None:
    """
    Append a small, SAFE JSON file to the SB3 checkpoint zip with just the
    non-executable metadata the secure loader needs (activation + net_arch + SDE).

    Args:
        model: Trained PPO model
        save_stem: Path used with `model.save(...)`. Can be with or without ".zip".
                   We will write the JSON into that zip archive.
    """
    # Resolve the actual .zip path that SB3 produced
    zip_path = Path(save_stem)
    if zip_path.suffix != ".zip":
        zip_path = zip_path.with_suffix(".zip")

    # --- Gather minimal metadata (no pickle, no code) ---
    # Activation function name
    act_attr = getattr(model.policy, "activation_fn", th.nn.ReLU)
    if isinstance(act_attr, type):
        act_name = act_attr.__name__  # e.g., "ReLU"
    else:
        act_name = act_attr.__class__.__name__  # instance -> "ReLU", "Tanh", ...

    # net_arch (read back if available; only for reference)
    def _infer_net_arch_from_policy() -> Any:
        me = getattr(model.policy, "mlp_extractor", None)
        if me is not None and hasattr(me, "net_arch"):
            return me.net_arch

        # Fallback: reconstruct sizes from Linear layers
        def _layers(seq) -> list[int]:
            out = []
            for m in getattr(seq, "_modules", {}).values():
                if isinstance(m, th.nn.Linear):
                    out.append(int(m.out_features))
            return out

        shared = _layers(getattr(me, "shared_net", th.nn.Sequential()))
        pi = _layers(getattr(me, "policy_net", th.nn.Sequential()))
        vf = _layers(getattr(me, "value_net", th.nn.Sequential()))
        return (shared + [dict(pi=pi, vf=vf)]) if shared else dict(pi=pi, vf=vf)

    net_arch = _infer_net_arch_from_policy()
    use_sde = bool(getattr(model, "use_sde", False))

    meta: Dict[str, Any] = {
        "format": "sb3-safe-meta@1",
        "algo": "PPO",
        "activation_fn": act_name,  # e.g., "ReLU", "Tanh"
        "net_arch": net_arch,  # informational; secure loader still infers from weights
        "use_sde": use_sde,
    }

    # --- Write/replace JSON inside the zip ---
    with zipfile.ZipFile(zip_path, mode="a", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(SAFE_META_FILENAME, json.dumps(meta, indent=2))


class CustomActorCriticPolicy(ActorCriticPolicy):
    def _build_optimizer(self):
        actor_lr = 3e-4
        critic_lr = 3e-3

        actor_params = list(self.actor.parameters())
        critic_params = list(self.critic.parameters())

        self.optimizer = optim.Adam(
            [
                {"params": actor_params, "lr": actor_lr},
                {"params": critic_params, "lr": critic_lr},
            ]
        )


class DynamicSubprocVecEnv(SubprocVecEnv):
    def __init__(self, make_env_fn, num_envs, resample_every=10000000):
        self.make_env_fn = make_env_fn
        self.num_envs = num_envs
        self.resample_every = resample_every
        self.total_steps = 0
        env_fns = [make_env_fn(np.random.randint(1, 1000001)) for _ in range(num_envs)]
        super().__init__(env_fns)

    def step_async(self, actions):
        self.total_steps += self.num_envs
        if self.total_steps >= self.resample_every:
            print(
                f"[DynamicSubprocVecEnv] Resampling environments after {self.total_steps} steps."
            )
            self.resample_envs()
            self.total_steps = 0
        return super().step_async(actions)

    def resample_envs(self):
        self.close()  # kill old envs

        env_fns = [
            self.make_env_fn(np.random.randint(1, 1000001))
            for _ in range(self.num_envs)
        ]
        self.__init__(self.make_env_fn, self.num_envs, self.resample_every)


def make_env_fn(seed):
    def _init():
        task = random_task(sim_dt=1 / 50, horizon=30, seed=seed)
        print(f"Creating env with seed {seed}")
        return make_env(task, gui=False)

    return _init


def make_random_test_env(num_envs=5):
    seeds = [np.random.randint(1, 1000001) for _ in range(num_envs)]
    env_fns = [make_env_fn(seed) for seed in seeds]
    return SubprocVecEnv(env_fns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100000000)
    args = parser.parse_args()

    task = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC, seed=1)
    env = make_env(task, gui=False)

    # train_env = DynamicSubprocVecEnv(make_env_fn, num_envs=8, resample_every=2000000)
    # test_env = make_random_test_env(8)

    policy_kwargs = dict(
        net_arch=dict(pi=[512, 512], vf=[512, 512, 256]), activation_fn=th.nn.LeakyReLU
    )

    # model = PPO("MlpPolicy", env, verbose=1)
    model = PPO(
        CustomActorCriticPolicy,
        env,
        verbose=1,  
        device="cuda",
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=1024,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./logs/",
    )
    
    eval_callback = EvalCallback(
        env,
        best_model_save_path="./best_model/",
        log_path="./logs/",
        eval_freq=10000,
        deterministic=True,
        render=False,
    )
    
    model.learn(total_timesteps=args.timesteps, callback=eval_callback)
    # model.learn(args.timesteps)

    # Create model directory if it doesn't exist
    os.makedirs("model", exist_ok=True)

    # Save as usual
    save_stem = "model/ppo_policy"  # SB3 will create "model/ppo_policy.zip"
    model.save(save_stem)

    # Append minimal, safe metadata for the secure loader
    information_save(model, save_stem)
    
    best_stem = "best_model/best_model.zip"
    model = PPO.load(best_stem, device="cpu")
    information_save(model, best_stem)

    # train_env.close()
    # test_env.close()
    env.close()


if __name__ == "__main__":
    main()

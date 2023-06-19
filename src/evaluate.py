from typing import List

import numpy as np
import torch
from einops import rearrange
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from utils import RandomHeuristic

@torch.no_grad()
def get_observation_after_n_steps_wm(
        env,
        start_and_goal_observation,
        prev_trajectory,
        agent,
        steps=5,
        save_plots=False,
        idx = None
    ):

    if save_plots:
        fig, axs = plt.subplots(1, steps+1, figsize=(20, 4))  # one plot for start, goal and each of the 5 steps
    
        axs[0].imshow(start_and_goal_observation[0].permute(1, 2, 0))  # assumes image dimensions are (C, H, W)
        axs[0].set_title('Start Observation')
        axs[1].imshow(start_and_goal_observation[1].permute(1, 2, 0))  # assumes image dimensions are (C, H, W)
        axs[1].set_title('Goal Observation')


    ACTION_GET_GOAL = env.num_actions
    ACTION_START_PLANNING = env.num_actions + 1

    obs, goal_obs = start_and_goal_observation # s_t
    obs = obs.float().div(255)
    goal_obs = obs.float().div(255)
    assert 0 <= obs.min() <= 1 and 0 <= obs.max() <= 1
    assert obs.ndim == 3 # and obs.shape[1:] == (3, 64, 64)
    
    if agent is None:
        agent = RandomHeuristic(num_actions=env.num_actions)
        obs = obs.unsqueeze(0)

        for i in range(steps - 1):
            act = agent.act(obs)
            obs, reward, done, _ = env.step(act)

            obs = rearrange(torch.FloatTensor(obs).div(255), 'n h w c -> n c h w')

        return (obs.squeeze(0) * 255).byte()

    observations, actions = prev_trajectory
    observations = observations.float().div(255)
    observations = torch.cat(
        [observations, goal_obs.unsqueeze(0), obs.unsqueeze(0)], dim=0
    )
    observations = observations.unsqueeze(0).to(agent.device)


    actions = torch.cat([
        actions, 
        torch.tensor([ACTION_START_PLANNING]),
        torch.tensor([ACTION_GET_GOAL]),
        ], dim=0)

    actions = actions.unsqueeze(0).to(agent.device)

    obs_tokens = agent.tokenizer.encode(observations, should_preprocess=True).tokens
    act_tokens = rearrange(actions, 'b l -> b l 1')
    tokens = rearrange(torch.cat((obs_tokens[:, :-1], act_tokens), dim=2), 'b l k1 -> b (l k1)')  # (B, L(K+1))
    tokens = torch.cat((tokens, obs_tokens[:, -1, :]), dim=1)
    for i in range(steps - 1):
        with torch.no_grad():
            action_logits = agent.world_model(tokens).logits_actions[:, -1:, :-2]
            act_token = torch.argmax(action_logits, dim=-1)
            act = act_token.squeeze(0).cpu().numpy()
        obs, reward, done, _ = env.step(act)

        obs = rearrange(torch.FloatTensor(obs).div(255), 'n h w c -> n c h w').to(agent.device)
        tokens = torch.cat(
            [tokens, act_token, agent.tokenizer.encode(obs, should_preprocess=True).tokens], 
            dim=1
        )

        if save_plots:
            # adding the observation at each step to the plot
            axs[i + 2].imshow(obs.squeeze(0).cpu().permute(1, 2, 0))  # assumes image dimensions are (C, H, W)
            axs[i + 2].set_title(f'Observation After {i + 1} Steps')

    if save_plots:
        plt.tight_layout()
        plt.savefig(f'eval/{datetime.now().strftime("%H:%M:%S")}_{i}.png')
        plt.close(fig)  # close the figure


    return (obs.squeeze(0) * 255).byte()

@torch.no_grad()
def get_plans(
        env,
        start_and_goal_observation,
        prev_trajectory,
        agent,
        save_plots=False,
        idx = None
    ):

    if save_plots:
        fig, axs = plt.subplots(1, 6, figsize=(20, 4))  # one plot for start, goal and each of the 5 steps
    
        axs[0].imshow(start_and_goal_observation[0].permute(1, 2, 0))  # assumes image dimensions are (C, H, W)
        axs[0].set_title('Start Observation')
        axs[1].imshow(start_and_goal_observation[1].permute(1, 2, 0))  # assumes image dimensions are (C, H, W)
        axs[1].set_title('Goal Observation')


    ACTION_GET_GOAL = env.num_actions
    ACTION_START_PLANNING = env.num_actions + 1

    obs, goal_obs = start_and_goal_observation # s_t
    obs = obs.float().div(255)
    goal_obs = obs.float().div(255)
    assert 0 <= obs.min() <= 1 and 0 <= obs.max() <= 1
    assert obs.ndim == 3 # and obs.shape[1:] == (3, 64, 64)

    observations, actions = prev_trajectory
    observations = observations.float().div(255)
    observations = torch.cat(
        [observations, goal_obs.unsqueeze(0), obs.unsqueeze(0)], dim=0
    )
    observations = observations.unsqueeze(0).to(agent.device)

    actions = torch.cat([
        actions, 
        torch.tensor([ACTION_START_PLANNING]),
        torch.tensor([ACTION_GET_GOAL]),
        ], dim=0)

    actions = actions.unsqueeze(0).to(agent.device)

    obs_tokens = agent.tokenizer.encode(observations, should_preprocess=True).tokens
    act_tokens = rearrange(actions, 'b l -> b l 1')
    tokens = rearrange(torch.cat((obs_tokens[:, :-1], act_tokens), dim=2), 'b l k1 -> b (l k1)')  # (B, L(K+1))
    tokens = torch.cat((tokens, obs_tokens[:, -1, :]), dim=1)
    for i in range(4):
        with torch.no_grad():
            action_logits = agent.world_model(tokens).logits_actions[:, -1:, :-2]
            act_token = torch.argmax(action_logits, dim=-1)
        tokens = torch.cat([tokens, act_token], dim=1)

        obs_tokens = []
        for j in range(16):
            obs_token = agent.world_model(tokens).logits_observations[:, -1:, :].argmax(dim=-1)
            tokens = torch.cat([tokens, obs_token], dim=1)
            obs_tokens.append(obs_token)
        obs_tokens = torch.cat(obs_tokens, dim=1)

        if save_plots:
            # adding the observation at each step to the plot
            # here, we need to convert Image.Image to plt
            obs_img = decode_obs_tokens(obs_tokens, tokenizer=agent.tokenizer).cpu().squeeze(0).permute(1, 2, 0)
            axs[i + 2].imshow(obs_img)
            axs[i + 2].set_title(f'Imagination After {i + 1} Steps')

    if save_plots:
        plt.tight_layout()
        plt.savefig(f'eval/{datetime.now().strftime("%H:%M:%S")}_{i}_imag.png')
        plt.close(fig)  # close the figure


    return tokens

@torch.no_grad()
def decode_obs_tokens(obs_tokens, tokenizer) -> List[Image.Image]:
    embedded_tokens = tokenizer.embedding(obs_tokens)     # (B, K, E)
    z = rearrange(embedded_tokens, 'b (h w) e -> b e h w', h=int(np.sqrt(16))) # num_observations_tokens
    rec = tokenizer.decode(z, should_postprocess=True)         # (B, C, H, W)
    return torch.clamp(rec, 0, 1)


@torch.no_grad()
def get_difference(obs_1, obs_2, tokenizer):
    obs_1 = obs_1.unsqueeze(0).float().div(255)
    obs_2 = obs_2.unsqueeze(0).float().div(255)
    
    # get the difference between embedding
    obs_1 = tokenizer.preprocess_input(rearrange(obs_1, 'b c h w -> b c h w'))
    z1, _, _ = tokenizer(obs_1, should_preprocess=False, should_postprocess=False)
    
    obs_2 = tokenizer.preprocess_input(rearrange(obs_2, 'b c h w -> b c h w'))
    z2, _, _ = tokenizer(obs_2, should_preprocess=False, should_postprocess=False)
    
    dist_to_embeddings = torch.sum((z1 - z2) ** 2).item()
    return dist_to_embeddings

@torch.no_grad()
def evaluate(env, dataset, agent, planning_steps=5, save_plots=False, random=False):
    losses = []

    for episode in dataset.episodes:
        for i in range(0, len(episode) - 20, 20):
            start_idx = i
            end_idx = i + 20
            
            assert end_idx < len(episode) 
            # randomly select goal idx among the first 10 steps
            plan_idx = start_idx + 5
            plan_observation = episode.observations[plan_idx]
            goal_observation = episode.observations[plan_idx+planning_steps-1]
            observations = episode.observations[start_idx:plan_idx+1]
            actions = episode.actions[start_idx:plan_idx]

            env.reset()
            env.env.unwrapped.restore_full_state(episode.states[plan_idx])
            
            observation_after_n_steps = get_observation_after_n_steps_wm(
                env,
                start_and_goal_observation=(plan_observation, goal_observation),
                prev_trajectory=(observations, actions),
                agent=agent if not random else None,
                steps=planning_steps,
                save_plots=save_plots,
                idx = i
            )

            if not random and save_plots:
                get_plans(
                    env,
                    start_and_goal_observation=(plan_observation, goal_observation),
                    prev_trajectory=(observations, actions),
                    agent=agent,
                    save_plots=save_plots,
                    idx = i
                )

            loss = get_difference(
                observation_after_n_steps.to(agent.device), 
                goal_observation.to(agent.device),
                tokenizer=agent.tokenizer
            )
            losses.append(loss)

            del observation_after_n_steps
            del goal_observation

    metrics = {'world_model/eval/goal_distance': np.mean(losses)}
    return metrics


@torch.no_grad()
def evaluate_without_history(env, dataset, agent, planning_steps=5, save_plots=False, random=False):
    losses = []

    for episode in dataset.episodes:
        for i in range(0, len(episode) - planning_steps, planning_steps):
            start_idx = i
            end_idx = i + planning_steps
            
            assert end_idx < len(episode) 
            # randomly select goal idx among the first 10 steps
            plan_idx = start_idx
            plan_observation = episode.observations[plan_idx]
            goal_observation = episode.observations[plan_idx+planning_steps-1]
            observations = episode.observations[start_idx:plan_idx+1]
            actions = episode.actions[start_idx:plan_idx]

            env.reset()
            env.env.unwrapped.restore_full_state(episode.states[plan_idx])
            
            observation_after_n_steps = get_observation_after_n_steps_wm(
                env,
                start_and_goal_observation=(plan_observation, goal_observation),
                prev_trajectory=(observations, actions),
                agent=agent if not random else None,
                steps=planning_steps,
                save_plots=save_plots,
                idx = i
            )

            loss = get_difference(
                observation_after_n_steps.to(agent.device), 
                goal_observation.to(agent.device),
                tokenizer=agent.tokenizer
            )
            losses.append(loss)

            del observation_after_n_steps
            del goal_observation

    metrics = {'world_model/eval/goal_distance': np.mean(losses)}
    return metrics


import random
import torch
import numpy as np
from collections import deque
from race_env import RaceEnv
from model import DQN
import torch.optim as optim
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            indices,
            weights,
        )

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority

def train(env, model, optimizer, loss_fn, episodes=5000, start_episode=0, gamma=0.99, 
    epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, batch_size=256, buffer_capacity=10000, alpha=0.6, beta_start=0.4, beta_frames=100000):
    memory = PrioritizedReplayBuffer(buffer_capacity, alpha)
    scaler = GradScaler()
    max_reward = float('-inf')
    episode_rewards = []  # Store total rewards for each episode
    episode_scores = []   # Store scores for each episode

    target_model = DQN(input_dim=16, output_dim=3).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    def update_target_model():
        target_model.load_state_dict(model.state_dict())

    def remember(state, action, reward, next_state, done):
        memory.add(state, action, reward, next_state, done)

    def replay(beta):
        if len(memory.buffer) < batch_size:
            return
        states, actions, rewards, next_states, dones, indices, weights = memory.sample(batch_size, beta)
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)
        weights = torch.tensor(weights, dtype=torch.float32).to(device)

        with autocast():
            q_values = model(states)
            next_q_values = model(next_states)
            next_q_state_values = target_model(next_states)

            q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_value = next_q_state_values.gather(1, next_q_values.argmax(1).unsqueeze(1)).squeeze(1)
            expected_q_value = rewards + gamma * next_q_value * (1 - dones)

            loss = (q_value - expected_q_value.detach()).pow(2) * weights
            priorities = loss + 1e-5
            loss = loss.mean()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        memory.update_priorities(indices, priorities.data.cpu().numpy())

    def act(state):
        if np.random.rand() <= epsilon:
            return random.randrange(env.action_space.n)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with autocast():
            act_values = model(state_tensor)
        return torch.argmax(act_values).item()

    beta = beta_start
    beta_increment = (1.0 - beta_start) / beta_frames

    for episode in range(start_episode, start_episode + episodes):
        state = env.reset()
        done = False
        total_reward = 0
        total_score = 0
        step_count = 0
        while not done:
            action = act(state)
            next_state, reward, done, _ = env.step(action)
            remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            total_score = env.score  # Update score
            step_count += 1

        episode_rewards.append(total_reward)  # Store total reward for this episode
        episode_scores.append(total_score)    # Store score for this episode

        if epsilon > min_epsilon:
            epsilon *= epsilon_decay

        replay(beta)
        
        if total_reward > max_reward:
            max_reward = total_reward

        if episode % 10 == 0:
            print(f'Episode {episode + 1}, Total Reward: {total_reward}, Max Reward: {max_reward}, Total Score: {total_score}')

        if episode % 50 == 0:
            update_target_model()

        beta = min(1.0, beta + beta_increment)

    torch.save(model.state_dict(), 'trained_model.pth')

    # Plot learning curve
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.plot(range(episodes), episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Learning Curve')

    # Plot average reward
    avg_reward = np.mean(episode_rewards)
    plt.subplot(1, 3, 2)
    plt.plot(range(episodes), [avg_reward]*episodes, label='Average Reward')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Average Reward Over Episodes')
    plt.legend()

    # Plot average score
    avg_score = np.mean(episode_scores)
    max_score = np.max(episode_scores)
    plt.subplot(1, 3, 3)
    plt.plot(range(episodes), [avg_score]*episodes, label='Average Score')
    plt.plot(range(episodes), [max_score]*episodes, label='Max Score')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Average and Max Score Over Episodes')
    plt.legend()

    plt.tight_layout()
    plt.show()

from train import train
from test import play_game
from race_env import RaceEnv
from model import DQN
import torch.optim as optim
import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    env = RaceEnv(render_mode=False)
    model = DQN(input_dim=16, output_dim=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    loss_fn = nn.MSELoss()

    start_episode = 0

    train(env, model, optimizer, loss_fn, episodes=5000, start_episode=start_episode)

    test_env = RaceEnv(render_mode=True)  # Rendering enabled during play
    play_game(test_env, model)

if __name__ == "__main__":
    main()

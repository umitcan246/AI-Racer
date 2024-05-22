import torch
from race_env import RaceEnv
from model import DQN
import time
import pygame

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def play_game(env, model, episodes=1):
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        state = env.reset()
                        total_reward = 0
                        step_count = 0
                        print("Game restarted")
            
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action = torch.argmax(model(state_tensor)).item()
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            step_count += 1 

            # Render the game
            env.render()
            time.sleep(0.01)  # Control the rendering speed

        print(f"Episode {episode + 1} finished, Total Reward: {total_reward}, Steps: {step_count}")

if __name__ == "__main__":
    env = RaceEnv(render_mode=True)  # Rendering enabled during play
    model = DQN(input_dim=16, output_dim=3).to(device)
    
    # Load trained model weights
    checkpoint = torch.load("trained_model.pth")
    model.load_state_dict(checkpoint)
    model.eval()

    play_game(env, model)

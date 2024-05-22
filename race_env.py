import gym
from gym import spaces
import numpy as np
import pygame
import random

class RaceEnv(gym.Env):
    def __init__(self, render_mode=False):
        super(RaceEnv, self).__init__()
        self.action_space = spaces.Discrete(3)  # 0 = go straight, 1 = turn left, 2 = turn right
        self.observation_space = spaces.Box(low=-1, high=1, shape=(16,), dtype=np.float32)  # 16-dimensional observation space

        # Pygame settings
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 30)
        self.score_font = pygame.font.SysFont('Arial', 40, bold=True)
        self.done = False
        self.render_mode = render_mode

        pygame.display.set_caption("AI Racer")

        # Car settings
        self.car_pos = [375, 500]  # Initial position of the player's car
        self.car_speed = 5
        self.car_image = pygame.image.load('car.png')
        self.car_image = pygame.transform.scale(self.car_image, (50, 75))  # Adjusted size

        # Road and obstacle settings
        self.road_left = 100
        self.road_right = 700
        self.obstacle_speed = 5
        self.obstacle_image = pygame.image.load('obstacle.png')
        self.obstacle_image = pygame.transform.scale(self.obstacle_image, (50, 75))  # Adjusted size
        self.score = 0

        # Tree settings
        self.tree_image = pygame.image.load('tree.png')
        self.tree_image = pygame.transform.scale(self.tree_image, (50, 75))  # Adjusted size
        self.trees_left = self.create_trees(self.road_left - 75, 0, 600, 100)
        self.trees_right = self.create_trees(self.road_right + 25, 0, 600, 100)

    def create_random_obstacles(self):
        # Define a random pattern for obstacles
        obstacles = []
        for _ in range(5):
            x_pos = random.randint(self.road_left, self.road_right - 50)
            y_pos = random.randint(-1000, -200)
            obstacles.append([x_pos, y_pos])
        return obstacles

    def create_trees(self, x_start, y_start, y_end, spacing):
        trees = []
        for y in range(y_start, y_end, spacing):
            trees.append([x_start, y])
        return trees

    def reset(self):
        self.car_pos = [375, 500]  # Initial position of the player's car
        self.done = False
        self.obstacles = self.create_random_obstacles()  # Use random obstacles
        self.score = 0  # Reset score
        self.state = self.get_state()  # Initial state
        return self.state

    def get_state(self):
        state = [
            self.car_pos[0] / 800,  # Car x position 
            self.car_pos[1] / 600,  # Car y position 
            (self.car_pos[0] - self.road_left) / 800,  # Distance to left road boundary 
            (self.road_right - self.car_pos[0]) / 800  # Distance to right road boundary 
        ]
        
        for i in range(6):  # Add positions of the nearest six obstacles
            if i < len(self.obstacles):
                state.append((self.obstacles[i][0] - self.car_pos[0]) / 800)  # Obstacle x position 
                state.append((self.obstacles[i][1] - self.car_pos[1]) / 600)  # Obstacle y position 
            else:
                state.append(0)  # No obstacle x position
                state.append(0)  # No obstacle y position

        return np.array(state, dtype=np.float32)

    def step(self, action):
        prev_y = self.car_pos[1]  # Save previous y position

        if action == 1:
            self.car_pos[0] -= self.car_speed  # Turn left
        elif action == 2:
            self.car_pos[0] += self.car_speed  # Turn right

        # Move obstacles
        for obs in self.obstacles:
            obs[1] += self.obstacle_speed

        # Generate new obstacles (random pattern)
       
        if self.obstacles and self.obstacles[0][1] > 600:
            self.obstacles.pop(0)
            new_obstacle = [random.randint(self.road_left, self.road_right - 50), random.randint(-1000, -200)]
            self.obstacles.append(new_obstacle)

        # Collision detection
        if self.car_pos[0] < self.road_left or self.car_pos[0] > self.road_right - 50:  
            self.done = True
            reward = -100  # Large negative reward for going off the road
            return self.get_state(), reward, self.done, {}

        for obstacle in self.obstacles:
            if self.check_collision(self.car_pos, obstacle):
                self.done = True
                reward = -100  # Large negative reward for collision
                return self.get_state(), reward, self.done, {}

        # Calculate distance from the center of the road
        road_center = (self.road_left + self.road_right) / 2
        distance_from_center = abs(self.car_pos[0] - road_center) / (self.road_right - self.road_left)

        # Reward shaping
        reward = 1 - distance_from_center  # Reward for staying close to the center
        if prev_y > self.car_pos[1]:
            reward += 10  # Extra reward for moving forward
        for obstacle in self.obstacles:
            if obstacle[1] > self.car_pos[1] and prev_y <= self.car_pos[1]:
                reward += 50  # Reward for avoiding obstacles

        self.score += 1  # Increase score every step

        info = {}  # Additional info
        self.state = self.get_state()  # New state

        # Visualization with Pygame
        if self.render_mode:
            self.render()

        return self.state, reward, self.done, info

    def check_collision(self, car_pos, obstacle_pos):
        car_rect = pygame.Rect(car_pos[0], car_pos[1], 50, 75)  # Adjusted for new car dimensions
        obstacle_rect = pygame.Rect(obstacle_pos[0], obstacle_pos[1], 50, 75)  # Adjusted for new obstacle dimensions
        return car_rect.colliderect(obstacle_rect)

    def render(self, mode='human'):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.screen.fill((0, 100, 0))  # Dark green background for grass
        # Draw the road
        pygame.draw.rect(self.screen, (50, 50, 50), pygame.Rect(self.road_left, 0, self.road_right - self.road_left, 800))
        # Draw road lanes
        lane_width = (self.road_right - self.road_left) // 3
        for lane in range(1, 3):
            lane_x = self.road_left + lane * lane_width
            for y in range(0, 800, 40):
                pygame.draw.line(self.screen, (255, 255, 0), (lane_x, y), (lane_x, y + 20), 4)  # Dashed lines in yellow
        # Draw road edges
        pygame.draw.line(self.screen, (255, 255, 255), (self.road_left, 0), (self.road_left, 800), 6)  # Left edge
        pygame.draw.line(self.screen, (255, 255, 255), (self.road_right, 0), (self.road_right, 800), 6)  # Right edge
        
        # Draw the trees
        for tree in self.trees_left:
            self.screen.blit(self.tree_image, tree)
        for tree in self.trees_right:
            self.screen.blit(self.tree_image, tree)
        
        # Draw the car
        self.screen.blit(self.car_image, self.car_pos)
        # Draw the obstacles
        for obstacle in self.obstacles:
            self.screen.blit(self.obstacle_image, obstacle)
        # Draw the score
        score_text = self.score_font.render(f'Score: {self.score}', True, (255, 255, 255))
        self.screen.blit(score_text, (310, 20))  # Centered at the top of the screen
        
        # Draw "Game Over" when the game is done
        if self.done and self.render_mode:
            game_over_text = self.font.render('Game Over', True, (255, 0, 0))
            self.screen.blit(game_over_text, (300, 250))
            self.clock.tick(300)
        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        pygame.quit()

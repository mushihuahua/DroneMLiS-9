import numpy as np
import random
import pygame
from drone import Drone
from flight_controller import FlightController
from pygame import Rect
import math


class CustomControllerVisual(FlightController):
    def __init__(self):
        self.q_table = np.zeros((100, 4))  # Example discretized state and action space
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.2  # Exploration rate
        self.state_space = 100
        self.action_space = 4
        self.drone = None

    def discretize_state(self, x, y, vx, vy, theta):
        # Normalize x and y to a fixed range (e.g., -1 to 1 -> 0 to 9)
        x_bin = int(np.clip((x + 1) * 5, 0, 9))  # Scale x from -1 to 1 into 0–9
        y_bin = int(np.clip((y + 1) * 5, 0, 9))  # Scale y from -1 to 1 into 0–9
        return x_bin + y_bin * 10

    def choose_action(self, state):
        """Choose an action using epsilon-greedy policy."""
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_space - 1)  # Explore
        return np.argmax(self.q_table[state])  # Exploit

    def get_thrusts(self, drone: Drone):
        """Calculate thrusts based on the current action."""
        state = self.discretize_state(drone.x, drone.y, drone.velocity_x, drone.velocity_y, drone.pitch)
        action = self.choose_action(state)

        left = drone.thrust_left
        right = drone.thrust_right

        if action == 0:
            left = min(left + 0.1, 1.0)  # Ensure thrust does not exceed 1.0
        elif action == 1:
            left = max(left - 0.1, 0.0)  # Ensure thrust does not go below 0.0
        elif action == 2:
            right = min(right + 0.1, 1.0)  # Ensure thrust does not exceed 1.0
        elif action == 3:
            right = max(right - 0.1, 0.0)  # Ensure thrust does not go below 0.0

        return left, right

    def train(self):
        """Train the drone with Q-learning and visualize the process."""
        pygame.init()
        clock = pygame.time.Clock()

        # Load graphics
        drone_img = pygame.image.load('graphics/drone_small.png')
        background_img = pygame.image.load('graphics/background.png')
        target_img = pygame.image.load('graphics/target.png')

        # Create the screen
        screen = pygame.display.set_mode((720, 480))

        episodes = 1000
        delta_time = self.get_time_interval()

        for episode in range(episodes):
            # Initialize the drone
            self.drone = self.init_drone()
            state = self.discretize_state(self.drone.x, self.drone.y, self.drone.velocity_x, self.drone.velocity_y, self.drone.pitch)
            total_reward = 0

            for step in range(self.get_max_simulation_steps()):
                # Handle Pygame events (to allow closing the window)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return  # Exit training

                # Choose and apply an action
                action = self.choose_action(state)
                left, right = self.get_thrusts(self.drone)
                self.drone.set_thrust((left, right))
                self.drone.step_simulation(delta_time)

                # Observe the new state and reward
                next_state = self.discretize_state(self.drone.x, self.drone.y, self.drone.velocity_x, self.drone.velocity_y, self.drone.pitch)
                reward = -np.sqrt((self.drone.x - self.drone.get_next_target()[0]) ** 2 +
                                  (self.drone.y - self.drone.get_next_target()[1]) ** 2)
                if self.drone.has_reached_target_last_update:
                    reward += 10000

                # Update Q-table
                self.q_table[state, action] += self.alpha * (
                    reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state, action]
                )

                # Transition to the next state
                state = next_state
                total_reward += reward

                # Visualization: Clear the screen and draw the updated state
                screen.blit(background_img, (0, 0))  # Draw background
                self.draw_drone(screen, self.drone, drone_img)  # Draw drone
                self.draw_target(screen, target_img, self.drone.get_next_target())  # Draw target
                pygame.display.flip()  # Update the display

                # Cap the frame rate to 60 FPS
                clock.tick(60)

                # Check if target is reached
                if self.drone.has_reached_target_last_update:
                    break

            print(f"Episode {episode + 1}, Total Reward: {total_reward}")

        pygame.quit()

    def draw_drone(self, screen, drone, drone_img):
        """Draw the drone on the screen."""
        scale = min(screen.get_height(), screen.get_width())

        def convert_to_screen_coordinates(x, y):
            return x * scale + screen.get_width() / 2, -y * scale + screen.get_height() / 2

        def convert_to_screen_size(game_size):
            return game_size * scale

        drone_x, drone_y = convert_to_screen_coordinates(drone.x, drone.y)
        drone_width = convert_to_screen_size(0.3)
        drone_height = convert_to_screen_size(0.15)

        drone_rect = Rect(drone_x - drone_width / 2, drone_y - drone_height / 2, drone_width, drone_height)
        drone_scaled_img = pygame.transform.scale(drone_img, (int(drone_width), int(drone_height)))
        drone_scaled_center = drone_scaled_img.get_rect(topleft=(drone_x - drone_width / 2, drone_y - drone_height / 2)).center
        rotated_drone_img = pygame.transform.rotate(drone_scaled_img, -drone.pitch * 180 / math.pi)
        drone_scaled_rect = rotated_drone_img.get_rect(center=drone_scaled_center)
        screen.blit(rotated_drone_img, drone_scaled_rect)

    def draw_target(self, screen, target_img, target_point):
        """Draw the target on the screen."""
        scale = min(screen.get_height(), screen.get_width())

        def convert_to_screen_coordinates(x, y):
            return x * scale + screen.get_width() / 2, -y * scale + screen.get_height() / 2

        def convert_to_screen_size(game_size):
            return game_size * scale

        target_size = convert_to_screen_size(0.1)
        target_x, target_y = convert_to_screen_coordinates(*target_point)
        target_scaled_img = pygame.transform.scale(target_img, (int(target_size), int(target_size)))
        screen.blit(target_scaled_img, (target_x - target_size / 2, target_y - target_size / 2))

import numpy as np
import pygame
import csv
import random
from drone import Drone
from typing import Tuple
from flight_controller import FlightController


class DroneReinforcementLearning:
    def __init__(self, q_table_path='q_table.csv'):
        self.drone = Drone()
        self.q_table = np.zeros((100, 4))  # Example discretized state and action space
        self.q_table_path = q_table_path  # File path for saving/loading Q-table
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.2  # Exploration rate
        self.state_space = 100
        self.action_space = 4
        self.drone.add_target_coordinate((0.35, 0.3))
        self.drone.add_target_coordinate((-0.35, 0.4))
        self.drone.add_target_coordinate((0.5, -0.4))
        self.drone.add_target_coordinate((-0.35, 0))

    def discretize_state(self, x, y, vx, vy, theta):
        # Normalize x and y to a fixed range (e.g., -1 to 1 -> 0 to 9)
        x_bin = int(np.clip((x + 1) * 5, 0, 9))  # Scale x from -1 to 1 into 0–9
        y_bin = int(np.clip((y + 1) * 5, 0, 9))  # Scale y from -1 to 1 into 0–9
        return x_bin + y_bin * 10

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_space - 1)  # Explore
        return np.argmax(self.q_table[state])  # Exploit

    def apply_action(self, action):
        left = self.drone.thrust_left
        right = self.drone.thrust_right    

        if action == 0:
            left = min(left + 0.1, 1.0)  # Ensure thrust does not exceed 1.0
        elif action == 1:
            left = max(left - 0.1, 0.0)  # Ensure thrust does not go below 0.0
        elif action == 2:
            right = min(right + 0.1, 1.0)  # Ensure thrust does not exceed 1.0
        elif action == 3:
            right = max(right - 0.1, 0.0)  # Ensure thrust does not go below 0.0

        self.drone.set_thrust((left, right))  

    def train(self, episodes, delta_time):
        for episode in range(episodes):
            self.drone.__init__()  # Reset drone
            state = self.discretize_state(self.drone.x, self.drone.y, self.drone.velocity_x, self.drone.velocity_y, self.drone.pitch)
            total_reward = 0
            previous_distance = np.sqrt((self.drone.x - self.drone.get_next_target()[0]) ** 2 +
                                        (self.drone.y - self.drone.get_next_target()[1]) ** 2)
            outofbounds = False

            for _ in range(1000):  # Max steps per episode
                action = self.choose_action(state)
                self.apply_action(action)
                self.drone.step_simulation(delta_time)

                current_distance = np.sqrt((self.drone.x - self.drone.get_next_target()[0]) ** 2 +
                                        (self.drone.y - self.drone.get_next_target()[1]) ** 2)

                reward = 0
                # Reward for reducing distance
                reward += previous_distance - current_distance

                # Large reward for reaching the target
                if self.drone.has_reached_target_last_update:
                    reward += 100

                # Penalty for large pitch or instability
             #   reward -= 0.1 * abs(self.drone.pitch)
            #    reward -= 0.1 * (self.drone.velocity_x**2 + self.drone.velocity_y**2)

                # Penalty for leaving boundaries
                if abs(self.drone.x) > 1 or abs(self.drone.y) > 1:
                    reward -= 50
                    outofbounds = True

                # Q-learning update
                next_state = self.discretize_state(self.drone.x, self.drone.y, self.drone.velocity_x, self.drone.velocity_y, self.drone.pitch)
                self.q_table[state, action] += self.alpha * (
                    reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state, action]
                )
                state = next_state
                previous_distance = current_distance
                total_reward += reward

                if self.drone.has_reached_target_last_update:
                    break
                elif outofbounds:
                    break

            print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    def save_q_table_csv(self):
        """Save the Q-table to a CSV file."""
        with open(self.q_table_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.q_table)
        print(f"Q-table saved to {self.q_table_path}")

    def load_q_table_csv(self):
        """Load the Q-table from a CSV file."""
        try:
            with open(self.q_table_path, mode='r') as file:
                reader = csv.reader(file)
                self.q_table = np.array([[float(value) for value in row] for row in reader])
            print(f"Q-table loaded from {self.q_table_path}")
        except FileNotFoundError:
            print(f"No saved Q-table found at {self.q_table_path}. Starting fresh.")


class CustomController(FlightController):
    def __init__(self):
        self.rl = DroneReinforcementLearning()  # Instantiate the reinforcement learning class

    def get_thrusts(self, drone: Drone) -> Tuple[float, float]:
        """Override to provide thrusts based on the reinforcement learning Q-table."""
        state = self.rl.discretize_state(drone.x, drone.y, drone.velocity_x, drone.velocity_y, drone.pitch)
        action = self.rl.choose_action(state)
        self.rl.drone = drone  # Update RL's drone instance to the current drone
        self.rl.apply_action(action)  # Apply the action to calculate thrusts
        return drone.thrust_left, drone.thrust_right

    def train(self):
        """Train the reinforcement learning model."""
        print("Starting training...")
        
        self.rl.train(episodes=1000, delta_time=self.get_time_interval())
        self.rl.save_q_table_csv()

    def load(self):
        """Load the saved Q-table."""
        self.rl.load_q_table_csv()

    def save(self):
        """Save the Q-table to a file."""
        self.rl.save_q_table_csv()

    def init_drone(self) -> Drone:
        """Initialize the drone with predefined targets."""
        drone = super().init_drone()
        drone.add_target_coordinate((0.35, 0.3))
        drone.add_target_coordinate((-0.35, 0.4))
        drone.add_target_coordinate((0.5, -0.4))
        drone.add_target_coordinate((-0.35, 0))
        return drone

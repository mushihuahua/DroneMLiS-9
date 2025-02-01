import numpy as np
from drone import Drone
import random

class DroneReinforcementLearning:
    def __init__(self):
        self.drone = Drone()
        self.q_table = np.zeros((100, 4))  # Example discretized state and action space
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.2  # Exploration rate
        self.state_space = 100
        self.action_space = 4
    
    def discretize_state(self, x, y, vx, vy, theta):
        # Simplified discretization for the state space
        return int((x % 10) + (y % 10) * 10)

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

            print(f"Episode {episode + 1}, Total Reward: {total_reward}")

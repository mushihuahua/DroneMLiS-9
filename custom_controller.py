from flight_controller import FlightController
from drone import Drone
from typing import Tuple

import numpy as np
import random
import csv

ACTIONS = [
    (0, +0.3), # Pitch right
    (0, -0.3), # Pitch left
    (+0.3, 0), # Thrust up
    (-0.3, 0)  # Thrust down
]


class CustomController(FlightController):

    def get_max_simulation_steps(self):
        return 3000
    
    def __discretize_state(self, dx, dy, pitch):
        dx_bin = int(np.clip((dx + 1) * 10, 0, 19))  # 20 bins for dx
        dy_bin = int(np.clip((dy + 1) * 10, 0, 19))  # 20 bins for dy
        pitch_bin = int(np.clip((pitch + 1) * 10, 0, 19))  # 20 bins for pitch

        return dx_bin + dy_bin * 20 + pitch_bin * 400    
    

    def __select_action(self, state: int, epsilon: float) -> int:
        action = 0
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, self.action_space - 1)  # Explore
        else:
            action = np.argmax(self.Qtable[state])  # Exploit
        
        return action


    def __init__(self):
        
        self.state_space = 20*20*20
        self.action_space = len(ACTIONS)

        self.Qtable = np.random.uniform(low=-0.1, high=0.1, size=(self.state_space, self.action_space))  
        self.Qtable_path = 'models/q_table.csv'  
        self.alpha = 0.1                                       # Learning rate
        self.gamma = 0.9                                        # Discount factor
        self.epsilon = 0.9                           # Exploration rate
        self.epsilon_decay_rate = 0.99995                           # epsilon decay rate
        self.minimum_epsilon = 0.001                           # minimum epsilon

        self.reward = 0

    def train(self):

        with open('evaluation/q_learning_results.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Episode', 'Total Reward', 'Steps', 'Epsilon'])

            for episode in range(2000):
                drone = self.init_train_drone()
                target = drone.get_next_target()

                dx = target[0] - drone.x
                dy = target[1] - drone.y

                state = self.__discretize_state(dx, dy, drone.pitch)
                action = 0

                total_reward = 0
                steps = 0

                for step in range(self.get_max_simulation_steps()):

                    crashed = False

                    dx = target[0] - drone.x
                    dy = target[1] - drone.y

                    action = self.__select_action(state, self.epsilon)
                    
                    left = np.clip(0.5 + ACTIONS[action][0] + ACTIONS[action][1] - drone.pitch , 0, 1)
                    right = np.clip(0.5 + ACTIONS[action][0] + -(ACTIONS[action][1] - drone.pitch), 0, 1)

                    drone.set_thrust((left, right))
                    drone.step_simulation(self.get_time_interval())

                    target = drone.get_next_target()
                    distance_to_target = np.sqrt(dy**2 + dx**2)

                    # REWARDS
                    
                    reward = - distance_to_target * 10

                    # # penalise it for too much pitch to keep it stabilised
                    # if abs(drone.pitch) > 0.4:  
                    #     reward -= 50 * abs(drone.pitch)

                    # heavy penalty for crashing
                    if abs(drone.x) >= 0.5 or abs(drone.y) >= 0.75:
                        reward -= 100  
                        crashed = True
                    
                    # some more reward when its close to the target
                    if distance_to_target < 0.2:
                        reward += 20  

                    # reward when it hits the target
                    if drone.has_reached_target_last_update:
                        reward += 400 

                    # update parameters according to algorithm
                    next_state = self.__discretize_state(dx, dy, drone.pitch)
                    self.Qtable[state, action] += self.alpha * (
                            reward + self.gamma * np.max(self.Qtable[next_state]) - self.Qtable[state, action]
                        )
                    
                    self.epsilon = max(self.minimum_epsilon, self.epsilon * self.epsilon_decay_rate)
                    
                    state = next_state
                    steps += 1
                    total_reward += reward

                    if(crashed):
                        break

                print(f"Episode {episode + 1} finished after {steps} steps. Reward: {total_reward}. Epsilon: {self.epsilon}")

                writer.writerow([episode+1, total_reward, steps, self.epsilon])
                    

    def get_thrusts(self, drone: Drone) -> Tuple[float, float]:

        target = drone.get_next_target()
        state = self.__discretize_state(target[0] - drone.x, target[1] - drone.y, drone.pitch)
        print(f"state: {state}, qtable: {self.Qtable[state]}")
        print(f"x: {drone.x}, y: {drone.y}, pitch {drone.pitch}")
        action = self.__select_action(state, epsilon=0.0001)

        left = np.clip(0.5 + ACTIONS[action][0] + ACTIONS[action][1] - drone.pitch , 0, 1)
        right = np.clip(0.5 + ACTIONS[action][0] + -(ACTIONS[action][1] - drone.pitch), 0, 1)

        dx = drone.get_next_target()[0] - drone.x
        dy = drone.get_next_target()[1] - drone.y

        print(np.sqrt(dx**2 + dy**2))

        return (left, right) 
        
    def init_train_drone(self) -> Drone:
        """Creates a Drone object initialised with a random set of target coordinates for training.

        Returns:
            Drone: An initial drone object with some programmed target coordinates.
        """
        drone = Drone()
        random.seed(42)
        drone.add_target_coordinate((random.uniform(-0.5, 0.5), random.uniform(-0.3, 0.3)))
        drone.add_target_coordinate((random.uniform(-0.5, 0.5), random.uniform(-0.3, 0.3)))
        drone.add_target_coordinate((random.uniform(-0.5, 0.5), random.uniform(-0.3, 0.3)))
        drone.add_target_coordinate((random.uniform(-0.5, 0.5), random.uniform(-0.3, 0.3)))
        drone.add_target_coordinate((random.uniform(-0.5, 0.5), random.uniform(-0.3, 0.3)))
        drone.add_target_coordinate((random.uniform(-0.5, 0.5), random.uniform(-0.3, 0.3)))
        drone.add_target_coordinate((random.uniform(-0.5, 0.5), random.uniform(-0.3, 0.3)))
        drone.add_target_coordinate((random.uniform(-0.5, 0.5), random.uniform(-0.3, 0.3)))
        random.seed()
        return drone

    def init_drone(self) -> Drone:
        """Creates a Drone object initialised with a deterministic set of target coordinates.

        Returns:
            Drone: An initial drone object with some programmed target coordinates.
        """
        drone = Drone()
        drone.add_target_coordinate((0.15, 0.1))
        drone.add_target_coordinate((0.4, 0.3))
        drone.add_target_coordinate((0, 0))


        drone.add_target_coordinate((-0.35, 0.4))
        drone.add_target_coordinate((0, 0))

        drone.add_target_coordinate((0.5, -0.4))
        drone.add_target_coordinate((-0.35, 0))
        return drone
    
    def save(self):
        """Save the Q-table to a CSV file."""
        with open(self.Qtable_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.Qtable)
        print(f"Q-table saved to {self.Qtable_path}")

        parameter_array = np.array([self.gamma, self.alpha, self.epsilon, self.epsilon_decay_rate, self.minimum_epsilon])
        np.save('models/custom_controller_parameters.npy', parameter_array)

        print(f"Parameters saved to custom_controller_parameters.npy")

    def load(self):
        """Load the Q-table and Parameters from files on disk."""

        try:
            parameter_array = np.load('models/custom_controller_parameters.npy')
            self.gamma = parameter_array[0]
            self.alpha = parameter_array[1]
            self.epsilon = parameter_array[2]
            self.epsilon_decay_rate = parameter_array[3]
            self.minimum_epsilon = parameter_array[3]
        except:
            print("Could not load parameters, sticking with default parameters.")

        try:
            with open(self.Qtable_path, mode='r') as file:
                reader = csv.reader(file)
                self.Qtable = np.array([[float(value) for value in row] for row in reader])
            print(f"Q-table loaded from {self.Qtable_path}")
        except FileNotFoundError:
            print(f"No saved Q-table found at {self.Qtable_path}.")

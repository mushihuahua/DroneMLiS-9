from flight_controller import FlightController
from drone import Drone
from typing import Tuple

import numpy as np
import random

ACTIONS = [
    (0.0, 0.0),
    (-0.1, +0.1), 
    (+0.1, -0.1),
    (+0.05, +0.05),
    (-0.05, -0.05)
]


class CustomController(FlightController):

    def get_max_simulation_steps(self):
        return 300000
    
    def __discretize_state(self, x, y, pitch):
        x_bin = int(np.clip((x + 1) * 5, 0, 9))  # 10 bins
        y_bin = int(np.clip((y + 1) * 5, 0, 9))  # 10 bins
        pitch_bin = int(np.clip((pitch + 1) * 5, 0, 9))  # 10 bins for pitch

        return x_bin + y_bin * 10 + pitch_bin * 100  


    def __init__(self):
        
        self.state_space = 1000
        self.action_space = len(ACTIONS)

        self.Qtable = np.random.uniform(low=-0.1, high=0.1, size=(self.state_space, self.action_space))  
        self.Qtable_path = 'q_table.csv'  
        self.alpha = 0.1                                       # Learning rate
        self.gamma = 0.9                                        # Discount factor
        self.epsilon = 0.7                               # Exploration rate

        self.reward = 0

    def train(self):

        for episode in range(100):
            drone = self.init_drone()
            state = self.__discretize_state(drone.x, drone.y, drone.pitch)

            action = 0

            for _ in range(self.get_max_simulation_steps()):

                reward = 0
                crashed = False

                if random.uniform(0, 1) < self.epsilon:
                    action = random.randint(0, self.action_space - 1)  # Explore
                else:
                    action = np.argmax(self.Qtable[state])  # Exploit
                
                left = np.clip(drone.thrust_left + ACTIONS[action][0], 0, 1)
                right = np.clip(drone.thrust_right + ACTIONS[action][1], 0, 1)

                drone.set_thrust((left, right))
                drone.step_simulation(self.get_time_interval())

                target = drone.get_next_target()
                distance_to_target = np.sqrt((drone.x - target[0]) ** 2 +
                            (drone.y - target[1]) ** 2)
                
                reward = -distance_to_target * 10

                if abs(drone.pitch) < 0.2:
                    reward += 10  

                if abs(drone.x) > 1 or abs(drone.y) > 1:
                    reward -= 1000  
                    crashed = True
                else:
                    reward += 10

                if np.abs(left - right) < 0.01:
                    reward -= 1  

                next_state = self.__discretize_state(drone.x, drone.y, drone.pitch)
                self.Qtable[state, action] += self.alpha * (
                        reward + self.gamma * np.max(self.Qtable[next_state]) - self.Qtable[state, action]
                    )
                
                self.epsilon = max(0.1, self.epsilon * 0.995)
                
                state = next_state

                if(crashed):
                    break

                print(f"position {drone.x}, {drone.y}")
                print(distance_to_target)


    def get_thrusts(self, drone: Drone) -> Tuple[float, float]:

        state = self.__discretize_state(drone.x, drone.y, drone.pitch)
        print(f"state: {state}, qtable: {self.Qtable[state]}")
        print(f"x: {drone.x}, y: {drone.y}, pitch {drone.pitch}")
        action = np.argmax(self.Qtable[state])

        left = np.clip(drone.thrust_left + ACTIONS[action][0], 0, 1)
        right = np.clip(drone.thrust_right + ACTIONS[action][1], 0, 1)


        return (left, right) 
    
    def load(self):
        pass
    def save(self):
        pass
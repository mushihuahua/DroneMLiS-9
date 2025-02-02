from flight_controller import FlightController
from drone import Drone
from typing import Tuple

import numpy as np
import random

ACTIONS = [
    # (0.0, 0.0),
    (0, +0.3), 
    (0, -0.3),
    (+0.3, 0),
    (-0.3, 0)
]


class CustomController(FlightController):

    def get_max_simulation_steps(self):
        return 3000
    
    def __discretize_state(self, dx, dy, pitch):
        dx_bin = int(np.clip((dx + 1) * 10, 0, 19))  # 20 bins for dx
        dy_bin = int(np.clip((dy + 1) * 10, 0, 19))  # 20 bins for dy
        pitch_bin = int(np.clip((pitch + 1) * 10, 0, 19))  # 20 bins for pitch

        return dx_bin + dy_bin * 20 + pitch_bin * 400    

    def __init__(self):
        
        self.state_space = 20*20*20
        self.action_space = len(ACTIONS)

        self.Qtable = np.random.uniform(low=-0.1, high=0.1, size=(self.state_space, self.action_space))  
        self.Qtable_path = 'q_table.csv'  
        self.alpha = 0.1                                       # Learning rate
        self.gamma = 0.9                                        # Discount factor
        self.epsilon = 0.8                            # Exploration rate

        self.reward = 0

    def train(self):

        for episode in range(2000):
            drone = self.init_drone()
            # drone.x, drone.y = random.uniform(-0.75, 0.75), random.uniform(-0.5, 0.5)
            target = drone.get_next_target()

            dx = target[0] - drone.x
            dy = target[1] - drone.y

            state = self.__discretize_state(dx, dy, drone.pitch)


            action = 0
            time_alive = 0

            for _ in range(self.get_max_simulation_steps()):

                time_alive += 1

                crashed = False

                dx = target[0] - drone.x
                dy = target[1] - drone.y

                if random.uniform(0, 1) < self.epsilon:
                    action = random.randint(0, self.action_space - 1)  # Explore
                else:
                    action = np.argmax(self.Qtable[state])  # Exploit
                
                left = np.clip(0.5 + ACTIONS[action][0] + ACTIONS[action][1] - drone.pitch , 0, 1)
                right = np.clip(0.5 + ACTIONS[action][0] + -(ACTIONS[action][1] - drone.pitch), 0, 1)

                drone.set_thrust((left, right))
                drone.step_simulation(self.get_time_interval())

                target = drone.get_next_target()
                distance_to_target = np.sqrt(dy**2 + dx**2)
                
                reward = - distance_to_target 

                if abs(drone.pitch) > 0.5:  
                    reward -= 50 * abs(drone.pitch)

                # reward -= 10 * (drone.pitch ** 2)

                # reward -=  10 * abs(left - right)

                if abs(drone.x) > 0.5 or abs(drone.y) > 0.75:
                    reward -= 100  
                    crashed = True
                # else:
                #     reward += time_alive / 10 

                if distance_to_target < 0.15:
                    reward += 100  # Encourage a confident final approach
                
                # reward += (drone.velocity_x * dx)
                # reward += (drone.velocity_y * dy)

                if drone.has_reached_target_last_update:
                    reward += 200

                next_state = self.__discretize_state(dx, dy, drone.pitch)
                self.Qtable[state, action] += self.alpha * (
                        reward + self.gamma * np.max(self.Qtable[next_state]) - self.Qtable[state, action]
                    )
                
                self.epsilon = max(0.01, self.epsilon * 0.995)
                
                state = next_state

                if(crashed):
                    break
                


                # print(f"position {drone.x}, {drone.y}")
                # print(distance_to_target)


    def get_thrusts(self, drone: Drone) -> Tuple[float, float]:

        target = drone.get_next_target()
        state = self.__discretize_state(target[0] - drone.x, target[1] - drone.y, drone.pitch)
        print(f"state: {state}, qtable: {self.Qtable[state]}")
        print(f"x: {drone.x}, y: {drone.y}, pitch {drone.pitch}")
        if random.uniform(0, 1) < 0.01:
            action = random.randint(0, self.action_space - 1)  # Explore
        else:
            action = np.argmax(self.Qtable[state])  # Exploit

        left = np.clip(0.5 + ACTIONS[action][0] + ACTIONS[action][1] - drone.pitch , 0, 1)
        right = np.clip(0.5 + ACTIONS[action][0] + -(ACTIONS[action][1] - drone.pitch), 0, 1)

        dx = drone.get_next_target()[0] - drone.x
        dy = drone.get_next_target()[1] - drone.y

        print(np.sqrt(dx**2 + dy**2))

        return (left, right) 
        
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

        drone.add_target_coordinate((0, 0.2))
        drone.add_target_coordinate((0, 0.4))
        drone.add_target_coordinate((0, -0.4))
        drone.add_target_coordinate((0, 0))
        drone.add_target_coordinate((0, -0.3))
        return drone
    
    def load(self):
        pass
    def save(self):
        pass
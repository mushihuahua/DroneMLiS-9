# rl_controller.py

import numpy as np
import math
import random
import pickle
from drone import Drone
from flight_controller import FlightController
from typing import Tuple

class RLController(FlightController):
    def __init__(self):
        # Q-table for state-action pairs: keys are (state, action) tuples
        self.q_table = {}
        # Hyperparameters for Q-learning
        self.epsilon = 0.1      # Exploration rate
        self.alpha = 0.5        # Learning rate
        self.gamma = 0.99       # Discount factor

        # Discretization: number of bins for each state variable
        self.num_bins = 10

        # Define a discrete set of actions. Each action is a tuple (thrust_left, thrust_right),
        # with each value in {0, 0.5, 1.0}.
        self.actions = [(left, right) 
                        for left in np.linspace(0, 1, 3)
                        for right in np.linspace(0, 1, 3)]

        # Variables to store the last state and action for Q-learning updates.
        self.last_state = None
        self.last_action = None

    def discretize_state(self, drone: Drone):
        """
        Convert the continuous state of the drone into a discrete tuple.
        This example uses the drone's x, y, x velocity, y velocity, and pitch.
        The min/max values here are assumptions and may need tuning.
        """
        # Define assumed min and max values for the state variables.
        x_min, x_max = -1.0, 1.0
        y_min, y_max = -1.0, 1.0
        vx_min, vx_max = -1.0, 1.0
        vy_min, vy_max = -1.0, 1.0
        pitch_min, pitch_max = -math.pi/4, math.pi/4

        def discretize(value, min_val, max_val):
            value = np.clip(value, min_val, max_val)
            scaled = (value - min_val) / (max_val - min_val)
            return int(scaled * (self.num_bins - 1))
        
        return (
            discretize(drone.x, x_min, x_max),
            discretize(drone.y, y_min, y_max),
            discretize(drone.velocity_x, vx_min, vx_max),
            discretize(drone.velocity_y, vy_min, vy_max),
            discretize(drone.pitch, pitch_min, pitch_max)
        )

    def get_q(self, state, action):
        """Return the Q-value for the (state, action) pair, defaulting to 0.0 if not present."""
        return self.q_table.get((state, action), 0.0)

    def set_q(self, state, action, value):
        """Set the Q-value for the (state, action) pair."""
        self.q_table[(state, action)] = value

    def choose_action(self, state):
        """Choose an action using an epsilon-greedy strategy."""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            # Evaluate all Q-values for the given state.
            qs = [self.get_q(state, a) for a in self.actions]
            max_q = max(qs)
            # If multiple actions have the same Q-value, choose one at random.
            best_actions = [a for a, q in zip(self.actions, qs) if q == max_q]
            return random.choice(best_actions)

    # --- FlightController Interface Methods ---

    def get_thrusts(self, drone: Drone) -> Tuple[float, float]:
        """
        This method is called at each simulation step.
        It discretizes the current state and then uses an epsilon-greedy policy to choose an action.
        The chosen action (thrusts) is returned.
        """
        state = self.discretize_state(drone)
        action = self.choose_action(state)
        # Store state and action to update Q-values later.
        self.last_state = state
        self.last_action = action
        return action

    def update_q_value(self, reward, new_state):
        """
        Update the Q-value for the last (state, action) pair using the Q-learning update rule:
            Q(s, a) = Q(s, a) + alpha * [reward + gamma * max_a' Q(s', a') - Q(s, a)]
        """
        best_future_q = max([self.get_q(new_state, a) for a in self.actions])
        old_q = self.get_q(self.last_state, self.last_action)
        new_q = old_q + self.alpha * (reward + self.gamma * best_future_q - old_q)
        self.set_q(self.last_state, self.last_action, new_q)

    def init_drone(self) -> Drone:
        """
        Initialise the drone for a new episode.
        This method sets up the drone and a predetermined set of target coordinates.
        """
        drone = Drone()
        drone.add_target_coordinate((0.35, 0.3))
        drone.add_target_coordinate((-0.35, 0.4))
        drone.add_target_coordinate((0.5, -0.4))
        drone.add_target_coordinate((-0.35, 0))
        return drone

    def get_max_simulation_steps(self) -> int:
        """Override the default maximum simulation steps."""
        return 500  # Or any value you prefer

    def get_time_interval(self) -> float:
        """Override the default time interval between simulation steps."""
        return 0.01  # Or any value you prefer

    def train(self):
        """
        The training loop runs for a number of episodes.
        In each episode, the RL controller interacts with the drone,
        updating its Q-table based on the observed rewards.
        """
        num_episodes = 1000  # Adjust the number of training episodes as needed

        for episode in range(num_episodes):
            drone = self.init_drone()
            total_reward = 0
            for step in range(self.get_max_simulation_steps()):
                # Select an action using the current policy.
                action = self.get_thrusts(drone)
                drone.set_thrust(action)
                # Step the simulation.
                drone.step_simulation(self.get_time_interval())
                # Observe the new (discretized) state.
                new_state = self.discretize_state(drone)
                # Compute reward (for example: negative distance to the target, plus a bonus if reached).
                reward = self.get_reward(drone)
                total_reward += reward
                # Update Q-value using the reward and new state.
                self.update_q_value(reward, new_state)
                # If the drone reached the target, end the episode early.
                if drone.has_reached_target_last_update:
                    break
            print(f"Episode {episode}: Total Reward = {total_reward}")

    def get_reward(self, drone: Drone) -> float:
        """
        Compute the reward for the current state.
        In this example:
          - The reward is the negative Euclidean distance from the drone to its current target.
          - An additional bonus is given if the target was reached.
        """
        target = drone.get_next_target()
        distance = math.sqrt((drone.x - target[0])**2 + (drone.y - target[1])**2)
        reward = -distance
        if drone.has_reached_target_last_update:
            reward += 1000  # Bonus reward for reaching the target.
        return reward

    def save(self):
        """Save the Q-table to disk."""
        with open("q_table.pkl", "wb") as f:
            pickle.dump(self.q_table, f)
        print("Q-table saved.")

    def load(self):
        """Load the Q-table from disk."""
        try:
            with open("q_table.pkl", "rb") as f:
                self.q_table = pickle.load(f)
            print("Q-table loaded.")
        except FileNotFoundError:
            print("No saved Q-table found. Starting with an empty Q-table.")

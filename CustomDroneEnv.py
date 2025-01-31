import numpy as np
from drone import Drone

class CustomDroneEnv:
    def __init__(self, max_steps=1000, delta_time=0.1, game_target_size=0.1):
        """
        Initializes the custom drone environment.

        Parameters:
        - max_steps: Maximum number of steps per episode.
        - delta_time: Time step for each simulation update.
        - game_target_size: Radius within which the target is considered reached.
        """
        self.max_steps = max_steps
        self.delta_time = delta_time
        self.game_target_size = game_target_size
        self.current_step = 0

        # Initialize the drone
        self.drone = Drone()
        self.action_space_size = 4  # Four discrete actions
        self.observation_space_size = 6  # [x, y, velocity_x, velocity_y, pitch, pitch_velocity]

    def reset(self):
        """
        Resets the environment to an initial state.

        Returns:
        - state: The initial state of the environment.
        """
        self.drone.__init__()  # Reinitialize the drone
        self.current_step = 0
        return self.get_state()

    def step(self, action):
        """
        Executes the given action in the environment.

        Parameters:
        - action: The action to take (e.g., thrust adjustments).

        Returns:
        - state: The next state of the environment.
        - reward: The reward for the action taken.
        - done: Whether the episode has ended.
        - info: Additional debugging information.
        """
        self.current_step += 1

        # Apply the action to the drone
        self.apply_action(action)
        self.drone.step_simulation(self.delta_time)

        # Calculate the reward
        target = self.drone.get_next_target()
        distance_to_target = np.sqrt((self.drone.x - target[0])**2 + (self.drone.y - target[1])**2)
        reward = -distance_to_target  # Negative distance encourages getting closer

        # Bonus reward for reaching the target
        done = False
        if self.drone.has_reached_target_last_update:
            reward += 100
            done = True
        # Penalty for going out of bounds
        elif abs(self.drone.x) > 1 or abs(self.drone.y) > 1:
            reward -= 50
            done = True
        # End the episode if max steps are reached
        elif self.current_step >= self.max_steps:
            done = True

        # Get the updated state
        state = self.get_state()
        return state, reward, done, {}

    def apply_action(self, action):
        """
        Applies the given action to the drone.

        Parameters:
        - action: The action to apply.
        """
        # Actions correspond to discrete thrust adjustments
        if action == 0:  # Increase left thrust
            self.drone.set_thrust((self.drone.thrust_left + 0.1, self.drone.thrust_right))
        elif action == 1:  # Decrease left thrust
            self.drone.set_thrust((self.drone.thrust_left - 0.1, self.drone.thrust_right))
        elif action == 2:  # Increase right thrust
            self.drone.set_thrust((self.drone.thrust_left, self.drone.thrust_right + 0.1))
        elif action == 3:  # Decrease right thrust
            self.drone.set_thrust((self.drone.thrust_left, self.drone.thrust_right - 0.1))

        # Clamp thrust values within [0, 1]
        left_thrust = np.clip(self.drone.thrust_left, 0, 1)
        right_thrust = np.clip(self.drone.thrust_right, 0, 1)
        self.drone.set_thrust((left_thrust, right_thrust))

    def get_state(self):
        """
        Retrieves the current state of the drone.

        Returns:
        - state: A numpy array representing the drone's state.
        """
        return np.array([
            self.drone.x,
            self.drone.y,
            self.drone.velocity_x,
            self.drone.velocity_y,
            self.drone.pitch,
            self.drone.pitch_velocity
        ])

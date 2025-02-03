import numpy as np
import random
from collections import deque
from typing import Tuple
from drone import Drone
from flight_controller import FlightController
# === Simple Perceptron Q-Network Using Numpy ===

class SimpleQNetwork:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        # Initialize weights with small random numbers.
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros((1, output_dim))
    
    def forward(self, x):
        """
        Forward pass through the network.
        Returns:
            q_values: The output Q-values.
            cache: A dictionary with intermediate values needed for backprop.
        """
        # First layer: linear
        z1 = x.dot(self.W1) + self.b1
        # ReLU activation
        h1 = np.maximum(0, z1)
        # Second layer: linear
        q_values = h1.dot(self.W2) + self.b2

        cache = {
            'x': x,
            'z1': z1,
            'h1': h1
        }
        return q_values, cache

    def backward(self, dloss, cache, lr: float):
        """
        Backward pass to update network parameters using gradients from the loss.
        dloss: Gradient of the loss with respect to network output (shape: [, output_dim]).
        cache: Cached values from the forward pass.
        lr: Learning rate.
        x is the network's input.
        z1 is the result of the first linear layer (before activation).
        h1 is the activated output from the first layer.
        """
        x = cache['x']
        z1 = cache['z1']
        h1 = cache['h1']

        # Gradients for second layer
        dW2 = h1.T.dot(dloss)
        db2 = np.sum(dloss, axis=0, keepdims=True)

        # Backprop through second layer to hidden layer
        dh1 = dloss.dot(self.W2.T)
        # Backprop through ReLU
        dz1 = dh1 * (z1 > 0)

        # Gradients for first layer
        dW1 = x.T.dot(dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Update weights and biases (simple gradient descent)
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def copy_parameters_from(self, other_network):
        """Copy parameters from another SimpleQNetwork instance."""
        self.W1 = np.copy(other_network.W1)
        self.b1 = np.copy(other_network.b1)
        self.W2 = np.copy(other_network.W2)
        self.b2 = np.copy(other_network.b2)

# === DQN Agent Using the Simple Perceptron ===

class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 32,
        lr: float = 0.01,
        gamma: float = 0.99,
        epsilon: float = 1,
        epsilon_min: float = 0.1,
        epsilon_decay: float = 0.999,
        target_update_freq: int = 50
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.target_update_freq = target_update_freq
        self.total_steps = 0

        # Initialize the online network and the target network.
        self.online_net = SimpleQNetwork(state_dim, hidden_dim, action_dim)
        self.target_net = SimpleQNetwork(state_dim, hidden_dim, action_dim)
        self.target_net.copy_parameters_from(self.online_net)
    
    def select_action(self, state: np.ndarray) -> int:
        """Select an action using the ε-greedy policy."""
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            # Forward pass through online network.
            state = state.reshape(1, -1)
            q_values, _ = self.online_net.forward(state)
            return int(np.argmax(q_values))
    

    
    def train_step(self, state, action, reward, next_state, done):
        
        # Compute Q-values for current states.
        state, next_state = state.reshape(1, -1), next_state.reshape(1, -1)
        q_values, cache = self.online_net.forward(state)
        q_next, _ = self.target_net.forward(next_state)
        max_next_q = np.max(q_next, axis=1)
        target_q = np.copy(q_values)
        
        # Compute Q-values for next states from target network.
        q_next, _ = self.target_net.forward(next_state)  
        max_next_q = np.max(q_next, axis=1)  
        
        # Compute the target Q-values.
        # If done, target = reward, else = reward + gamma * max_next_q
        target_q = np.copy(q_values)
        target = reward if done else reward + self.gamma * max_next_q
        target_q[0, action] = target
        # Compute loss and its gradient.
        # Using Mean Squared Error (MSE) loss.
        loss = np.mean((q_values - target_q) ** 2)
        # Gradient of MSE loss: 2 * (prediction - target) 
        grad_loss = 2 * (q_values - target_q) 
        
        # Backpropagate the gradient through the online network.
        self.online_net.backward(grad_loss, cache, self.lr)
        
        # Decay epsilon.
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update target network parameters periodically.
        self.total_steps += 1
        if self.total_steps % self.target_update_freq == 0:
            self.target_net.copy_parameters_from(self.online_net)
        
        return loss  # Optionally return the loss for logging.

# === Integration with the Drone Environment ===

# Assuming you have a Drone class and a FlightController base class, and you have defined discrete ACTIONS.

ACTIONS = [
    (0, +0.3),
    (0, -0.3),
    (+0.3, 0),
    (-0.3, 0)
]

# Here we create a custom flight controller that uses our DQN agent.
class CustomControllerv2(FlightController):
    def __init__(self):
        # The state is defined as [dx, dy, pitch]
        state_dim = 3
        action_dim = len(ACTIONS)
        self.agent = DQNAgent(state_dim, action_dim)
        self.max_simulation_steps = 3000

    def _get_state(self, drone, target: Tuple[float, float]) -> np.ndarray:
        """
        Compute the state vector from the drone and target.
        For example, state = [target_x - drone.x, target_y - drone.y, drone.pitch]
        """
        dx = target[0] - drone.x
        dy = target[1] - drone.y
        pitch = drone.pitch
        return np.array([dx, dy, pitch])

    def train(self, num_episodes: int = 100):
        for episode in range(num_episodes):
            drone = self.init_drone()
            target = drone.get_next_target()
            state = self._get_state(drone, target)
            prev_distance = 0
            reward = 0
            for step in range(self.max_simulation_steps):
                # Select action using the agent.
                action_index = self.agent.select_action(state)
                thrust_input = ACTIONS[action_index]
                
                # Compute thrust values (this is just one way to combine the action with pitch).
                left = np.clip(0.5 + thrust_input[0] + thrust_input[1] - drone.pitch, 0, 1)
                right = np.clip(0.5 + thrust_input[0] - (thrust_input[1] - drone.pitch), 0, 1)
                
                drone.set_thrust((left, right))
                drone.step_simulation(self.get_time_interval())
                
                # Get updated state.
                target = drone.get_next_target()
                next_state = self._get_state(drone, target)
                
                # Compute a simple reward.
                dx = target[0] - drone.x
                dy = target[1] - drone.y
                distance_to_target = np.sqrt(dx**2 + dy**2)
               # reward = -distance_to_target

                done = False
                # If drone goes out of bounds, penalize and mark episode as done.
                if abs(drone.x) > 0.5 or abs(drone.y) > 0.75:
                    reward -= 1000
                    done = True
                # Bonus for reaching the target.
                #if distance_to_target < 0.15:
                #    reward += 10
                if drone.has_reached_target_last_update:
                    reward += 2000
                reward += 0.01 * step  # Small penalty per step

                if distance_to_target < prev_distance:  
                    reward += 10  # Bonus for moving in the right direction
                else:
                    reward -= 5  # Penalty for moving away
                prev_distance = distance_to_target
               # print(f"reward {reward}")

               # loss = self.agent.train_step()  # Optionally log the loss.
                
                state = next_state
                
                if done:
                    break
            
            print(f"Episode {episode + 1} finished after {step + 1} steps.")

    def save(self, filename='model_weights.npy'):
        # Example: if you want to save the Q-table
        return
        np.save(filename, self.Qtable)
        print(f"Model saved to {filename}")

    def load(self, filename='model_weights.npy'):
        self.Qtable = np.load(filename)
        print(f"Model loaded from {filename}")

    def get_thrusts(self, drone) -> Tuple[float, float]:
        target = drone.get_next_target()
        state = self._get_state(drone, target)
        action_index = self.agent.select_action(state)

        thrust_input = ACTIONS[action_index]
        left = np.clip(0.5 + thrust_input[0] + thrust_input[1] - drone.pitch, 0, 1)
        right = np.clip(0.5 + thrust_input[0] - (thrust_input[1] - drone.pitch), 0, 1)
        return (left, right)
    
    def init_train_drone(self) -> Drone:
        """Creates a Drone object initialised with a random set of target coordinates for training.

        Returns:
            Drone: An initial drone object with some programmed target coordinates.
        """
        drone = Drone()
        random.seed(42)
        drone.add_target_coordinate((random.uniform(-0.75, 0.75), random.uniform(-0.5, 0.5)))
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
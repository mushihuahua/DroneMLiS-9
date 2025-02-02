import numpy as np
import random
import pickle
from collections import deque
from drone import Drone
from flight_controller import FlightController
import time

# --- Simple Feedforward Neural Network (Manual Backpropagation) ---
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights randomly
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.1  # Input -> Hidden
        self.b1 = np.zeros((1, self.hidden_size))  # Bias for hidden layer
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.1  # Hidden -> Output
        self.b2 = np.zeros((1, self.output_size))  # Bias for output layer
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, x):
        """Compute forward pass and return activations"""
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.relu(self.z1)  # Activation function
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2  # No activation for output (Q-values)

    def backward(self, x, target, output):
        """Compute backward pass (gradient descent)"""
        epsilon = 1e-8  # Small constant to prevent division by zero

        if np.isnan(output).any() or np.isnan(target).any():
            print("NaN detected in output or target")
            target = np.nan_to_num(target)  # Replace NaNs with 0
            output = np.nan_to_num(output)  # Ensure output is not NaN

        error = output - target  # Compute error (loss gradient)

        # Backpropagation
        dW2 = np.dot(self.a1.T, error) / (np.linalg.norm(self.a1.T) + epsilon)  # Gradient for W2
        db2 = np.sum(error, axis=0, keepdims=True) / (np.linalg.norm(error) + epsilon)
        
        d_hidden = (np.dot(error, self.W2.T) * self.relu_derivative(self.z1)) / (np.linalg.norm(self.W2.T) + epsilon)
        dW1 = np.dot(x.T, d_hidden) / (np.linalg.norm(x.T) + epsilon)  # Gradient for W1
        db1 = np.sum(d_hidden, axis=0, keepdims=True) / (np.linalg.norm(d_hidden) + epsilon)

        # Update weights with learning rate
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
    

    
    def train(self, x, target):
        """Perform one training step"""
        output = self.forward(x)
        self.backward(x, target, output)

    def predict(self, x):
        """Predict Q-values for a given state"""
        return self.forward(x)

class DeepQController(FlightController):
    def __init__(self):
        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.05  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay rate for epsilon
        self.learning_rate = 0.01
        self.batch_size = 32
        self.replay_capacity = 10000
        self.testRewards = True
        self.prev_distance = 0
        # Define the state size: here we use 5 features (x, y, vx, vy, pitch)
        self.state_size = 5
        
        # Define action space
        self.action_list = [(left, right) for left in np.linspace(0, 1, 5) for right in np.linspace(0, 1, 5)]
        self.action_size = len(self.action_list)
        
        # Neural Network for Q-value approximation
        self.q_network = SimpleNeuralNetwork(self.state_size, 64, self.action_size, self.learning_rate)
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=self.replay_capacity)
    
    def preprocess_state(self, drone: Drone):
        """Normalize the drone's state"""
        return np.array([
            np.clip((drone.x + 1.0) / 2.0, 0, 1),
            np.clip((drone.y + 1.0) / 2.0, 0, 1),
            np.clip((drone.velocity_x + 1.0) / 2.0, 0, 1),
            np.clip((drone.velocity_y + 1.0) / 2.0, 0, 1),
            np.clip((drone.pitch + (np.pi / 4)) / (np.pi / 2), 0, 1)
        ])
    
    def choose_action(self, state):
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_values = self.q_network.predict(state.reshape(1, -1))
            return np.argmax(q_values)

    def get_thrusts(self, drone: Drone):
        """Called each simulation step, choosing an action"""
        state = self.preprocess_state(drone)
        action_index = self.choose_action(state)
        return self.action_list[action_index]

    def update_network(self):
        """Sample a minibatch and perform a gradient descent step"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = np.array(states)
        next_states = np.array(next_states)
        
        q_values_next = self.q_network.predict(next_states)
        max_q_next = np.max(q_values_next, axis=1)
        
        target_q_values = self.q_network.predict(states)
        for i in range(self.batch_size):
            target_q_values[i, actions[i]] = rewards[i] + (self.gamma * max_q_next[i] * (1 - dones[i]))
        
        self.q_network.train(states, target_q_values)

    def train(self):
        """Run training loop"""
        num_episodes = 500
        
        for episode in range(num_episodes):
            drone = self.init_drone()
            total_reward = 0
            
            for step in range(self.get_max_simulation_steps()):
                state = self.preprocess_state(drone)
                action_index = self.choose_action(state)
                action = self.action_list[action_index]
                
                drone.set_thrust(action)
                drone.step_simulation(self.get_time_interval())
                
                next_state = self.preprocess_state(drone)
                reward = -np.linalg.norm(np.array(drone.get_next_target()) - np.array([drone.x, drone.y]))

                done = drone.has_reached_target_last_update
                
                if self.testRewards:
                    target = np.array(drone.get_next_target())
                    position = np.array([drone.x, drone.y])
                    distance = np.linalg.norm(target - position)
                    reward = self.previous_distance - distance  
                    self.previous_distance = distance


                if drone.has_reached_target_last_update:
                     reward += 100  # Big reward for success
                
                self.replay_buffer.append((state, action_index, reward, next_state, done))
                self.update_network()
                
                total_reward += reward
                if done:
                    break
            
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            print(f"Episode {episode}: Reward = {total_reward}, Epsilon = {self.epsilon}")
        
        self.save()

    def save(self):
        """Save model weights"""
        with open("q_network.pkl", "wb") as f:
            pickle.dump(self.q_network, f)

    def load(self):
        """Load model weights"""
        try:
            with open("q_network.pkl", "rb") as f:
                self.q_network = pickle.load(f)
            print("Model loaded.")
        except FileNotFoundError:
            print("No saved model found.")

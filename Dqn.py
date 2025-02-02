import numpy as np
import random
from drone import Drone
# Define the neural network
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.learning_rate = learning_rate
        # He initialization for ReLU activation
        self.weights1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
        self.bias2 = np.zeros((1, output_size))
    
    def forward(self, state):
        # Ensure state is at least 2D (for a single sample, it becomes (1, input_size))
        state = np.atleast_2d(state)
        self.z1 = np.dot(state, self.weights1) + self.bias1
        self.a1 = np.maximum(0, self.z1)  # ReLU activation
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        return self.z2  # Linear output
    
    def backward(self, state, target, output):
        error = output - target
        d_weights2 = np.dot(self.a1.T, error)
        d_bias2 = np.sum(error, axis=0, keepdims=True)
        
        hidden_error = np.dot(error, self.weights2.T)
        # Apply derivative of ReLU: set gradient to zero where z1 was negative
        hidden_error[self.z1 <= 0] = 0
        
        d_weights1 = np.dot(state.T, hidden_error)
        d_bias1 = np.sum(hidden_error, axis=0, keepdims=True)
        
        self.weights2 -= self.learning_rate * d_weights2
        self.bias2 -= self.learning_rate * d_bias2
        self.weights1 -= self.learning_rate * d_weights1
        self.bias1 -= self.learning_rate * d_bias1

# Deep Q-Learning Agent
# DQNAgent remains mostly unchanged except that we now assume that states stored in memory
# are 1D arrays (of shape (state_dim,)), not 2D arrays.
class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_size=8, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.q_network = NeuralNetwork(state_dim, hidden_size, action_dim)
        self.target_network = NeuralNetwork(state_dim, hidden_size, action_dim)
        self.update_target_network()
        
        self.memory = []  # Replay memory
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:
            self.memory.pop(0)
    
    def act(self, state):
        # state is now a 1D array of shape (state_dim,)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        q_values = self.q_network.forward(state)  # forward converts it to 2D
        return np.argmax(q_values)
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # Convert list of 1D arrays to a 2D array of shape (batch_size, state_dim)
        states = np.array(states)
        next_states = np.array(next_states)
        
        q_values_next = self.target_network.forward(next_states)
        q_values_target = self.q_network.forward(states)
        q_values_target = q_values_target.reshape(-1, self.action_dim)
        q_values_next = q_values_next.reshape(-1, self.action_dim)

        for i in range(batch_size):
            if dones[i]:
                q_values_target[i, actions[i]] = rewards[i]
            else:
                q_values_target[i, actions[i]] = rewards[i] + self.gamma * np.max(q_values_next[i])
        
        outputs = self.q_network.forward(states)
        self.q_network.backward(states, q_values_target, outputs)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        self.target_network.weights1 = np.copy(self.q_network.weights1)
        self.target_network.bias1 = np.copy(self.q_network.bias1)
        self.target_network.weights2 = np.copy(self.q_network.weights2)
        self.target_network.bias2 = np.copy(self.q_network.bias2)

# Custom Drone Environment
class CustomDroneEnv:
    def __init__(self, max_steps=1000, delta_time=0.1, game_target_size=0.1):
        self.max_steps = max_steps
        self.delta_time = delta_time
        self.game_target_size = game_target_size
        self.current_step = 0
        
        self.drone = Drone()
        self.action_space_size = 4
        self.observation_space_size = 6
    
    def reset(self):
        self.drone.__init__()
        self.current_step = 0
        return self.get_state()
    
    def step(self, action):
        self.current_step += 1
        self.apply_action(action)
        self.drone.step_simulation(self.delta_time)
        
        target = self.drone.get_next_target()
        distance_to_target = np.sqrt((self.drone.x - target[0])**2 + (self.drone.y - target[1])**2)
        reward = -distance_to_target
        
        done = False
        if self.drone.has_reached_target_last_update:
            reward += 1000
            done = True
        elif abs(self.drone.x) > 1 or abs(self.drone.y) > 1:
            reward -= 50
            done = True
        elif self.current_step >= self.max_steps:
            done = True
        
        return self.get_state(), reward, done, {}
    
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
    
    def get_state(self):
        return np.array([
            self.drone.x,
            self.drone.y,
            self.drone.velocity_x,
            self.drone.velocity_y,
            self.drone.pitch,
            self.drone.pitch_velocity
        ])

# Main training loop
if __name__ == "__main__":
    # Create environment, agent, etc.
    env = CustomDroneEnv()  # Assuming CustomDroneEnv is defined elsewhere
    state_dim = env.observation_space_size
    action_dim = env.action_space_size
    
    agent = DQNAgent(state_dim, action_dim)
    episodes = 500
    
    for episode in range(episodes):
        # Get state as a 1D array (env.reset() returns a 1D np.array)
        state = env.reset()  
        total_reward = 0
        
        for t in range(200):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            # Store states as 1D arrays without reshaping to (1, state_dim)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:
                print(f"Episode {episode + 1}: Total Reward = {total_reward}")
                break
        
        agent.replay(batch_size=64)
        if episode % 10 == 0:
            agent.update_target_network()
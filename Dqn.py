import numpy as np
import random

# Define the neural network
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # Initialize weights and biases
        self.learning_rate = learning_rate
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.01
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.01
        self.bias2 = np.zeros((1, output_size))
    
    def forward(self, state):
        # Forward pass
        self.z1 = np.dot(state, self.weights1) + self.bias1
        self.a1 = np.maximum(0, self.z1)  # ReLU activation
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        return self.z2  # Linear output
    
    def backward(self, state, target, output):
        # Compute loss gradient
        error = output - target
        d_weights2 = np.dot(self.a1.T, error)
        d_bias2 = np.sum(error, axis=0, keepdims=True)
        
        hidden_error = np.dot(error, self.weights2.T)
        hidden_error[self.z1 <= 0] = 0  # Derivative of ReLU
        
        d_weights1 = np.dot(state.T, hidden_error)
        d_bias1 = np.sum(hidden_error, axis=0, keepdims=True)
        
        # Update weights and biases
        self.weights2 -= self.learning_rate * d_weights2
        self.bias2 -= self.learning_rate * d_bias2
        self.weights1 -= self.learning_rate * d_weights1
        self.bias1 -= self.learning_rate * d_bias1

# Deep Q-Learning Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_size=64, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
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
        if len(self.memory) > 10000:  # Limit memory size
            self.memory.pop(0)
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)  # Explore
        q_values = self.q_network.forward(state)
        return np.argmax(q_values)  # Exploit
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states)
        next_states = np.array(next_states)
        
        # Compute target Q-values
        q_values_next = self.target_network.forward(next_states)
        q_values_target = self.q_network.forward(states)
        for i in range(batch_size):
            if dones[i]:
                q_values_target[i, actions[i]] = rewards[i]
            else:
                q_values_target[i, actions[i]] = rewards[i] + self.gamma * np.max(q_values_next[i])
        
        # Train the Q-network
        outputs = self.q_network.forward(states)
        self.q_network.backward(states, q_values_target, outputs)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        # Copy weights from Q-network to target network
        self.target_network.weights1 = np.copy(self.q_network.weights1)
        self.target_network.bias1 = np.copy(self.q_network.bias1)
        self.target_network.weights2 = np.copy(self.q_network.weights2)
        self.target_network.bias2 = np.copy(self.q_network.bias2)

# Example Usage
if __name__ == "__main__":
    env = ...  # Your environment (e.g., gym.make("CartPole-v1"))
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim)
    episodes = 500
    
    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(state, (1, state_dim))
        total_reward = 0
        
        for t in range(200):  # Max steps per episode
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, (1, state_dim))
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:
                print(f"Episode {episode + 1}: Total Reward = {total_reward}")
                break
        
        agent.replay(batch_size=64)
        if episode % 10 == 0:  # Update target network periodically
            agent.update_target_network()
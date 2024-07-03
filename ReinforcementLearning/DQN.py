import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

# Initialize the gym environment
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Hyperparameters
learning_rate = 0.001
gamma = 0.95  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995
epsilon_min = 0.01
batch_size = 64
memory = deque(maxlen=2000)

# Build the DQN model
model = Sequential()
model.add(Flatten(input_shape=(1, state_size)))
model.add(Dense(24, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(action_size, activation='linear'))
model.compile(loss='mse', optimizer=Adam(lr=learning_rate))

# Function to replay experiences and train the model
def replay(memory, batch_size):
    minibatch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target = (reward + gamma * np.amax(model.predict(next_state)[0]))
        target_f = model.predict(state)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1, verbose=0)

# Train the DQN model
episodes = 1000
for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, 1, state_size])
    for time in range(500):
        if np.random.rand() <= epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(model.predict(state)[0])  # Exploit learned values
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, 1, state_size])
        memory.append((state, action, reward, next_state, done))
        state = next_state
        if done:
            print(f"episode: {e}/{episodes}, score: {time}, e: {epsilon:.2}")
            break
        if len(memory) > batch_size:
            replay(memory, batch_size)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
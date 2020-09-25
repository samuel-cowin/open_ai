from collections import deque
import gym
from tensorflow import keras
import numpy as np
import dqn_utils
import matplotlib.pyplot as plt

# Define OpenAI environment
env = gym.make('CartPole-v0')

# Define the hyperameters for this implementation
batch_size = 32
gamma = 0.9
min_epsilon = 0.01
n_neurons = 24
n_episodes = 1000
n_steps = 200

# Create the Neural Network Model
nn_model = dqn_utils.build_policy_nn(n_neurons=n_neurons, n_inputs=env.observation_space.shape[0], n_outputs=env.action_space.n)

# Initialize replay_buffer for sample efficiency
replay_buffer = deque(maxlen=2000)

# Implement the training loop
rewards_per = []
for e in range(n_episodes):
    o = env.reset()
    rew = 0
    for s in range(n_steps):
        # Define the gradient optimization and loss
        lr = 0.01 / (1 + e/n_steps)
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        loss_fn = keras.losses.mean_squared_error
        epsilon = min_epsilon + (1.0-min_epsilon)*(1.0-(e/n_episodes))
        o, r, done, _ = dqn_utils.dqn_step(env, o, nn_model, replay_buffer, epsilon)
        rew += r
        if done:
            break
    if e > 50:
        dqn_utils.dqn(batch_size, replay_buffer, nn_model, loss_fn, optimizer, n_outputs=env.action_space.n, gamma=gamma)
    if e % 50 == 0 and e > 0:
        print('Completed {} episodes.'.format(e))
    rewards_per.append(rew)

print('Maximum reward achieved was {}.'.format(np.max(rewards_per)))
plt.plot(range(n_episodes), rewards_per)
plt.show()
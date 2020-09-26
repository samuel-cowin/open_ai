import gym
from tensorflow import keras
import numpy as np
from dqn import run_dqn_training


# Define OpenAI environment
env = gym.make('CartPole-v0')

# Loop through hyperparameters of choosing
env.seed(42)
np.random.seed(42)
keras.backend.clear_session()
n_neurons = [30]
rewards = [run_dqn_training(env, n_neurons=n, n_episodes=2000, diag=True) for n in n_neurons]
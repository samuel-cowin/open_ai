import gym
from tensorflow import keras
import actor_critic_utils

# Define the hyperameters for this implementation
n_iterations = 2000
n_test = 20
n_neurons = 5
gamma = 0.95
alpha=0.1

# Define the gradient optimization and loss
optimizer = keras.optimizers.Adam(learning_rate=0.001)
loss = keras.losses.binary_crossentropy

# Define OpenAI environment
env = gym.make('CartPole-v1')

# Create the Neural Network Model
nn_model = actor_critic_utils.build_policy_nn(n_neurons=n_neurons)

# Implement the training loop
actor_critic_utils.ac_gradient(env=env, nn_model=nn_model, loss_fn=loss, optimizer=optimizer, gamma=gamma, alpha=alpha, n_iterations=n_iterations, render=True)

# input('Start simulation? (Enter to continue)')
# Show the results of the training

#     for r in all_rewards:
#         for w in r:
#             total_reward+=w

# print('Average reward during test per episode was {}.'.format((total_reward/n_test)))
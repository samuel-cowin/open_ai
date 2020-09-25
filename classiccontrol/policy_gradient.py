import gym
import tensorflow as tf
from tensorflow import keras
import policy_gradient_utils

# Define the hyperameters for this implementation
n_iterations = 150
n_test = 20
n_episodes = 10
n_max = 200
n_neurons = 5
gamma = 0.95

# Define the gradient optimization and loss
optimizer = keras.optimizers.Adam(learning_rate=0.001)
loss = keras.losses.binary_crossentropy

# Define OpenAI environment
env = gym.make('CartPole-v1')

# Create the Neural Network Model
nn_model = policy_gradient_utils.build_policy_nn(n_neurons=n_neurons)

# Implement the training loop
for i in range(n_iterations):
    all_rewards, all_grads = policy_gradient_utils.policy_gradient(
        env, nn_model, loss, n_ep=n_episodes, max_steps=n_max)
    all_final_rewards = policy_gradient_utils.discount_and_normalize_episodes(
        all_rewards, gamma)
    all_mean_gradients = []
    for var_i in range(len(nn_model.trainable_variables)):
        mean_grads = tf.reduce_mean([final_reward * all_grads[ep_i][step][var_i] for ep_i, final_rewards in enumerate(
            all_final_rewards) for step, final_reward in enumerate(final_rewards)], axis=0)
        all_mean_gradients.append(mean_grads)
    optimizer.apply_gradients(
        zip(all_mean_gradients, nn_model.trainable_variables))

input('Start simulation? (Enter to continue)')
# Show the results of the training
total_reward = 0
for i in range(n_test):
    all_rewards, all_grads = policy_gradient_utils.policy_gradient(
        env, nn_model, loss, n_ep=1, max_steps=n_max, render=True)
    all_final_rewards = policy_gradient_utils.discount_and_normalize_episodes(
        all_rewards, gamma)
    all_mean_gradients = []
    for var_i in range(len(nn_model.trainable_variables)):
        mean_grads = tf.reduce_mean([final_reward * all_grads[ep_i][step][var_i] for ep_i, final_rewards in enumerate(
            all_final_rewards) for step, final_reward in enumerate(final_rewards)], axis=0)
        all_mean_gradients.append(mean_grads)
    optimizer.apply_gradients(
        zip(all_mean_gradients, nn_model.trainable_variables))
    for r in all_rewards:
        for w in r:
            total_reward+=w

print('Average reward during test per episode was {}.'.format((total_reward/n_test)))
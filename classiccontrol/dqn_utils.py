"""
Methods to implement Deep Q-Networks for ClassicControl in OpenAI Gym
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np


def build_policy_nn(n_neurons=32, n_inputs=4, n_outputs=2, activation_dense='elu'):
    """
    Basic Neural Network for taking the observation and outputting the probability 
    of the action to move left or right
    """

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(
        n_neurons, activation=activation_dense, input_shape=[n_inputs]))
    model.add(keras.layers.Dense(n_neurons, activation=activation_dense))
    model.add(keras.layers.Dense(n_outputs))
    return model

def epsilon_greedy(s, model, num_a=2, epsilon=0.01):
    """
    Returns the action that is either the best of the target values or 
    a random action taken with epsilon probability
    """

    rand_a = np.random.randint(num_a)
    q_values = model.predict(s[np.newaxis])
    best = np.argmax(q_values[0])
    return np.random.choice([best, rand_a], p=[1-epsilon, epsilon])

def sample_batch_replay_buffer(batch_size, replay_buffer):
    """
    Given the batch size, sample that many experiences from the replay buffer
    and return these experiences
    """
    
    indices = np.random.randint(len(replay_buffer), size=batch_size)
    batch = [replay_buffer[index] for index in indices]
    s, a, r, s_prime, done = [np.array([ex[i] for ex in batch]) for i in range(5)]
    return s, a, r, s_prime, done

def dqn_step(env, s, model, replay_buffer, epsilon=0.01):
    """
    Implement one step through the environment given the current state, action and model
    """

    a = epsilon_greedy(s, model, epsilon=epsilon)
    s_prime, r, done, info = env.step(a)
    replay_buffer.append((s, a, r, s_prime, done))
    return s_prime, r, done, info

def dqn(batch_size, replay_buffer, model, target, loss_fn, optimizer, n_outputs=2, gamma=0.95):
    """
    Method to sample from the replay buffer, calculate the target q values, and compute the 
    gradients in the direction of this target from the current experienced states
    """

    all_s, all_a, all_r, all_s_prime, all_done = sample_batch_replay_buffer(batch_size, replay_buffer)
    best_next_a = np.argmax(model.predict(all_s_prime), axis=1)
    next_mask = tf.one_hot(best_next_a, n_outputs, on_value=1.0, off_value=0.0).numpy()
    next_best_q = (target.predict(all_s_prime) * next_mask).sum(axis=1)
    target_q = (all_r + 
                       (1 - all_done) * gamma * next_best_q)
    target_q = target_q.reshape(-1, 1)
    mask = tf.one_hot(all_a, n_outputs, on_value=1.0, off_value=0.0)
    with tf.GradientTape() as tape:
        all_q = model(all_s)
        q_values = tf.reduce_sum(all_q * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_q, q_values))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

"""
Methods to implement Policy Gradient for ClassicControl in OpenAI Gym
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np


def build_policy_nn(n_neurons=10, n_inputs=4, activation_dense='elu', activation_out='sigmoid'):
    """
    Basic Neural Network for taking the observation and outputting the probability 
    of the action to move left or not
    """

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(
        n_neurons, activation=activation_dense, input_shape=[n_inputs]))
    model.add(keras.layers.Dense(n_neurons, activation=activation_dense))
    model.add(keras.layers.Dense(1, activation=activation_out))
    return model


def discount_and_normalize_episodes(all_rewards, gamma):
    """
    Method to normalize the total discounted reward for all actions
    """

    total_discounted = [discounted_rewards_step(r, gamma) for r in all_rewards]
    flatten = np.concatenate(total_discounted)
    mean = flatten.mean()
    std = flatten.std()
    return [(discounted-mean)/std for discounted in total_discounted]


def discounted_rewards_step(rewards, gamma=0.9):
    """
    Method to calculate the total discounted reward of an action
    """

    discounted = np.array(rewards)
    for i in range(len(rewards)-2, -1, -1):
        discounted[i] += discounted[i+1] * gamma
    return discounted


def policy_gradient_update(env, obs, nn_model, loss_fn, render=False):
    """
    Method to implement one gradient update and take action in that direction
    """
    
    with tf.GradientTape() as tape:
        left_p = nn_model(obs[np.newaxis])
        a = (tf.random.uniform([1, 1], dtype=tf.float32) > left_p)
        y_target = tf.constant([[1.]]) - tf.cast(a, tf.float32)
        loss = tf.reduce_mean(loss_fn(y_target, left_p))
    grads = tape.gradient(loss, nn_model.trainable_variables)
    o, r, done, _ = env.step(int(a[0, 0].numpy()))
    if render:
        env.render()
    return o, r, done, grads


def policy_gradient(env, nn_model, loss_fn, n_ep=100, max_steps=200, render=False):
    """
    Method to carry out the policy gradient algorithm over a number of episodes
    """

    all_rewards = []
    all_gradients = []
    for _ in range(n_ep):
        current_rewards = []
        current_gradients = []
        o = env.reset()
        for _ in range(max_steps):
            o, r, done, grads = policy_gradient_update(
                env, o, nn_model, loss_fn, render)
            current_rewards.append(r)
            current_gradients.append(grads)
            if done:
                break
        all_rewards.append(current_rewards)
        all_gradients.append(current_gradients)
    return all_rewards, all_gradients

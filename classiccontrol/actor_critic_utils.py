"""
Methods to implement Actor-Critic for ClassicControl in OpenAI Gym
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from collections import defaultdict


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


def ac_update(env, obs, nn_model, loss_fn, render=False):
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
    return o, r, done, grads, int(a[0, 0].numpy())


def ac_gradient(env, nn_model, loss_fn, optimizer, gamma=0.9, alpha=0.1, n_iterations=1000, render=False):
    """
    Method to carry out the actor_critic algorithm over a number of episodes
    """

    q_w = defaultdict(lambda: defaultdict(float))
    theta = defaultdict(lambda: defaultdict(list))

    for _ in range(n_iterations):
        o = env.reset()
        done = False

        while not done:
            o_prime, r, done, grads, a = ac_update(
                env, o, nn_model, loss_fn, render)
            a_target = max(q_w[o_prime.tobytes()]) if len(q_w[o_prime.tobytes()])>0  else a
            delta = r + gamma*q_w[o_prime.tobytes()][a_target]-q_w[o.tobytes()][a]
            theta[o.tobytes()][a].append([tf.reduce_mean(alpha*q_w[o.tobytes()][a]*g) for g in grads])
            print(theta[o.tobytes()][a])
            optimizer.apply_gradients(
                zip(theta[o.tobytes()][a], nn_model.trainable_variables))
            r_theta = theta[o.tobytes()][a]
            r_q_w = q_w[o.tobytes()][a]
            q_w[o.tobytes()][a] += alpha*delta
            o = o_prime
            
    return r_theta, r_q_w


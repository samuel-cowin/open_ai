from collections import deque
from tensorflow import keras
import numpy as np
import dqn_utils
import matplotlib.pyplot as plt

def run_dqn_training(env, batch_size=32, gamma=0.95, min_epsilon=0.01, n_neurons=36, n_episodes=500, n_steps=200, lr=1e-3, diag=False):
    """
    Takes the hyperparameters and implements a DQN for the provided environment
    """

    # Create the Neural Network Model and the target clone
    nn_model = dqn_utils.build_policy_nn(n_neurons=n_neurons, n_inputs=env.observation_space.shape[0], n_outputs=env.action_space.n)
    target = keras.models.clone_model(nn_model)
    target.set_weights(nn_model.get_weights())

    # Define the gradient optimization and loss
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    loss_fn = keras.losses.Huber()

    # Initialize replay_buffer for sample efficiency
    replay_buffer = deque(maxlen=3000)

    # Implement the training loop
    rewards_per = []
    for e in range(n_episodes):
        o = env.reset()
        rew = 0
        for _ in range(n_steps):
            epsilon = min_epsilon + (1.0-min_epsilon)*(1.0-(e/n_episodes))
            o, r, done, _ = dqn_utils.dqn_step(env, o, nn_model, replay_buffer, epsilon)
            rew += r
            if done:
                break
        if e > 50:
            dqn_utils.dqn(batch_size, replay_buffer, nn_model, target, loss_fn, optimizer, n_outputs=env.action_space.n, gamma=gamma)
        if e % 50 == 0 and e > 0:
            print('Completed {} episodes.'.format(e))
        rewards_per.append(rew)

    if diag:
        print('Maximum reward achieved was {}.'.format(np.max(rewards_per)))
        plt.plot(range(n_episodes), rewards_per)
        plt.show()

    return rewards_per
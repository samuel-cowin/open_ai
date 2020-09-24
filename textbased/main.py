import time
import gym
import value_iteration_utils
import q_learning_utils

# Choose TextBased Gym environment and input training parameters
env = gym.make("Taxi-v3")
n_ep = 10000
n_sim = 5
gamma = 0.9
alpha = 0.1
min_epsilon = 0.1
delta=0.001

"""
Find optimal value function and policy through Value Iteration
"""
v_opt, _ = value_iteration_utils.value_iteration(env.P, gamma=gamma, delta=delta)

"""
Find optimal policy through Q-Learning
"""
q_opt, pi_opt = q_learning_utils.q_learning(env, n_ep=n_ep, min_epsilon=min_epsilon, alpha=alpha, gamma=gamma)

# Simulate optimal policy on environment and print statistics
input("Press enter to start simulation\n")
total_reward = 0
for e in range(n_sim):
    episode_reward = 0
    t = 0
    done = False
    s = env.reset()
    while not done:
        env.render()
        time.sleep(0.25)
        (s_prime, r, done, _) = env.step(pi_opt(s))
        episode_reward += r
        s = s_prime
        t += 1

    total_reward += episode_reward
    print("Completed episode {} in {} timesteps with {} reward".format(e, t, episode_reward))

print("Completed {} episodes with {} average reward and {} total reward".format(
    int(n_sim), total_reward / float(n_sim), total_reward))

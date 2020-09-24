"""
Q-Learning Algorithm to solve the TextBased problems from OpenAI gym
"""

from collections import defaultdict
import numpy as np


def epsilon_greedy(q_, p_dist, eps=0.01):
    """
    Calculate the best policy with epsilon randomness and return this policy
    """
    
    def policy(s):
        """
        Callable method depending on if the policy or the action are needed
        """

        rand_a = p_dist.action_space.sample()
        best = max(q_[s], key=q_[s].get) if len(q_[s]) > 0 else rand_a
        return np.random.choice([rand_a, best], p=[eps, 1-eps])
    return policy

def calc_update(target_policy_value, current_policy_value, reward, gamma):
    """
    From the found target and current values, calculate the difference
    """
    
    delta = reward + gamma * target_policy_value - current_policy_value
    return delta

def q_learning_update(p_dist, q_pi, epsilon, s, gamma, alpha):
    """
    From the current values and state, find the new optimal values
    """
    
    a = epsilon_greedy(q_pi, p_dist, epsilon)(s)
    s_prime, r, done, _ = p_dist.step(a)
    a_target = epsilon_greedy(q_pi, p_dist, 0)(s_prime)
    delta = calc_update(q_pi[s_prime][a_target], q_pi[s][a], r, gamma)
    q_pi[s][a] += alpha * delta
    s = s_prime
    return q_pi, s, done

def q_learning(p_dist, n_ep=1000, min_epsilon=0.01, alpha=0.01, gamma=0.9):
    """
    Go through a number of episodes of the environment until the goal is reached on each
    Find the state-action values that result in the optimal policy through the environment
    """

    q_pi = defaultdict(lambda: defaultdict(float))
    for e in range(n_ep):
        s = p_dist.reset()
        epsilon = (1.0-min_epsilon) * (1.0-float(e)/float(n_ep)) + min_epsilon
        done = False
        while not done:
            q_pi, s, done = q_learning_update(p_dist, q_pi, epsilon, s, gamma, alpha)
    return q_pi, epsilon_greedy(q_pi, p_dist, min_epsilon)
    
"""
Value Iteration Algorithm to solve the TextBased problems from OpenAI gym
"""
import numpy as np

def value_iteration(p_dist, gamma=0.9, delta=0.001):
    """
    Initialize value function of same size as environment space and iterate value function update
    until they are not changing significantly
    """
    v_opt = {s: 0.0 for s in p_dist}
    max_d = delta

    while max_d >= delta:
        """
        Call value_iteration_update to perform update for each element in value function
        Update delta value to reflect current changing behavior
        """
        v_curr = {s: value_iteration_update(p_dist, s, v_opt, gamma) for s in p_dist}
        max_d = max([abs(v_curr[s] - v_opt[s]) for s in p_dist])
        v_opt = v_curr
    
    pi_opt = greedy(p_dist, v_opt)
    return v_opt, pi_opt

def value_iteration_update(p_dist, s, v_curr, gamma=1.0):
    """
    Perform update for each action within the state
    """
    return max([calc_value(p_dist, s, a, v_curr, gamma) for a in p_dist[s]])

def calc_value(p_dist, s, a, v_curr, gamma=1.0):
    """
    Execute value iteration step modification 
    """
    return sum([prob * (r + gamma * v_curr[s_prime]) for (prob, s_prime, r, _) in p_dist[s][a]])

def greedy(p_dist, v_opt):
    """
    Calculate the best move and return this policy
    """
    return {s: np.argmax([calc_value(p_dist, s, a, v_opt) for a in p_dist[s]]) for s in p_dist}
import numpy as np
import copy
from sklearn.linear_model import LinearRegression
import qiskit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.analysis import hellinger_fidelity
import sklearn
from sklearn.metrics import mean_squared_error 
import scipy.stats

def chi2_distance(target, obs):
    target = copy.deepcopy(target)
    obs = copy.deepcopy(obs)
    obs = np.absolute(obs)
    if isinstance(target, np.ndarray):
        assert len(target) == len(obs)
        distance = 0
        for t, o in zip(target, obs):
            if abs(t - o) > 1e-10:
                distance += np.power(t - o, 2) / (t + o)
    elif isinstance(target, dict):
        distance = 0
        for o_idx, o in enumerate(obs):
            if o_idx in target:
                t = target[o_idx]
                if abs(t - o) > 1e-10:
                    distance += np.power(t - o, 2) / (t + o)
            else:
                distance += o
    else:
        raise Exception("Illegal target type:", type(target))
    return distance


def MSE(target, obs):
    """
    Mean Square Error
    """
    target = copy.deepcopy(target)
    obs = copy.deepcopy(obs)
    if isinstance(target, dict):
        se = 0
        for t_idx in target:
            t = target[t_idx]
            o = obs[t_idx]
            se += (t - o) ** 2
        mse = se / len(obs)
    elif isinstance(target, np.ndarray) and isinstance(obs, np.ndarray):
        target = target.reshape(-1, 1)
        obs = obs.reshape(-1, 1)
        squared_diff = (target - obs) ** 2
        se = np.sum(squared_diff)
        mse = np.mean(squared_diff)
    elif isinstance(target, np.ndarray) and isinstance(obs, dict):
        se = 0
        for o_idx in obs:
            o = obs[o_idx]
            t = target[o_idx]
            se += (t - o) ** 2
        mse = se / len(obs)
    else:
        raise Exception("target type : %s" % type(target))
    return mse


def MAPE(target, obs):
    """
    Mean absolute percentage error
    abs(target-obs)/target
    """
    target = copy.deepcopy(target)
    obs = copy.deepcopy(obs)
    epsilon = 1e-16
    if isinstance(target, dict):
        curr_sum = np.sum(list(target.values()))
        new_sum = curr_sum + epsilon * len(target)
        mape = 0
        for t_idx in target:
            t = (target[t_idx] + epsilon) / new_sum
            o = obs[t_idx]
            mape += abs((t - o) / t)
        mape /= len(obs)
    elif isinstance(target, np.ndarray) and isinstance(obs, np.ndarray):
        target = target.flatten()
        target += epsilon
        target /= np.sum(target)
        obs = obs.flatten()
        obs += epsilon
        obs /= np.sum(obs)
        mape = np.abs((target - obs) / target)
        mape = np.mean(mape)
    elif isinstance(target, np.ndarray) and isinstance(obs, dict):
        curr_sum = np.sum(list(target.values()))
        new_sum = curr_sum + epsilon * len(target)
        mape = 0
        for o_idx in obs:
            o = obs[o_idx]
            t = (target[o_idx] + epsilon) / new_sum
            mape += abs((t - o) / t)
        mape /= len(obs)
    else:
        raise Exception("target type : %s" % type(target))
    return mape * 100


def fidelity(target, obs):
    target = copy.deepcopy(target)
    obs = copy.deepcopy(obs)
    if isinstance(target, np.ndarray):
        assert len(target) == len(obs)
        fidelity = 0
        for t, o in zip(target, obs):
            if t > 1e-16:
                fidelity += o
    elif isinstance(target, dict):
        fidelity = 0
        for t_idx in target:
            t = target[t_idx]
            o = obs[t_idx]
            if t > 1e-16:
                fidelity += o
    else:
        raise Exception("target type : %s" % type(target))
    return fidelity


def cross_entropy(target, obs):
    target = copy.deepcopy(target)
    obs = copy.deepcopy(obs)
    if isinstance(target, dict):
        CE = 0
        for t_idx in target:
            t = target[t_idx]
            o = obs[t_idx]
            o = o if o > 1e-16 else 1e-16
            CE += -t * np.log(o)
        return CE
    elif isinstance(target, np.ndarray) and isinstance(obs, np.ndarray):
        obs = np.clip(obs, a_min=1e-16, a_max=None)
        CE = np.sum(-target * np.log(obs))
        return CE
    elif isinstance(target, np.ndarray) and isinstance(obs, dict):
        CE = 0
        for o_idx in obs:
            o = obs[o_idx]
            t = target[o_idx]
            o = o if o > 1e-16 else 1e-16
            CE += -t * np.log(o)
        return CE
    else:
        raise Exception("target type : %s, obs type : %s" % (type(target), type(obs)))


def relative_entropy(target, obs):
    target = copy.deepcopy(target)
    obs = copy.deepcopy(obs)
    return cross_entropy(target=target, obs=obs) - cross_entropy(
        target=target, obs=target
    )


def correlation(target, obs):
    """
    Measure the linear correlation between `target` and `obs`
    """
    target = copy.deepcopy(target)
    obs = copy.deepcopy(obs)
    target = target.reshape(-1, 1)
    obs = obs.reshape(-1, 1)
    reg = LinearRegression()
    reg.fit(X=obs, y=target)
    score = reg.score(X=obs, y=target)
    return score


def HOP(target, obs):
    """
    Measures the heavy output probability
    """
    target = copy.deepcopy(target)
    obs = copy.deepcopy(obs)
    target_median = np.median(target)
    hop = 0
    for t, o in zip(target, obs):
        if t > target_median:
            hop += o
    return hop


def absres(target, obs):
    """
    Measure the linear correlation between `target` and `obs`
    """
    #target = copy.deepcopy(target)
    #obs = copy.deepcopy(obs)
    #target = target.reshape(-1, 1)
    #obs = obs.reshape(-1, 1)
    return len(target), len(obs)


# def hellinger_fidelity(target,obs):
#         total = 0
#         for i in range(len(target)):
#             total += (np.sqrt(target[i]) - np.sqrt(obs[i]))**2
#             total += obs[i]
#         dist = np.sqrt(total)/np.sqrt(2)
#         return dist



# def hellinger_fidelity(target,obs):
#     target = copy.deepcopy(target)
#     obs = copy.deepcopy(obs)
#     tar_dict = {i: target[i] for i in range(len(target))}
#     obs_dict = {i: obs[i] for i in range(len(obs))}
#     return qiskit.quantum_info.analysis.hellinger_fidelity(tar_dict, obs_dict)




def hellinger_fidelity(target, obs):
    total = 0
    for i in range(len(target)):
        total += (np.sqrt(target[i]) - np.sqrt(obs[i]))**2
    dist = np.sqrt(total) / np.sqrt(2)
    return (1 - dist**2) ** 2 # Normalize to be in the range [0, 1]




def kl_divergence(p, q):
    """
    Compute the Kullback-Leibler divergence from distribution p to distribution q.
    """
    #return np.sum(np.where(p != 0, p * np.log(p / q), 0))
    return np.sum(np.where(p != 0, p * np.log(p / (q + np.finfo(float).eps)), 0))
    #return sum(p[i] * np.log(p[i]/q[i]) for i in range(len(p)))

def klfidelity(target, obs):
    """
    Compute the KL-divergence-based fidelity between two distributions.
    """
    target = copy.deepcopy(target)
    obs = copy.deepcopy(obs)
    assert len(target) == len(obs), "Arrays must have the same length"
    return kl_divergence(target, obs)


def jensen_shannon_fidelity(target, obs):
    """
    Compute the Jensen-Shannon divergence-based fidelity between two distributions.
    """
    target = copy.deepcopy(target)
    obs = copy.deepcopy(obs)
    assert len(target) == len(obs), "Arrays must have the same length"
    return jensen_shannon_divergence(target, obs)

def jensen_shannon_divergence(p, q):
    """
    Compute the Jensen-Shannon divergence between two distributions p and q.
    """
    #target = copy.deepcopy(target)
    #obs = copy.deepcopy(obs)
    m = 0.5 * (p + q)
    divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2
    return (1-np.sqrt(divergence))
    #return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def total_variation_distance(p, q):
    """
    Compute the Total Variation Distance (TVD) between two probability distributions p and q.
    """
    return 0.5 * np.sum(np.abs(p - q))

def tvd_fidelity(target, obs):
    """
    Compute the Total Variation Distance (TVD)-based fidelity between two distributions.
    """
    target = copy.deepcopy(target)
    obs = copy.deepcopy(obs)
    assert len(target) == len(obs), "Arrays must have the same length"
    return total_variation_distance(target, obs)


def func(target, obs):
    tot_len = len(target)
    ans = 0
    val = 0.01/tot_len
    for i in range(tot_len):
        if(target[i]>val):   
            ans += abs(target[i]-obs[i])
    return ans*0.5

def quartile1_hop(target, obs):
    k =1/4
    target = np.array(target)
    observed = np.array(obs)
    
    num_states = len(target)
    num_top_states = int(k * num_states)
    
    # Getting top k indeices
    top_indices = np.argsort(target)[::-1][:num_top_states]
    
    absolute_differences = np.abs(target[top_indices] - observed[top_indices])

    average_diff = np.sum(absolute_differences) * 0.5
    
    return average_diff


def quartile2_hop(target, obs):
    k =1/2
    target = np.array(target)
    observed = np.array(obs)
    
    num_states = len(target)
    num_top_states = int(k * num_states)
    
    # Getting top k indeices
    top_indices = np.argsort(target)[::-1][:num_top_states]
    
    absolute_differences = np.abs(target[top_indices] - observed[top_indices])

    average_diff = np.sum(absolute_differences) * 0.5
    
    return average_diff


def quartile3_hop(target, obs):
    k =3/4
    target = np.array(target)
    observed = np.array(obs)
    
    num_states = len(target)
    num_top_states = int(k * num_states)
    
    # Getting top k indeices
    top_indices = np.argsort(target)[::-1][:num_top_states]
    
    absolute_differences = np.abs(target[top_indices] - observed[top_indices])

    average_diff = np.sum(absolute_differences) * 0.5
    
    return average_diff



def kfraction_distance(target, observed, k):
    
    target = np.array(target)
    observed = np.array(observed)
    
    num_states = len(target)
    num_top_states = int(k * num_states)
    
    # Getting top k indeices
    top_indices = np.argsort(target)[::-1][:num_top_states]
    
    absolute_differences = np.abs(target[top_indices] - observed[top_indices])

    average_diff = np.sum(absolute_differences) * 0.5
    
    return average_diff
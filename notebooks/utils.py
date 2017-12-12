import matplotlib.pyplot as plt
import numpy as np

class SampleLogger():
    
    def __init__(self, metrics):
        self.stats = {}
        for metric in metrics:
            self.stats[metric] = []

    def log(self, sample_stats):
        for metric, stats in sample_stats.items():
            if metric not in self.stats:
                continue
            self.stats[metric].append(stats)
    
    def __getitem__(self, key):
        return self.stats.get(key, None)

def timeseries(estimates):
    n = len(estimates[0])
    m = len(estimates)
    series = np.ndarray(shape=(n, m), dtype=np.float32)
    for i, estimate in enumerate(estimates):
        for j, value in enumerate(estimate):
            series[j][i] = value
    return series

def rms_error(v1, v2):
    return np.sqrt(np.mean(np.square(v1 - v2)))

def rms_series(estimates, reference):
    series = []
    for values in estimates:
        series.append(rms_error(values, reference))
    return series

def display_final_estimate(V_final, V_optimal):
    print('>> RMS error =', rms_error(V_final, V_optimal))
    print(">> V_final:   ", end='')
    for v in V_final:
        print('{:.5f}'.format(v), end=' ')
    print()
    print(">> V_optimal: ", end='')
    for v in V_optimal:
        print('{:.5f}'.format(v), end=' ')
    print()

def plot_action_distribution(actions, size):
    plt.hist(actions, bins=size, facecolor='blue', edgecolor='grey')
    plt.title('Distribution of actions for a random policy')
    plt.show()

def plot_value_estimates(V_t, title):
    fig = plt.figure(figsize=(15, 3))
    for s, v in enumerate(V_t):
        plt.plot(v, label='s = {:2}'.format(s))
    plt.title(title)
    plt.xlabel('Episodes')
    plt.ylabel('$V^\\pi(s)$')
    plt.grid()
    plt.legend()
    plt.show()

def plot_rms_errors(errors):
    fig = plt.figure(figsize=(7, 5))
    for error_series in errors:
        for label, values in error_series.items():
            plt.plot(values, label=label)
    plt.title('RMS error, averaged over states')
    plt.xlabel('Walks / Episodes')
    plt.grid()
    plt.legend()
    plt.show()

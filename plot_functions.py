import numpy as np
import matplotlib.pyplot as plt

def plot_iter_vs_success_rate(filepath, robot_name):
    """data fields are:
    
    0 : maximum number of iterations
    1 : # successful projection runs
    2 : total # runs that was executed
    3 : # successful runs after enforcing the joint limits
    
    """
    d = np.loadtxt(filepath, delimiter=",")
    _, ncols = d.shape
    fig, ax = plt.subplots()
    legend_labels = ["Only projection", "After enforcing joint limits"]
    
    # plot success rate
    ax.plot(d[:, 0], d[:, 1] / d[:, 2])
    
    # plot success rate after enforce joint bounds (ejb)
    ax.plot(d[:, 0], d[:, 3] / d[:, 2])
    
    if ncols  == 5:
        ax.plot(d[:, 0], d[:, 4] / d[:, 2])
        legend_labels.append("Angle fix")
    
    ax.set_title(f"{robot_name}", fontsize=16)
    ax.set_xlabel("Maximum number of iterations for projection", fontsize=14)
    ax.set_ylabel("Succes rate", fontsize=14)
    
    if d.shape[1] > 3:
        ax.legend(legend_labels)
    return fig, ax

def read_and_transform_error_data(filepath):
    d = np.loadtxt(filepath, delimiter=",");
    
    # there is data for different runs in the array
    # find out where each run started to split it up
    idx = list(np.where(np.diff(d[:, 0]) != 1)[0])
    
    # append the last index of the data to make processing easier
    idx.append(len(d[:, 0]) - 1)
    
    num_runs = len(idx)
    max_iters = int(np.max(d[:, 0]))
    dd = np.zeros((max_iters, num_runs))
    
    print(f"Found {num_runs} with maximum {max_iters} iterations")
    
    # transform the data from d (Nx2) -> dd (iters x runs)
    for i in range(len(idx) - 1):
        dd[0:idx[i+1]-idx[i], i] = d[idx[i]:idx[i+1], 1]
    
    return dd

def plot_error_data(data, robot_name):
    fig, ax = plt.subplots()
    ax.plot(data, ".--")
    ax.set_title(f"{robot_name}", fontsize=14)
    ax.set_xlabel("Number of iterations for projection")
    ax.set_ylabel("Squared norm of constraint deviation")
    ax.set_xlim([0, data.shape[0]])
    return fig, ax
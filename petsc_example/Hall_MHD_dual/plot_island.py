
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_current_peak(dirname):
    
    filename = dirname + "/current_peak_integer.dat"
    data = np.loadtxt(filename)
    t_int = data[:,0]
    peak_int = data[:,1]
    
    filename = dirname + "/current_peak_half.dat"
    data = np.loadtxt(filename)
    t_half = data[:,0]
    peak_half = data[:,1]
        
    plt.figure()
    plt.plot(t_int, peak_int, label='Integer steps')
    plt.plot(t_half, peak_half, label='Half steps')
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Current peak")
    plt.savefig(dirname + "/current_peak.png", dpi=1200, bbox_inches='tight')

if __name__ == '__main__':
    
    dirname = "output/island_origin"
    
    plot_current_peak(dirname)
    
        
    
    
    
    
    
    
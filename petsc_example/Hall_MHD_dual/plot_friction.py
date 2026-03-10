
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def read_energy(filename):
    with open(filename, 'r') as f:
        lines = [line.strip().split() for line in f.readlines()]
    
    # Ensure each row has 5 columns (pad with 0s if needed)
    processed = []
    for row in lines:
        if len(row) < 5:
            row += ['0'] * (5 - len(row))  # Pad missing columns with 0
        processed.append(row[:5])  # Keep only first 5 columns if extras exist
    
    # Convert to numpy array of floats
    data = np.array(processed, dtype=float)
    
    t = data[:,0]
    E_total = data[:,1]
    E_kin = data[:,2]
    E_mag = data[:,3]
    E_var = data[:,4]
    
    return t, E_total, E_kin, E_mag, E_var

def plot_energy(dirname):
    
    t_int, E_total_int, E_kin_int, E_mag_int, E_var_int = read_energy(dirname+"/energy_integer.dat")
    t_int = t_int[1:]
    E_total_int = E_total_int[1:]
    
    mask_int = t_int < 100
    t_int = t_int[mask_int]
    E_total_int = E_total_int[mask_int]
    
    # t_half, E_total_half, E_kin_half, E_mag_half, E_var_half = read_energy(dirname+"/energy_half.dat")
    
    data = np.loadtxt(dirname+"/energy.dat")
    t = data[:,0]
    E_total = data[:,1]
    mask = t<100
    t = t[mask]
    E_total = E_total[mask]
        
    plt.figure()
    plt.plot(t_int, E_total_int, label='Helicity conservative scheme')
    plt.plot(t, E_total, label='Helicity nonconservative scheme')
    plt.legend()
    plt.title("Total Energy")
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.savefig(dirname + "/total_energy.png", dpi=1200, bbox_inches='tight')

if __name__ == '__main__':
    
    dirname = "output/friction/N10Order1"
    
    plot_energy(dirname)
    
        
    
    
    
    
    
    
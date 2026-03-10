
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
    
    E_var[1] = 0.0
    
    E_var = np.cumsum(E_var)
    
    E_modify = E_total - E_var
    
    return t, E_total, E_kin, E_mag, E_var, E_modify

def read_helicity(filename):
    
    data = np.loadtxt(filename)
    t = data[:,0]
    H = data[:,1]
    return t, H

def plot_energy(dirname):
    
    
    plt.figure()
    
    plt.legend()
    plt.savefig(dirname+"/energy.png", dpi=600)
    
def read_ref_energy_helicity(filename):
    
    data = np.loadtxt(filename)
    t = data[:,0]
    E = data[:,1]
    H = data[:,2]
    return t, E, H

def plot(dirname):
    
    colors = sns.color_palette()
    
    t_half, E_total_half, E_kin_half, E_mag_half, E_var_half, E_modify_half = read_energy(dirname+"/energy_half.dat")
    t_int, E_total_int, E_kin_int, E_mag_int, E_var_int, E_modify_int = read_energy(dirname+"/energy_integer.dat")
    t1, H1 = read_helicity(dirname+"/m_helicity1.dat")
    t2, H2 = read_helicity(dirname+"/m_helicity2.dat")
    
    t_ref, E_ref, H_ref = read_ref_energy_helicity(dirname+"/MHD_BDF_A.out")
    
    cuttime = 500
    mask_ref = t_ref < cuttime
    mask_1 = t1 < cuttime
    mask_int = t_int < cuttime
    mask_int[0] = False
    mask_half = t_half < cuttime
    
    plt.figure()
    # plt.plot(t_half[1:], E_total_half[1:], label=r'$\mathcal{E}_{half}$')
    plt.plot(t_int[mask_int], E_total_int[mask_int], color=colors[0], label='helicity conservative scheme')
    plt.plot(t_ref[mask_ref], E_ref[mask_ref], color=colors[1], label='helicity nonconservative scheme')
    # plt.plot(t2, H2, label=r'$\mathcal{H}_{M,2}$')
    plt.legend(loc='right')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title('Energy')
    plt.savefig(dirname+"/energy.png", dpi=600)
    
    plt.figure()
    # plt.plot(t_half[1:], E_total_half[1:], label=r'$\mathcal{E}_{half}$')
    plt.plot(t1[mask_1], H1[mask_1], color=colors[0], label='helicity conservative scheme')
    plt.plot(t_ref[mask_ref], H_ref[mask_ref], color=colors[1], label='helicity nonconservative scheme')
    # plt.plot(t2, H2, label=r'$\mathcal{H}_{M,2}$')
    # plt.ylim(0.0, 0.65)
    plt.legend(loc='right')
    plt.xlabel('Time')
    plt.ylabel('Helicity')
    plt.title('Helicity')
    plt.savefig(dirname+"/helicity.png", dpi=600)
    

if __name__ == '__main__':
    dirname = "/share/home/xiruijie/Hall_MHD_dual/output/hopf/order1/N16"
    
    plot(dirname)
        
    
    
    
    
    
    
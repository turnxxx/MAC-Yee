
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
    
    t_half, E_total_half, E_kin_half, E_mag_half, E_var_half, E_modify_half = read_energy(dirname+"/energy_half.dat")
    
    t_int, E_total_int, E_kin_int, E_mag_int, E_var_int, E_modify_int = read_energy(dirname+"/energy_integer.dat")
    
    colors = sns.color_palette()
    
    plt.figure()
    plt.plot(t_int[1:], E_kin_int[1:], color=colors[0], label=r'$\mathcal{E}_{k,\text{int}}$')
    plt.plot(t_half[1:], E_kin_half[1:], '--',color=colors[0], label=r'$\mathcal{E}_{k,\text{half}}$')
    

    plt.plot(t_int[1:], E_mag_int[1:], color=colors[1], label=r'$\mathcal{E}_{m,\text{int}}$')
    plt.plot(t_half[1:], E_mag_half[1:], '--',color=colors[1], label=r'$\mathcal{E}_{m,\text{half}}$')
    
    plt.plot(t_int[1:], E_total_int[1:], color = colors[2], label=r'$\mathcal{E}_{\text{int}}$')
    plt.plot(t_half[1:], E_total_half[1:], '--', color = colors[2], label=r'$\mathcal{E}_{\text{half}}$')
    # plt.plot(t_int[1:], E_modify_int[1:], label=r'$\tilde{\mathcal{E}}_{\text{int}}$')
    # plt.plot(t_half[1:], E_modify_half[1:], label=r'$\tilde{\mathcal{E}}_{\text{half}}$')
    plt.legend(loc='right')
    plt.savefig(dirname+"/energy.png", dpi=600)

def plot_helicity(dirname):
    
    t1, H1 = read_helicity(dirname+"/m_helicity1.dat")
    
    t2, H2 = read_helicity(dirname+"/m_helicity2.dat")
    
    plt.figure()
    plt.plot(t1, H1, label=r'$\mathcal{H}_{M,1}$')
    plt.plot(t2, H2, label=r'$\mathcal{H}_{M,2}$')
    plt.legend()
    plt.ylim(-1e-10, 1e-10)
    plt.savefig(dirname+"/magnetic_helicity.png", dpi=600)
    
    
    

if __name__ == '__main__':
    dirname = "/share/home/xiruijie/Hall_MHD_dual/output/orszag_tang_kraus/order1/N256"
    
    plot_energy(dirname)
    
    plot_helicity(dirname)
        
    
    
    
    
    
    
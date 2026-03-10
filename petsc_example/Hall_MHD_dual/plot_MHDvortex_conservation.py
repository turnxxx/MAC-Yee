
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

CONSERVATION_FOLDER='output/MHDvortex_conservation'
DPI=600

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

def plot_energy(orderlist, filename, subtract_first=True, type='Total'):
    
    plt.figure()
    
    for order in orderlist:
        
        subfolder = CONSERVATION_FOLDER + "/order" + str(order) + "/N40"
        
        if type == 'magnetic_helicity':
            data1 = np.loadtxt(subfolder+"/m_helicity1.dat")
            data2 = np.loadtxt(subfolder+"/m_helicity2.dat")    
            t1 = data1[:,0]
            helicity1 = data1[:,1]
            t2 = data2[:,0]
            helicity2 = data2[:,1]
        elif type == 'divergence':
            
            data_int = np.loadtxt(subfolder+"/divergence_integer.dat")
            data_half = np.loadtxt(subfolder+"/divergence_half.dat")
            
            t_int = data_int[:,0]
            divu_int = data_int[:,1]
            divB_int = data_int[:,2]
            
            t_half = data_half[:,0]
            divJ_half = data_half[:,1]
            
        else:
            t_int, E_total_int, E_kin_int, E_mag_int, E_var_int = read_energy(subfolder+"/energy_integer.dat")
            t_half, E_total_half, E_kin_half, E_mag_half, E_var_half = read_energy(subfolder+"/energy_half.dat")
        
        if type == 'Total':
            label_1 = r'$E_{2,h}$'
            label_2 = r'$E_{1,h}$'
            if subtract_first:
                E_total_int -= E_total_int[1]
                E_total_half -= E_total_half[1]
                label_1 = r'$E_{2,h} - E_{2,h}^1$'
                label_2 = r'$E_{1,h} - E_{1,h}^1$'

            plt.plot(t_half[1:], E_total_half[1:], label=label_2)
            plt.plot(t_int[1:], E_total_int[1:], label=label_1)            
            plt.legend()
            plt.xlabel("Time")
            plt.ylabel("Total Energy")
            plt.ylim(-5e-10, 5e-10)
            
        elif type == 'kinetic':
        
            plt.plot(t_half[1:], E_kin_half[1:], label='half-integer steps')
            plt.plot(t_int[1:], E_kin_int[1:], label='integer steps')
            plt.xlabel("Time")
            plt.ylabel("Kinetic Energy")
            plt.legend()
            
        elif type == 'magnetic':
        
            plt.plot(t_half[1:], E_mag_half[1:], label='half-integer steps')
            plt.plot(t_int[1:], E_mag_int[1:], label='integer steps')
            plt.xlabel("Time")
            plt.ylabel("Magnetic Energy")
            plt.legend()
            
        elif type == 'magnetic_helicity':
            
            if subtract_first:
                helicity1 -= helicity1[0]
                helicity2 -= helicity2[0]
                label_1 = r'$\mathcal{H}_{M,1}-\mathcal{H}_{M,1}^1$'
                label_2 = r'$\mathcal{H}_{M,2}-\mathcal{H}_{M,2}^1$'
            else:
                label_1 = r'$\mathcal{H}_{M,1}$'
                label_2 = r'$\mathcal{H}_{M,2}$'
            
            plt.plot(t1, helicity1, label=label_1)
            plt.plot(t2, helicity2, label=label_2)
            
            plt.legend()
            plt.xlabel("Time")
            plt.ylabel("Magnetic Helicity")        
            plt.ylim(-1e-13, 1e-13)
            
        elif type == 'divergence':
            plt.plot(t_int, divu_int, label=r'$\|\nabla \cdot u\|_{L^2}$')
            plt.plot(t_int, divB_int, label=r'$\|\nabla \cdot B\|_{L^2}$')
            plt.plot(t_half, divJ_half, label=r'$\|\nabla \cdot J\|_{L^2}$')
            plt.legend()
            plt.xlabel("Time")
            plt.ylabel("Error")
            
    save_filename = CONSERVATION_FOLDER + "/" + filename 
    plt.savefig(save_filename, dpi=DPI, bbox_inches='tight')
    
def plot_divergence(dirname):
    
    data_int = np.loadtxt(dirname+"/divergence_integer.dat")
    data_half = np.loadtxt(dirname+"/divergence_half.dat")
    
    t_int = data_int[:,0]
    divu_int = data_int[:,1]
    divB_int = data_int[:,2]
    
    t_half = data_half[:,0]
    divJ_half = data_half[:,1]
    
    plt.figure()
    plt.plot(t_int, divu_int, label=r'$\|\nabla \cdot u\|_{L^2}$')
    plt.plot(t_int, divB_int, label=r'$\|\nabla \cdot B\|_{L^2}$')
    plt.plot(t_half, divJ_half, label=r'$\|\nabla \cdot J\|_{L^2}$')
    
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Error")
    outputfile = dirname + "/divergence.png"
    plt.savefig(outputfile, dpi=DPI, bbox_inches='tight')
    


if __name__ == '__main__':
        
    plot_energy(orderlist=[1], filename="total_energy", subtract_first=True, type='Total')
    plot_energy(orderlist=[1], filename="magnetic_helicity", subtract_first=True, type='magnetic_helicity')
    
    plot_energy(orderlist=[1], filename="kinetic_energy", subtract_first=False, type='kinetic')
    plot_energy(orderlist=[1], filename="magnetic_energy", subtract_first=False, type='magnetic')
    
    
    plot_energy(orderlist=[1], filename="divergence_error", subtract_first=False, type='divergence')    
    
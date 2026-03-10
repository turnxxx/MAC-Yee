
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

MHDVORTEX_PATH= 'output/MHDvortex'
DPI=600

def GetErrorsFromFile(filename, pattern, with_comma=False):
    pattern1 = ", "+pattern if with_comma else pattern
    with open(filename, 'r') as file:
        for line in file:
            if pattern1 in line:
                    # Extract the number after the pattern
                    error = float(line.split(':')[1].strip())
                    return error
    return None
    
def show_triangle(h, error, order):
    width = 0.2
    i = (len(h) + 1) // 2 - 1
    xshift = 0.1 * h[i]
    yshift = -0.1 * error[i]
    x1 = h[i]
    x2 = h[i]+width*(h[i-1]-h[i])
    y1 = error[i]
    y2 = y1*(x2/x1)**order
    x1 += xshift
    x2 += xshift
    y1 += yshift
    y2 += yshift
    triangleX = [x1, x2, x2]  # x的坐标
    triangleY = [y1, y1, y2]  # y的坐标
    plt.fill(np.array(triangleX), np.array(triangleY), 'k', alpha=0.5)
    plt.text(x2+xshift*2.25, 0.5*(y1+y2), "slope " + str(order), fontsize=8, horizontalalignment='center', verticalalignment='center')
       
def plot_error_alltogether(N_list, order, pattern_list, titlename_list, outputfilename):
    
    colors = sns.color_palette()
    markers = ['o', 's', 'D', '^', 'p', '*', 'h', 'H', 'x', 'd']
    
    plt.clf()
    min_error_i = 0
    min_error = 1e10
    error_list_list = []
    for i in range(len(pattern_list)):
        pattern = pattern_list[i]
        error_list = []
        h_list = []
        for N in N_list:
            h_list.append(1/N)
            filename = MHDVORTEX_PATH + '/order'+str(order)+'/N'+str(N)+'/Hall_MHD.out'
            error = GetErrorsFromFile(filename, pattern, with_comma=True)
            if(error == None):
                print("Error: ", pattern, " not found in ", filename)
                continue
            if error < min_error:
                min_error = error
                min_error_i = i
            error_list.append(error)
        error_list_list.append(error_list)
        
        # log scale
        plt.plot(h_list, error_list, linestyle='--', marker=markers[i%len(markers)], color=colors[i], label=titlename_list[i])
        
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$h$')
    plt.ylabel('Error')
    plt.legend()
    
    plt.xticks(h_list, h_list)
    
    # show triangle
    show_triangle(h_list, error_list_list[min_error_i], order=order)
    
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.minorticks_off()
    
    plt.savefig(outputfilename, bbox_inches='tight', dpi=DPI) 
    
def plot_spatial_all():
    
    # if ERROR_MODE == 'Herr_all':
    #     pattern_list_int = ['Hdiv error of u2', \
    #                     'Hcurl error of w1', \
    #                     'L2 error of p3', \
    #                     'Hcurl error of A1', \
    #                     'Hdiv error of B2', \
    #                     'Hcurl error of j1']

    #     titlename_list_int = [r"$\mathbf{H}(\operatorname{div})$ error of $\mathbf{u}_2$", \
    #                         r"$\mathbf{H}(\operatorname{curl})$ error of $\mathbf{\omega}_1$", \
    #                             r"$L^2$ error of $p_3$",\
    #                             r"$\mathbf{H}(\operatorname{curl})$ error of $\mathbf{A}_1$",\
    #                             r"$\mathbf{H}(\operatorname{div})$ error of $\mathbf{B}_2$",\
    #                             r"$\mathbf{H}(\operatorname{curl})$ error of $\mathbf{J}_1$"]
        
    #     pattern_list_half = ['Hcurl error of u1', \
    #                     'Hdiv error of w2', \
    #                     'H1 error of p0', \
    #                     'Hdiv error of A2', \
    #                     'Hcurl error of B1', \
    #                     'HDiv error of j2']

    #     titlename_list_half = [r"$\mathbf{H}(\operatorname{curl})$ error of $\mathbf{u}_1$", \
    #                     r"$\mathbf{H}(\operatorname{div})$ error of $\mathbf{\omega}_2$", \
    #                     r"$H^1$ error of $p_0$", \
    #                     r"$\mathbf{H}(\operatorname{div})$ error of $\mathbf{A}_2$", \
    #                     r"$\mathbf{H}(\operatorname{curl})$ error of $\mathbf{B}_1$", \
    #                     r"$\mathbf{H}(\operatorname{div})$ error of $\mathbf{J}_2$"]
    # elif ERROR_MODE == 'Herr_hydro':
    #     pattern_list_int = ['Hdiv error of u2', \
    #                     'Hcurl error of w1', \
    #                     'L2 error of p3']

    #     titlename_list_int = [r"$\mathbf{H}(\operatorname{div})$ error of $\mathbf{u}_2$", \
    #                         r"$\mathbf{H}(\operatorname{curl})$ error of $\mathbf{\omega}_1$", \
    #                             r"$L^2$ error of $p_3$"]
        
    #     pattern_list_half = ['Hcurl error of u1', \
    #                 'Hdiv error of w2', \
    #                 'H1 error of p0']
    
    #     titlename_list_half = [r"$\mathbf{H}(\operatorname{curl})$ error of $\mathbf{u}_1$", \
    #                         r"$\mathbf{H}(\operatorname{div})$ error of $\mathbf{\omega}_2$", \
    #                         r"$H^1$ error of $p_0$"]
        
    # elif ERROR_MODE == 'L2err_all':
    #     pattern_list_int = ['L2 error of u2', \
    #                     'L2 error of w1', \
    #                     'L2 error of p3', \
    #                     'L2 error of A1', \
    #                     'L2 error of B2', \
    #                     'L2 error of j1']

    #     titlename_list_int = [r"$L^2$ error of $\mathbf{u}_2$", \
    #                         r"$L^2$ error of $\mathbf{\omega}_1$", \
    #                             r"$L^2$ error of $p_3$",\
    #                             r"$L^2$ error of $\mathbf{A}_1$",\
    #                             r"$L^2$ error of $\mathbf{B}_2$",\
    #                             r"$L^2$ error of $\mathbf{J}_1$"]
                                
    #     pattern_list_half = ['L2 error of u1', \
    #                 'L2 error of w2', \
    #                 'L2 error of p0', \
    #                 'L2 error of A2', \
    #                 'L2 error of B1', \
    #                 'L2 error of j2']
    
    #     titlename_list_half = [r"$L^2$ error of $\mathbf{u}_1$", \
    #                         r"$L^2$ error of $\mathbf{\omega}_2$", \
    #                         r"$L^2$ error of $p_0$", \
    #                         r"$L^2$ error of $\mathbf{A}_2$", \
    #                         r"$L^2$ error of $\mathbf{B}_1$", \
    #                         r"$L^2$ error of $\mathbf{J}_2$"]
    
    pattern_list_int = ['L2 error of u2', \
                        'L2 error of p3', \
                        'L2 error of A1', \
                        'L2 error of B2']

    titlename_list_int = [r"$L^2$ error of $\mathbf{u}_2$", \
                            r"$L^2$ error of $p_3$",\
                            r"$L^2$ error of $\mathbf{A}_1$",\
                            r"$L^2$ error of $\mathbf{B}_2$"]
                            
    pattern_list_half = ['L2 error of u1', \
                'L2 error of p0', \
                'L2 error of A2', \
                'L2 error of B1']

    titlename_list_half = [r"$L^2$ error of $\mathbf{u}_1$", \
                        r"$L^2$ error of $p_0$", \
                        r"$L^2$ error of $\mathbf{A}_2$", \
                        r"$L^2$ error of $\mathbf{B}_1$"]
    
    pattern_list_primal_dual = ['L2 error of u', \
                                'L2 error of w', \
                                'L2 error of p', \
                                'L2 error of A', \
                                'L2 error of B', \
                                'L2 error of j']
    title_list_primal_dual = [r"$L^2$ error of $\mathbf{u}$", \
                              r"$L^2$ error of $\mathbf{\omega}$", \
                              r"$L^2$ error of $p$", \
                              r"$L^2$ error of $\mathbf{A}$", \
                              r"$L^2$ error of $\mathbf{B}$", \
                              r"$L^2$ error of $\mathbf{J}$"]
    
    N_list = [40,60,80,100,120,150,180]
    order = 2
    
    plot_error_alltogether(N_list=N_list, order=order, pattern_list=pattern_list_int, 
    titlename_list=titlename_list_int, outputfilename=MHDVORTEX_PATH + '/MHDvortex_error_int_order'+str(order)+'.png')
    plot_error_alltogether(N_list=N_list, order=order, pattern_list=pattern_list_half, 
    titlename_list=titlename_list_half, outputfilename=MHDVORTEX_PATH + '/MHDvortex_error_half_order'+str(order)+'.png')
    plot_error_alltogether(N_list=N_list, order=order, pattern_list=pattern_list_primal_dual, 
    titlename_list=title_list_primal_dual, outputfilename=MHDVORTEX_PATH + '/MHDvortex_error_primal_dual_order'+str(order)+'.png')

if __name__ == '__main__':
    
    plot_spatial_all()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

SPATIAL_ERROR_PATH = 'output/spatial_test'
SPATIAL_ERROR_MODE = 'Herr_all' # 'Herr_all' or 'L2err_all' or 'Herr_hydro' or 'Herr_mag'
DPI=600


def GetErrorsFromFile(filename, pattern, with_comma=False):
    pattern1 = ", "+pattern if with_comma else pattern
    with open(filename, 'r') as file:
        for line in file:
            if pattern1 in line:
                    # Extract the number after the pattern
                    error = float(line.split(':')[1].strip())
                    print(pattern," in ", filename, ":", error)
                    return error
    return None
    
    
def test_GetErrorsFromFile():
    pattern_list_int = ['L2 error of u2', \
                    'Div error of u2', \
                    'Hdiv error of u2', \
                    'L2 error of w1', \
                    'Hcurl error of w1', \
                    'L2 error of p3', \
                    'L2 error of A1', \
                    'Hcurl error of A1', \
                    'L2 error of B2', \
                    'Hdiv error of B2', \
                    'Div error of B2', \
                    'L2 error of j1', \
                    'Hcurl error of j1']
    
    pattern_list_half = ['L2 error of u1', \
                    'Hcurl error of u1', \
                    'L2 error of w2', \
                    'Hdiv error of w2', \
                    'L2 error of p0', \
                    'H1 error of p0', \
                    'L2 error of A2', \
                    'Hdiv error of A2', \
                    'L2 error of B1', \
                    'Hcurl error of B1', \
                    'L2 error of j2', \
                    'Div error of j2', \
                    'HDiv error of j2']
    

    filename='output/temporal/dt0.1N8Order2/Hall_MHD.out'
    
    for pattern in pattern_list_int:
        error = GetErrorsFromFile(filename, pattern, with_comma=True)
        print(pattern, ": " ,error)
        
    print("\n")
    for pattern in pattern_list_half:
        error = GetErrorsFromFile(filename, pattern, with_comma=True)
        print(pattern, ": " ,error)
        

def show_triangle(h, error, order, color=None, pos_idx=None):
    
    if pos_idx is not None:
        i = pos_idx
    else:
        i = (len(h) + 1) // 2 - 1
    
    width = 0.2
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
    
    if color is None:
        plt.fill(np.array(triangleX), np.array(triangleY), 'k', alpha=0.5)
    else:
        plt.fill(np.array(triangleX), np.array(triangleY), color=color, alpha=0.5)

    plt.text(x2+xshift*2, 0.5*(y1+y2), "slope " + str(order), fontsize=8, horizontalalignment='center', verticalalignment='center')
        
        
def plot_temporal_error(dt_list, order, N, pattern_list, titlename_list, outputfilename):
        
    # cmap = plt.get_cmap('Dark2')
    #     colors = sns.color_palette("tab10", len(pattern_list))
    #     colors = sns.color_palette("husl", len(pattern_list))

    # colors_origin = sns.color_palette("husl", 8)
    # color_i = [0,1,2,5,3,6]
    # colors = [colors_origin[i] for i in color_i]
    
    colors = sns.color_palette()
    markers = ['o', 's', 'D', '^', 'p', '*', 'h', 'H', 'x', 'd']
    
    plt.clf()
    min_error_i = 0
    min_error = 1e10
    error_list_list = []
    for i in range(len(pattern_list)):
        pattern = pattern_list[i]
        error_list = []
        for dt in dt_list:
            filename='output/temporal/dt'+str(dt)+'N'+str(N)+'Order'+str(order)+'/Hall_MHD.out'
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
        plt.plot(dt_list, error_list, linestyle='--', marker=markers[i%len(markers)], color=colors[i], label=titlename_list[i])
        
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\Delta t$')
    plt.ylabel('Error')
    plt.legend()
    
    plt.xticks(dt_list, dt_list)
    
    # show triangle
    show_triangle(dt_list, error_list_list[min_error_i], order=2)
    
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.minorticks_off()
    
    plt.savefig(outputfilename, bbox_inches='tight', dpi=DPI)    
    
def plot_spatial_error(N_lists, order_list, pattern, titlename, outputfilename, with_comma=False, showtriangle=True, ylim=None):
    
    markers = ['o', 's', 'D', '^', 'p', '*', 'h', 'H', 'x', 'd']
    plt.clf()    
    ax = plt.gca()
    for i_order in range(len(order_list)):
        order = order_list[i_order]
        h_list = []
        for N in N_lists[i_order]:
            h_list.append(1/N)
        error_list = []
        for N in N_lists[i_order]:
            filename= SPATIAL_ERROR_PATH + '/order'+str(order)+'/N'+str(N)+'/Hall_MHD.out'
            error = GetErrorsFromFile(filename, pattern, with_comma)
            if(error == None):
                print("Error: ", pattern, " not found in ", filename)
                continue
            error_list.append(error)
        p = ax.loglog(h_list, error_list, 's--', label='k = '+str(order))
        # plt.plot(h_list, error_list, linestyle='--', marker=markers[order%len(markers)], color=colors[order], label='k = '+str(order))
        if showtriangle:
            show_triangle(h_list, error_list, order, color=p[0].get_color(), pos_idx=-2)
        
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel(r'$h$')
    plt.ylabel('Error')
    plt.title(titlename)
    plt.legend()
    
    # plt.xticks(h_list, h_list)
    # plt.minorticks_off()
    
    # plt.grid(True, which="both", ls="--", linewidth=0.5)
    
    plt.savefig(outputfilename, bbox_inches='tight', dpi=DPI)

def plot_temporal(N, order):
    
    dt_list = [0.2, 0.1, 0.05, 0.025, 0.0125]
    
    pattern_list_int = ['Hdiv error of u2', \
                    'Hcurl error of w1', \
                    'L2 error of p3', \
                    'Hcurl error of A1', \
                    'Hdiv error of B2', \
                    'Hcurl error of j1']
    
    titlename_list_int = [r"$\mathbf{H}(\operatorname{div})$ error of $\mathbf{u}_{2,h}$", \
                      r"$\mathbf{H}(\operatorname{curl})$ error of $\mathbf{\omega}_{1,h}$", \
                        r"$L^2$ error of $P_{3,h}$",\
                        r"$\mathbf{H}(\operatorname{curl})$ error of $\mathbf{A}_{1,h}$",\
                        r"$\mathbf{H}(\operatorname{div})$ error of $\mathbf{B}_{2,h}$",\
                        r"$\mathbf{H}(\operatorname{curl})$ error of $\mathbf{J}_{1,h}$"]
    
    pattern_list_half = ['Hcurl error of u1', \
                    'Hdiv error of w2', \
                    'H1 error of p0', \
                    'Hdiv error of A2', \
                    'Hcurl error of B1', \
                    'HDiv error of j2']
    
    titlename_list_half = [r"$\mathbf{H}(\operatorname{curl})$ error of $\mathbf{u}_{1,h}$", \
                        r"$\mathbf{H}(\operatorname{div})$ error of $\mathbf{\omega}_{2,h}$", \
                        r"$H^1$ error of $P_{0,h}$", \
                        r"$\mathbf{H}(\operatorname{div})$ error of $\mathbf{A}_{2,h}$", \
                        r"$\mathbf{H}(\operatorname{curl})$ error of $\mathbf{B}_{1,h}$", \
                        r"$\mathbf{H}(\operatorname{div})$ error of $\mathbf{J}_{2,h}$"]
    
    outputfile_int = 'output/temporal/temporal_error_int_'+'order'+str(order)+"N"+str(N)+'.png'
    outputfile_half = 'output/temporal/temporal_error_half_'+'order'+str(order)+"N"+str(N)+'.png'
    
    plot_temporal_error(dt_list, order, N, pattern_list_int, titlename_list_int, outputfile_int)
    plot_temporal_error(dt_list, order, N, pattern_list_half, titlename_list_half, outputfile_half)
    
def plot_spatial():
    
    N_lists = [[4,8,16,32], \
                [4,8,16,32], \
                [4,8,16,32]]
    
    order_list = [1, 2, 3]
    
    pattern_list_int = ['Hdiv error of u2', \
                    'Hcurl error of w1', \
                    'L2 error of p3', \
                    'Hcurl error of A1', \
                    'Hdiv error of B2', \
                    'Hcurl error of j1',\
                    'Div error of u2', \
                    'Div error of B2']

    titlename_list_int = [r"$\mathbf{H}(\operatorname{div})$ error of $\mathbf{u}_{2,h}$", \
                        r"$\mathbf{H}(\operatorname{curl})$ error of $\mathbf{\omega}_{1,h}$", \
                            r"$L^2$ error of $P_{3,h}$",\
                            r"$\mathbf{H}(\operatorname{curl})$ error of $\mathbf{A}_{1,h}$",\
                            r"$\mathbf{H}(\operatorname{div})$ error of $\mathbf{B}_{2,h}$",\
                            r"$\mathbf{H}(\operatorname{curl})$ error of $\mathbf{J}_{1,h}$", \
                            r"$L^2$ error of $\operatorname{div}(\mathbf{u}_{2,h})$", \
                            r"$L^2$ error of $\operatorname{div}(\mathbf{B}_{2,h})$"]
    
    filename_list_int = ["u2Hdiverror", \
                      "w1Hcurlerror", \
                        "p3L2error",\
                        "A1Hcurlerror",\
                        "B2Hdiverror",\
                        "j1Hcurlerror",\
                        "u2Diverror", \
                        "B2Diverror"]
    
    triangle_list_int = [True, \
                     True, \
                    True, \
                    True, \
                    True, \
                    True, \
                    False, \
                    False]
                            
    pattern_list_half = ['Hcurl error of u1', \
                    'Hdiv error of w2', \
                    'H1 error of p0', \
                    'Hdiv error of A2', \
                    'Hcurl error of B1', \
                    'HDiv error of j2',\
                    'Div error of j2']
                    
    titlename_list_half = [r"$\mathbf{H}(\operatorname{curl})$ error of $\mathbf{u}_{1,h}$", \
                        r"$\mathbf{H}(\operatorname{div})$ error of $\mathbf{\omega}_{2,h}$", \
                        r"$H^1$ error of $P_{0,h}$", \
                        r"$\mathbf{H}(\operatorname{div})$ error of $\mathbf{A}_{2,h}$", \
                        r"$\mathbf{H}(\operatorname{curl})$ error of $\mathbf{B}_{1,h}$", \
                        r"$\mathbf{H}(\operatorname{div})$ error of $\mathbf{J}_{2,h}$", \
                        r"$L^2$ error of $\operatorname{div}(\mathbf{J}_{2,h})$"]
    
    filename_list_half = ["u1Hcurlerror", \
                        "w2Hdiverror", \
                        "p0H1error", \
                        "A2Hdiverror", \
                        "B1Hcurlerror", \
                        "j2Hdiverror", \
                        "j2Diverror"]
    
    triangle_list_half = [True, \
                     True, \
                    True, \
                    True, \
                    True, \
                    True, \
                    False]
    
    pattern_list_primal_dual = ['L2 error of u:', \
                                'L2 error of w:', \
                                'L2 error of p:', \
                                'L2 error of A:', \
                                'L2 error of B:', \
                                'L2 error of j:']
    title_list_primal_dual = [r"$L^2$ norm of $(\mathbf{u}_{1,h}-\mathbf{u}_{2,h})$", \
                              r"$L^2$ norm of $(\mathbf{\omega}_{1,h}-\mathbf{\omega}_{2,h})$", \
                              r"$L^2$ norm of $(P_{0,h}-P_{3,h})$", \
                              r"$L^2$ norm of $(\mathbf{A}_{1,h}-\mathbf{A}_{2,h})$", \
                              r"$L^2$ norm of $(\mathbf{B}_{1,h}-\mathbf{B}_{2,h})$", \
                              r"$L^2$ norm of $(\mathbf{j}_{1,h}-\mathbf{j}_{2,h})$"]
    
    filename_list_primal_dual = ['u', \
                                'w', \
                                'p', \
                                'A', \
                                'B', \
                                'j']
    
    for i_pattern in range(len(pattern_list_int)):
        pattern = pattern_list_int[i_pattern]
        titlename = titlename_list_int[i_pattern]
        outputfile = SPATIAL_ERROR_PATH + '/'+'2503_'+filename_list_int[i_pattern]+'.png'
        if pattern == 'Div error of B2':
            ylim = [1e-15, 1e-10]
        elif pattern == 'Div error of u2':
            ylim = [1e-16, 5e-12]
        else:
            ylim = None
        plot_spatial_error(N_lists=N_lists, order_list=order_list, pattern=pattern, titlename=titlename, outputfilename=outputfile, with_comma=True, showtriangle=triangle_list_int[i_pattern],ylim=ylim)
        
    for i_pattern in range(len(pattern_list_half)):
        pattern = pattern_list_half[i_pattern]
        titlename = titlename_list_half[i_pattern]
        outputfile = SPATIAL_ERROR_PATH + '/'+'2503_'+filename_list_half[i_pattern]+'.png'
        if pattern == 'Div error of j2':
            ylim = [1e-15, 1e-10]
        else:
            ylim = None
        plot_spatial_error(N_lists=N_lists, order_list=order_list, pattern=pattern, titlename=titlename, outputfilename=outputfile, with_comma=True, showtriangle=triangle_list_half[i_pattern], ylim=ylim)
        
    for i_pattern in range(len(pattern_list_primal_dual)):
        pattern = pattern_list_primal_dual[i_pattern]
        titlename = title_list_primal_dual[i_pattern]
        outputfile = SPATIAL_ERROR_PATH + '/'+'2503_'+filename_list_primal_dual[i_pattern]+'_primal_dual'+'.png'
        plot_spatial_error(N_lists=N_lists, order_list=order_list, pattern=pattern, titlename=titlename, outputfilename=outputfile, with_comma=False)
        
        
def plot_spatial_error_alltogether(N_list, order, pattern_list, titlename_list, outputfilename):
    
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
            filename = SPATIAL_ERROR_PATH + '/order'+str(order)+'/N'+str(N)+'/Hall_MHD.out'
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
    
    if SPATIAL_ERROR_MODE == 'Herr_all':
        pattern_list_int = ['Hdiv error of u2', \
                        'Hcurl error of w1', \
                        'L2 error of p3', \
                        'Hcurl error of A1', \
                        'Hdiv error of B2', \
                        'Hcurl error of j1']

        titlename_list_int = [r"$\mathbf{H}(\operatorname{div})$ error of $\mathbf{u}_2$", \
                            r"$\mathbf{H}(\operatorname{curl})$ error of $\mathbf{\omega}_1$", \
                                r"$L^2$ error of $p_3$",\
                                r"$\mathbf{H}(\operatorname{curl})$ error of $\mathbf{A}_1$",\
                                r"$\mathbf{H}(\operatorname{div})$ error of $\mathbf{B}_2$",\
                                r"$\mathbf{H}(\operatorname{curl})$ error of $\mathbf{J}_1$"]
        
        pattern_list_half = ['Hcurl error of u1', \
                        'Hdiv error of w2', \
                        'H1 error of p0', \
                        'Hdiv error of A2', \
                        'Hcurl error of B1', \
                        'HDiv error of j2']

        titlename_list_half = [r"$\mathbf{H}(\operatorname{curl})$ error of $\mathbf{u}_1$", \
                        r"$\mathbf{H}(\operatorname{div})$ error of $\mathbf{\omega}_2$", \
                        r"$H^1$ error of $p_0$", \
                        r"$\mathbf{H}(\operatorname{div})$ error of $\mathbf{A}_2$", \
                        r"$\mathbf{H}(\operatorname{curl})$ error of $\mathbf{B}_1$", \
                        r"$\mathbf{H}(\operatorname{div})$ error of $\mathbf{J}_2$"]
    elif SPATIAL_ERROR_MODE == 'Herr_hydro':
        pattern_list_int = ['Hdiv error of u2', \
                        'Hcurl error of w1', \
                        'L2 error of p3']

        titlename_list_int = [r"$\mathbf{H}(\operatorname{div})$ error of $\mathbf{u}_2$", \
                            r"$\mathbf{H}(\operatorname{curl})$ error of $\mathbf{\omega}_1$", \
                                r"$L^2$ error of $p_3$"]
        
        pattern_list_half = ['Hcurl error of u1', \
                    'Hdiv error of w2', \
                    'H1 error of p0']
    
        titlename_list_half = [r"$\mathbf{H}(\operatorname{curl})$ error of $\mathbf{u}_1$", \
                            r"$\mathbf{H}(\operatorname{div})$ error of $\mathbf{\omega}_2$", \
                            r"$H^1$ error of $p_0$"]
        
    elif SPATIAL_ERROR_MODE == 'Herr_mag':
        pattern_list_int = ['Hcurl error of A1', \
                        'Hdiv error of B2', \
                        'Hcurl error of j1']

        titlename_list_int = [r"$\mathbf{H}(\operatorname{curl})$ error of $\mathbf{A}_1$",\
                                r"$\mathbf{H}(\operatorname{div})$ error of $\mathbf{B}_2$",\
                                r"$\mathbf{H}(\operatorname{curl})$ error of $\mathbf{J}_1$"]
        
        pattern_list_half = ['Hdiv error of A2', \
                        'Hcurl error of B1', \
                        'HDiv error of j2']

        titlename_list_half = [r"$\mathbf{H}(\operatorname{div})$ error of $\mathbf{A}_2$", \
                        r"$\mathbf{H}(\operatorname{curl})$ error of $\mathbf{B}_1$", \
                        r"$\mathbf{H}(\operatorname{div})$ error of $\mathbf{J}_2$"]
        
    elif SPATIAL_ERROR_MODE == 'L2err_all':
        pattern_list_int = ['L2 error of u2', \
                        'L2 error of w1', \
                        'L2 error of p3', \
                        'L2 error of A1', \
                        'L2 error of B2', \
                        'L2 error of j1']

        titlename_list_int = [r"$L^2$ error of $\mathbf{u}_2$", \
                            r"$L^2$ error of $\mathbf{\omega}_1$", \
                                r"$L^2$ error of $p_3$",\
                                r"$L^2$ error of $\mathbf{A}_1$",\
                                r"$L^2$ error of $\mathbf{B}_2$",\
                                r"$L^2$ error of $\mathbf{J}_1$"]
                                
        pattern_list_half = ['L2 error of u1', \
                    'L2 error of w2', \
                    'L2 error of p0', \
                    'L2 error of A2', \
                    'L2 error of B1', \
                    'L2 error of j2']
    
        titlename_list_half = [r"$L^2$ error of $\mathbf{u}_1$", \
                            r"$L^2$ error of $\mathbf{\omega}_2$", \
                            r"$L^2$ error of $p_0$", \
                            r"$L^2$ error of $\mathbf{A}_2$", \
                            r"$L^2$ error of $\mathbf{B}_1$", \
                            r"$L^2$ error of $\mathbf{J}_2$"]
    
    pattern_list_primal_dual = ['L2 error of u:', \
                                'L2 error of w:', \
                                'L2 error of p:', \
                                'L2 error of A:', \
                                'L2 error of B:', \
                                'L2 error of j:']
    title_list_primal_dual = [r"$L^2$ error of $\mathbf{u}$", \
                              r"$L^2$ error of $\mathbf{\omega}$", \
                              r"$L^2$ error of $p$", \
                              r"$L^2$ error of $\mathbf{A}$", \
                              r"$L^2$ error of $\mathbf{B}$", \
                              r"$L^2$ error of $\mathbf{J}$"]
    
    N_list = [4,8,16,32]
    order = 2
    
    plot_spatial_error_alltogether(N_list=N_list, order=order, pattern_list=pattern_list_int, 
    titlename_list=titlename_list_int, outputfilename=SPATIAL_ERROR_PATH + '/spatial_error_int_order'+str(order)+'.png')
    plot_spatial_error_alltogether(N_list=N_list, order=order, pattern_list=pattern_list_half, 
    titlename_list=titlename_list_half, outputfilename=SPATIAL_ERROR_PATH + '/spatial_error_half_order'+str(order)+'.png')
    plot_spatial_error_alltogether(N_list=N_list, order=order, pattern_list=pattern_list_primal_dual, 
    titlename_list=title_list_primal_dual, outputfilename=SPATIAL_ERROR_PATH + '/spatial_error_primal_dual_order'+str(order)+'.png')

if __name__ == '__main__':
    
    # plot_temporal(N=32, order=2)
    # plot_temporal(N=8, order=3)
    
    # plot_spatial_all()
    
    
    plot_spatial()
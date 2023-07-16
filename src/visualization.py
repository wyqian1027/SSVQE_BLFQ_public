from matplotlib import pyplot as plt
import numpy as np


# Visualizing parameters and exp_val lists
def plot_param_list(vals, fs=15):
    ''' Plot param list history '''
    vals = np.array(vals)
    n, m = vals.shape
    x = np.arange(1, n+1)   
    fig, axs = plt.subplots(m, 2, squeeze=False, figsize=(16,4*m))
    colors = ['r', 'g', 'b', 'orange', 'c', 'm', 'y', 'k']*(m//7+2)
    for i in range(m):
        r, c = i//2, i%2
        axs[r,c].plot(x, vals[:,i], marker='o', linestyle='solid', color=colors[i % len(colors)], markersize=2, label=f'params[{i}]')
        axs[r,c].set_ylabel("", fontsize=fs) 
        axs[r,c].set_xlabel("", fontsize=fs)
        axs[r,c].tick_params(axis='both', which='major', labelsize=fs) # choose 'both', 'x', 'y'
        axs[r,c].legend(fontsize=fs)
    for i in range(m, m*2):
        fig.delaxes(axs.flatten()[i]) # clean up unused axs
    plt.show()

def plot_exp_val_list(vals, errors=[], labels=[], fs=15, figsize=(8,6), legend_loc='best', ref_evals=[]):
    ''' Plot exp_val history '''
    m = len(vals); 
    if labels == []: labels = [f'Cost Exp {i}' for i in range(m)]
    assert len(labels) == m
    fig, axs = plt.subplots(1, 1, squeeze=False, figsize=figsize)
    colors = ['r', 'g', 'b', 'orange', 'c', 'm', 'y', 'k']
    
    xmax = 0
    for i in range(m):
        tvals = vals[i]
        n = len(tvals)
        max_error_shown = 20
        xmax = max(n+10, xmax)
        err_int = max(n//max_error_shown, 1)
        iters = np.arange(1, n+1)
        axs[0,0].plot(iters, tvals[:,0], linestyle='solid', lw=1, color=colors[i % len(colors)],  label=labels[i])#marker='o', markersize=1)
        if errors != []: 
            terrs = errors[i]
            axs[0,0].errorbar(iters[0:n+1:err_int], tvals[0:n+1:err_int,0], yerr=terrs[0:n+1:err_int,0], 
                              color=colors[i], capsize=3, capthick=1, elinewidth=0.5, fmt='o', markersize=0)
    axs[0,0].set_ylabel("Cost expectation", fontsize=fs) 
    axs[0,0].set_xlabel("Iter steps", fontsize=fs)
    axs[0,0].tick_params(axis='both', which='major', labelsize=fs) # choose 'both', 'x', 'y'

    for i, y in enumerate(ref_evals):
        axs[0,0].axhline(y=y, xmin=-10, xmax=xmax, lw=1, ls='--', color='gray', label='Exact' if i==0 else None)
    axs[0,0].legend(fontsize=fs, loc=legend_loc)
    plt.show()


from qiskit.quantum_info.states.densitymatrix import DensityMatrix

def plot_state_hinton2(state, figsize=(8, 6), fs=14, save_fig_name=""):
    # Figure data
    rho = DensityMatrix(state)
    num = rho.num_qubits
    max_weight = 2 ** np.ceil(np.log(np.abs(rho.data).max()) / np.log(2))
    datareal = np.real(rho.data)
    dataimag = np.imag(rho.data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    column_names = [bin(i)[2:].zfill(num) for i in range(2**num)]
    row_names = [bin(i)[2:].zfill(num) for i in range(2**num)]
    ly, lx = datareal.shape
    # Real
    if ax1:
        ax1.patch.set_facecolor("gray")
        ax1.set_aspect("equal", "box")
        ax1.xaxis.set_major_locator(plt.NullLocator())
        ax1.yaxis.set_major_locator(plt.NullLocator())

        for (x, y), w in np.ndenumerate(datareal):
            color = "white" if w > 0 else "black"
            size = np.sqrt(np.abs(w) / max_weight)
            rect = plt.Rectangle(
                [0.5 + x - size / 2, 0.5 + y - size / 2],
                size,
                size,
                facecolor=color,
                edgecolor=color,
            )
            ax1.add_patch(rect)

        ax1.set_xticks(0.5 + np.arange(lx))
        ax1.set_yticks(0.5 + np.arange(ly))
        ax1.set_xlim([0, lx])
        ax1.set_ylim([ly, 0])
        ax1.set_yticklabels(row_names, fontsize=fs)
        ax1.set_xticklabels(column_names, fontsize=fs, rotation=90)
        ax1.invert_yaxis()
        ax1.set_title("Real", fontsize=fs)
    # Imaginary
    if ax2:
        ax2.patch.set_facecolor("gray")
        ax2.set_aspect("equal", "box")
        ax2.xaxis.set_major_locator(plt.NullLocator())
        ax2.yaxis.set_major_locator(plt.NullLocator())

        for (x, y), w in np.ndenumerate(dataimag):
            color = "white" if w > 0 else "black"
            size = np.sqrt(np.abs(w) / max_weight)
            rect = plt.Rectangle(
                [0.5 + x - size / 2, 0.5 + y - size / 2],
                size,
                size,
                facecolor=color,
                edgecolor=color,
            )
            ax2.add_patch(rect)

        ax2.set_xticks(0.5 + np.arange(lx))
        ax2.set_yticks(0.5 + np.arange(ly))
        ax2.set_xlim([0, lx])
        ax2.set_ylim([ly, 0])
        ax2.set_yticklabels(row_names, fontsize=fs)
        ax2.set_xticklabels(column_names, fontsize=fs, rotation=90)
        ax2.invert_yaxis()
        ax2.set_title("Imag", fontsize=fs)
    
    fig.tight_layout()  
    if save_fig_name != "":
        fig.savefig(save_fig_name, dpi=fig.dpi, bbox_inches='tight')

    


def plot_pdfs(blfq_file, xs, pdfs, errors=[], labels=[r'$q_\pi(x)$', r'$q_\rho(x)$'],
              fs=24, figsize=(8,6), 
              legend_loc='best',
              text_loc=(0.5, 1), text='',
              no_left_label=False,
              save_fig_name=""
             ):
    # if 0 not in xs and 1 not in xs:
    #     xs = np.concatenate([[0], xs, [1]])
    #     pdfs = [np.concatenate([[0], pdf, [0]]) for pdf in pdfs]
    #     if errors: errors = [np.concatenate([[0], error, [0]]) for error in errors]
        # print(xs, pdfs, errors)
    fig, axs = plt.subplots(1, 1, squeeze=False, figsize=figsize)
    # plt.rcParams["font.size"] = fs
    # plt.rcParams["font.family"] = "Times New Roman"
    # plt.rcParams["font.sans-serif"] = ["Times New Roman"]
    colors = ['r', 'g', 'b', 'orange', 'c', 'm', 'y', 'k']
    linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
    markers = ['o', 's', '<', '>']
    xmax = 0
    for axis in ['top','bottom','left','right']:
        axs[0,0].spines[axis].set_linewidth(1.5)


    for i, pdf in enumerate(pdfs):
        # axs[0,0].plot(xs, pdf, linestyle=linestyles[i], lw=2, color=colors[i % len(colors)], 
        #               markersize=8, marker=markers[i], 
        #               label=labels[i],
        #               zorder=i)
        if errors != []: 
            terrs = errors[i]
            axs[0,0].errorbar(xs, pdf, yerr=terrs, lw=0,
                              color=colors[2*i], capsize=10, capthick=1, elinewidth=0.5, marker=markers[i], markersize=8-2*i,
                              label=labels[i], zorder=i)
    if not no_left_label: 
        axs[0,0].set_ylabel(r"$q(x)$", fontsize=fs) 
    axs[0,0].set_xlabel(r"$x$", fontsize=fs)
    epslon = 0.05
    axs[0,0].set_ylim(bottom=0-epslon, top=2.0+epslon)
    axs[0,0].set_xlim(left=0-epslon, right=1+epslon)
    axs[0,0].tick_params(axis='both', which='major', labelsize=fs) # choose 'both', 'x', 'y'

    axs[0,0].axhline(y=1, xmin=0, xmax=1, ls=linestyles[2], lw=2, color='gray', #label='Exact' if i==0 else None,
                    zorder=10)
    exact_r = 0.01
    exact_xs = np.arange(0,1+exact_r,exact_r)
    pdf_pion = [blfq_file.get_pdf(x, idx=0) for x in exact_xs]
    pdf_rho =  [blfq_file.get_pdf(x, idx=1) for x in exact_xs]
    pdf_pion_dict = {float(np.round(x, 4)): pdf_pion[i] for i, x in enumerate(exact_xs)}
    pdf_rho_dict = {float(np.round(x, 4)): pdf_rho[i] for i, x in enumerate(exact_xs)}
    axs[0,0].plot(exact_xs, pdf_pion, linestyle='solid', lw=2, color='k', 
                      markersize=0, marker='o', label=r'$\pi$ (exact)', zorder=5)
    axs[0,0].plot(exact_xs, pdf_rho,  linestyle='dashed', lw=2, color='k', 
                      markersize=0, marker='o', label=r'$\rho$ (exact)', zorder=5)
    
    axs[0,0].legend(fontsize=fs, loc=legend_loc)
    # axs.plot()
    axs[0,0].ticklabel_format(useOffset=False)

    # font = {'family': "Times New Roman",
    #     'color':  'darkred',
    #     'weight': 'normal',
    #     'size': fs,
    # }    
    axs[0,0].text(*text_loc, text)#, fontdict=font)

    s1 = s2 = 0
    # print(pdf_pion_dict)
    for i, x in enumerate(xs):
        x = float(np.round(x, 4))
        s1 += (pdfs[0][i] - pdf_pion_dict[x])**2/pdf_pion_dict[x]
        s2 += (pdfs[1][i] - pdf_rho_dict[x])**2/pdf_rho_dict[x]
    print(f"chiSquare pion = {s1:.3f}")
    print(f"chiSquare rho = {s2:.3f}")
    print(f"chiSquare both = {s1+s2:.3f}")

    if save_fig_name != "":
        fig.savefig(save_fig_name, dpi=fig.dpi, bbox_inches='tight')

    plt.show()
"""
Functions for plotting
"""


def plot_comparison(X, Y, xlabel = None, ylabel = None, filename = None):
    """
    Plots two sets of data against each other

    :param X: First set of data points
    :type X: array
    :param Y: Second set of data points
    :type Y: array
    :param xlabel: Label for x-axis
    :type xlabel: string
    :param ylabel: Label for y-ayis
    :type ylabel: string
    :param filename: File to save the plot to. If '' the plot is shown instead of saved.
                     If the dimensionality of y is higher than 1, the filename will be prefixed
                     by the dimension.
    :type filename: string

    """
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        from matplotlib import rc
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Plotting functions require the modules 'seaborn' and 'matplotlib'")

    # set matplotlib defaults
    sns.set(font_scale=2.)
    sns.set_style("whitegrid",{'grid.color':'.92','axes.edgecolor':'0.92'})
    rc('text', usetex=False)

    # convert to arrays
    x = np.asarray(X)
    y = np.asarray(Y)

    # NOTE: If energy is not kcal/mol this might not be sensible
    min_val = int(min(x.min(), y.min()) - 1)
    max_val = int(max(x.max(), y.max()) + 1)

    fig, ax = plt.subplots()

    # Create the scatter plot
    ax.scatter(x, y)

    # Set limits
    ax.set_xlim([min_val,max_val])
    ax.set_ylim([min_val,max_val])

    lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
            ]

    # now plot both limits against eachother
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Set labels
    if not is_none(xlabel):
        if is_string(xlabel): 
            ax.set_xlabel(xlabel)
        else:
            raise InputError("Wrong data type of variable 'xlabel'. Expected string, got %s" % str(xlabel))
    if not is_none(ylabel):
        if is_string(ylabel): 
            ax.set_ylabel(ylabel)
        else:
            raise InputError("Wrong data type of variable 'ylabel'. Expected string, got %s" % str(ylabel))

    if x.ndim != 1 or y.ndim != 1:
        raise InputError("Input must be one dimensional")

    # Plot or save
    if is_none(filename):
        plt.show()
    elif is_string(filename):
        if "." not in filename:
            filename = filename + ".pdf"
        plt.savefig(filename, pad_inches=0.0, bbox_inches = "tight", dpi = 300) 
    else:
        raise InputError("Wrong data type of variable 'filename'. Expected string")



#plt.errorbar(unique_costs, lol, yerr = lal, fmt = 'o-')
#plt.plot(unique_costs, lul, 'o-')
#plt.ylim([0.3,20])
#plt.xscale('log')
#plt.yscale('log')
#plt.show()

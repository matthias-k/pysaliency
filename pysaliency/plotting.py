from __future__ import absolute_import, print_function, division, unicode_literals

try:
    import matplotlib.pyplot as plt
except ImportError:
    # If matplotlib is not there, just ignore it
    pass

import numpy as np
from scipy.ndimage import zoom


def plot_information_gain(information_gain, ax=None, color_range = None, image=None, frame=False,
                          thickness = 1.0, zoom_factor=1.0, threshold=0.05, rel_levels=None,
                          alpha=0.5, color_offset = 0.25, plot_color_bar=True):
    """
    Create pixel space information gain plots as in the paper.

    Parameters:
    -----------

    information gain: the information gain to plot.
    ax: the matplotlib axes object to use. If none, use current axes.
    color_range: Full range of colorbar
    """
    if ax is None:
        ax = plt.gca()
    ig = information_gain

    if zoom_factor != 1.0:
        ig = zoom(ig, zoom_factor, order=0)

    if color_range is None:
        color_range = (ig.min(), ig.max())
    if not isinstance(color_range, (tuple, list)):
        color_range = (-color_range, color_range)

    color_total_max = max(np.abs(color_range[0]), np.abs(color_range[1]))

    if image is not None:
        if image.ndim == 3:
            image = image.sum(axis=-1)
        ax.imshow(image, alpha=0.3)

    if rel_levels is None:
        rel_levels = [0.1, 0.4, 0.7]

    # from https://stackoverflow.com/questions/8580631/transparent-colormap
    cm = plt.cm.get_cmap('RdBu')
    cm._init()
    alphas = (np.abs(np.linspace(-1.0, 1.0, cm.N)))
    alphas = np.ones_like(alphas)*alpha
    cm._lut[:-3, -1] = alphas

    levels = []
    colors = []

    min_val = np.abs(ig.min())
    max_val = np.abs(ig.max())

    total_max = max(min_val, max_val)

    def get_color(val):
        # value relative -1 .. 1
        rel_val = val / color_total_max
        # shift around 0
        rel_val = (rel_val + np.sign(rel_val) * color_offset) / (1+color_offset)
        # transform to 0 .. 1
        rel_val = (0.5 + rel_val / 2)
        return cm(rel_val)

    if min_val / total_max > threshold:
        for l in [1.0]+rel_levels[::-1]:
            val = -l*min_val
            levels.append(val)

            colors.append(get_color(val))
    else:
        levels.append(-total_max)
        colors.append('white')

    # We want to use the color from the value nearer to zero
    colors = colors[1:]
    colors.append((1.0, 1.0, 1.0, 0.0))

    if max_val / total_max > threshold:
        for l in rel_levels+[1.0]:
            val = l*max_val
            levels.append(val)

            colors.append(get_color(val))
    else:
        levels.append(total_max)

    #print rel_vals
    ax.contourf(ig, levels=levels,
                colors=colors,
                vmin=-color_total_max, vmax=color_total_max
                )
    ax.contour(ig, levels=levels,
               # colors=colors,
               #          vmin=-color_range, vmax=color_range
               colors = 'gray',
               linestyles='solid',
               linewidths=0.6*thickness
               )

    if plot_color_bar:
        ## Draw color range bar
        h = 100
        w = 10
        t = np.empty((100, 10, 4))
        for y in range(h):
            for x in range(w):
                val = (y/h) * (color_range[1] - color_range[0]) + color_range[0]
                color = np.asarray(get_color(val))
                if not -min_val <= val <= max_val:
                    color[-1] *= 0.4
                else:
                    color[-1] = 1
                t[y, x, :] = color

        ax.imshow(t, extent=(0.95*ig.shape[1], 0.98*ig.shape[1],
                             0.1*ig.shape[0], 0.9*ig.shape[0]))

    ax.set_xlim(0, ig.shape[1])
    ax.set_ylim(ig.shape[0], 0)

    if frame:
        # Just a frame
        ax.set_xticks([])
        ax.set_yticks([])
        [i.set_linewidth(i.get_linewidth()*thickness) for i in ax.spines.itervalues()]
    else:
        ax.set_axis_off()


def normalize_log_density(log_density):
    """ convertes a log density into a map of the cummulative distribution function.
    """
    density = np.exp(log_density)
    flat_density = density.flatten()
    inds = flat_density.argsort()[::-1]
    sorted_density = flat_density[inds]
    cummulative = np.cumsum(sorted_density)
    unsorted_cummulative = cummulative[np.argsort(inds)]
    return unsorted_cummulative.reshape(log_density.shape)

def visualize_distribution(log_densities, ax = None):
    if ax is None:
        ax = plt.gca()
    t = normalize_log_density(log_densities)
    img = ax.imshow(t, cmap=plt.cm.viridis)
    levels = levels=[0, 0.25, 0.5, 0.75, 1.0]
    cs = ax.contour(t, levels=levels, colors='black')
    #plt.clabel(cs)

    return img, cs

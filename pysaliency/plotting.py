from __future__ import absolute_import, print_function, division, unicode_literals

try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
except ImportError:
    # If matplotlib is not there, just ignore it
    pass

from boltons.iterutils import windowed
import numpy as np
from scipy.ndimage import zoom

from .utils import remove_trailing_nans


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

def visualize_distribution(log_densities, ax=None, levels=None, level_colors='black', cmap=plt.cm.viridis):
    if ax is None:
        ax = plt.gca()
    t = normalize_log_density(log_densities)
    img = ax.imshow(t, cmap=cmap)
    if levels is None:
        levels = [0, 0.25, 0.5, 0.75, 1.0]
    cs = ax.contour(t, levels=levels, colors=level_colors)
    #plt.clabel(cs)

    return img, cs


def advanced_arrow(x, y, dx, dy, linewidth=1, headwidth=3, headlength=None, linestyle='-', ax=None, color=None, zorder=None, alpha=1.0, arrow_style='-|>'):
    """careful: this uses axes data and figure inches coordinates. They can change if the axes limits are changed, which
    makes the arrow look strange"""

    if ax is None:
        ax = plt.gca()

    if headlength is None:
        headlength = 1.5 * headwidth

    trans_data_to_inches = mpl.transforms.composite_transform_factory(ax.transData, ax.get_figure().dpi_scale_trans.inverted())
    start = (x, y)
    end = (x + dx, y + dy)
    #ax.scatter([x, x+dx], [y, y+dy], 1, color='black', zorder=100)
    start_inches = trans_data_to_inches.transform(start)
    end_inches = trans_data_to_inches.transform(end)

    distance_inches = end_inches - start_inches
    distance_inches_length = np.sqrt(np.sum(np.square(distance_inches)))

    # make sure head is not longer than total length
    headlength = min(headlength, distance_inches_length * 72)

    new_distance_inches = (distance_inches_length - headlength / 72)
    new_end_inches = start_inches + distance_inches * (new_distance_inches / distance_inches_length)
    new_end_data = trans_data_to_inches.inverted().transform(new_end_inches)
    line = ax.plot(
        [x, new_end_data[0]],
        [y, new_end_data[1]],
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        solid_capstyle="butt", # otherwise line is slightly too long
        zorder=zorder,
        alpha=alpha,
    )

    color = line[0].get_color()

    arrow = mpl.patches.FancyArrowPatch(
        (x, y), (x+dx,y+dy),
        arrowstyle=mpl.patches.ArrowStyle(
            arrow_style,
            head_width=headwidth,
            head_length=headlength,
        ),
        mutation_scale=1,
        shrinkA=0,
        shrinkB=0,
        linewidth=0,
        color=color,
        alpha=alpha,
        zorder=zorder
    )
    ax.add_patch(arrow)


def plot_scanpath(stimuli, fixations, index, ax=None, show_history=True, show_current_fixation=True, visualize_next_saccade=False, include_next_saccade=False, history_color='red', next_saccade_color='cyan', current_fixation_size=3, fixation_color='blue',  history_alpha=1.0, history_linestyle='-', saccade_width=2, fixation_size=10):
    if ax is None:
        ax = plt.gca()
    x_hist = list(remove_trailing_nans(fixations.x_hist[index]))
    y_hist = list(remove_trailing_nans(fixations.y_hist[index]))

    if include_next_saccade:
        assert visualize_next_saccade is False
        x_hist.append(fixations.x[index])
        y_hist.append(fixations.y[index])

    headwidth = 1.5 * saccade_width
    headlength = 3 * saccade_width

    if show_history:
        for (x1, x2), (y1, y2) in zip(windowed(x_hist, 2), windowed(y_hist, 2)):
            advanced_arrow(x1, y1, x2-x1, y2-y1,
                linewidth=saccade_width,
                headwidth=headwidth,
                headlength=headlength,
                color=history_color,
                linestyle=history_linestyle,
                zorder=10,
                alpha=history_alpha,
                ax=ax,
            )

        ax.scatter(x_hist, y_hist, fixation_size, color=fixation_color, zorder=40)


    if show_current_fixation:
        x1 = x_hist[-1]
        y1 = y_hist[-1]
        ax.scatter([x1], [y1], 3, color='red', zorder=10,)

    if visualize_next_saccade:
        x1 = x_hist[-1]
        y1 = y_hist[-1]

        x2 = fixations.x[index]
        y2 = fixations.y[index]

        advanced_arrow(
            x1, y1, x2-x1, y2-y1,
            linewidth=saccade_width,
            headwidth=headwidth,
            headlength=headlength,
            color=next_saccade_color,
            linestyle=(0, (2,1)),
            zorder=10,
            ax=ax,
        )

        ax.scatter([x2], [y2], fixation_size, color=fixation_color, zorder=40)
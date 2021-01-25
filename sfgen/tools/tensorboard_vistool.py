import copy
import math
from pprint import pprint


import numpy as np
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd

from IPython.display import display, HTML

from sfgen.tools.utils import flatten_dict



class VisDataObject:
    """docstring for VisDataObject"""
    def __init__(
        self,
        tensorboard_data,
        settings={},
        color='',
        label='',
        marker='',
        linestyle='',
        alpha=0):


        self.tensorboard_data = tensorboard_data
        self.settings_df = tensorboard_data.settings_df
        self.data_df = tensorboard_data.data_df
        self.settings = settings

        self.color = color
        self.label = label
        self.marker = marker
        self.linestyle = linestyle
        self.alpha = alpha
        self.colors = self.defaultcolors()
        self.colororder = self.defaultcolororder()


    def plot_data(self, ax, key, individual_lines=False, **kwargs):
        if individual_lines:
            self.plot_individual_lines(ax, key, **kwargs)
        else:
            self.plot_mean_stderr(ax, key, **kwargs)

    def plot_mean_stderr(self, ax, key, datapoint=0, xlabel_key=None, stderr=True, settings_idx=-1, label_settings=[], **kwargs):
        df, all_data = self.tensorboard_data[key]
        settings = df['experiment_settings'].tolist()

        if settings_idx > -1:
            settings = [settings[settings_idx]]
        else:
            settings_idx = 0

        for idx, setting in enumerate(settings):
            # this is a list of all the lines
            data = all_data[setting]

            # -----------------------
            # create array of data
            # -----------------------
            lengths = [len(d) for d in data]
            min_length = min(lengths)
            joint = np.array([d[:min_length] for d in data])

            # -----------------------
            # compute mean/std(err)
            # -----------------------
            mean = joint.mean(0)
            err = joint.std(0)
            if xlabel_key is None:
                x = np.arange(len(mean))
            else:
                # use key as x-axis
                _, xdata = self.tensorboard_data[xlabel_key]
                # all same, so pick 1st
                x = xdata[setting][0]
                x= x[:len(mean)]

            if stderr:
                err = err/np.sqrt(len(data))

            # -----------------------
            # compute plot/fill kwargs
            # -----------------------
            # load default
            plot_kwargs, fill_kwargs = self.default_plot_fill_kwargs()
            # update with filter settings
            self.update_plot_fill_kwargs(
                plot_kwargs, fill_kwargs,
                label_settings=label_settings,
                datapoint=datapoint,
                idx=settings_idx,
                )

            ax.plot(x, mean, **plot_kwargs)
            ax.fill_between(x, mean-err, mean+err, **fill_kwargs)

    def plot_individual_lines(self, ax, **kwargs):
        raise NotImplementedError

    def label_from_settings(self, columns=[], idx=0):
        """Create a lable using the values in the columns provided. idx indicates which index of the column
        
        Args:
            columns (list, optional): Description
            idx (int, optional): Description
        
        Returns:
            TYPE: Description
        """
        if isinstance(columns, dict):
            columns = flatten_dict(columns, sep=":")
            columns = list(columns.keys())

        key0 = columns[0]
        val0 = self.tensorboard_data.settings_df[key0].to_numpy()[idx]
        label = f"{key0}={val0}"
        for key in columns[1:]:
            val = self.tensorboard_data.settings_df[key].to_numpy()[idx]
            label += f", {key}={val}"
        return label


    def update_plot_fill_kwargs(self, plot_kwargs, fill_kwargs, datapoint=0, label_settings=[], idx=0):
        """Update plot, fill kwargs using settings for this data object
        
        Args:
            plot_kwargs (dict): 
            fill_kwargs (dict): 
            label_settings (list, optional): which settings to use to construct label. not yet implemented.
        """
        if self.color:
            color = self.color
        else:
            color = self.colororder[datapoint]

        plot_kwargs['color'] = self.colors.get(color, color)
        fill_kwargs['color'] = self.colors.get(color, color)


        if self.linestyle:
            plot_kwargs['linestyle'] = self.linestyle
        if self.marker:
            plot_kwargs['marker'] = self.marker
            # plot_kwargs['markersize'] = markersize

        if self.alpha:
          plot_kwargs['alpha'] = self.alpha
          if 'alpha' in fill_kwargs:
            fill_kwargs['alpha'] = fill_kwargs['alpha']*self.alpha
          else:
            fill_kwargs['alpha'] = self.alpha

        if self.label:
            plot_kwargs['label'] = self.label
        else:
            plot_kwargs['label'] = self.label_from_settings(columns=label_settings, idx=idx)


    @staticmethod
    def lineargs(): 
        return [
            dict(
                linestyle='-.',
                # marker='o',
                ),
            dict(
                linestyle='--',
                # marker='X',
                ),
            dict(
                linestyle='-.',
                # marker='^',
                ),
            dict(
                linestyle='--',
                # marker='x',
                ),
        ]

    @staticmethod
    def default_plot_fill_kwargs():
        plot_kwargs=dict(
            linewidth=4,
            )
        fill_kwargs=dict(
            alpha=.2,
            )

        return plot_kwargs, fill_kwargs

    @staticmethod
    def defaultcolors():
        return dict(
            grey = '#99999e',
            dark_grey = "#363535",
            # =========================================
            # red
            # =========================================
            red='#d9432f',
            light_red='#FF0000',
            dark_red='#8B0000',
            # =========================================
            # Purple
            # =========================================
            light_purple = "#9d81e3",
            purple = "#7C66B4",
            dark_purple = "#4c2d9c",
            # =========================================
            # green
            # =========================================
            light_green = "#2da858",
            green = "#489764",
            dark_green = "#117333",
            # =========================================
            # blue
            # =========================================
            light_blue = "#7dc8e3",
            grey_blue = "#57c4fa",
            blue = "#5C94E1",
            dark_blue = "#1152ad",
            ##
            # Orange
            #
            orange = "#f5a742",
        )

    @staticmethod
    def defaultcolororder():
        return [
            'red', 'blue', 'purple', 'orange', 'green', 'grey', 'dark_grey',
            'dark_red', 'dark_blue', 'dark_purple', 'dark_orange', 'dark_green',
            'light_red', 'light_blue', 'light_purple', 'light_orange', 'light_green',
        ]


class Vistool:
    def __init__(self,
        tensorboard_data,
        plot_settings=[],
        metadata_stats=['num_seeds'],
        metadata_settings_dict={},
        metadata_settings_list=['settings'],
        filter_key=None,
        key_with_legend=None,
        plot_data_kwargs={},
        common_settings={},
        ):

        self.tensorboard_data = tensorboard_data
        self.plot_settings = plot_settings
        self.key_with_legend = key_with_legend
        self.plot_data_kwargs = dict(plot_data_kwargs)
        self.filter_key = filter_key
        self.common_settings = dict(common_settings)

        # default settings for displaying metadata
        self.metadata_stats = metadata_stats
        # combine list and dict together
        self.metadata_settings = list(metadata_settings_list)
        self.metadata_settings.extend(
            list(flatten_dict(metadata_settings_dict, sep=':').keys())
            )

    def plot_filters(self,
        # ======================================================
        # Arguments for getting matches for data
        # ======================================================
        data_filters,
        filter_key=None,
        common_settings={},
        topk=1,
        filter_kwargs={},
        # ======================================================
        # Arguments for displaying dataframe
        # ======================================================
        display_settings=[],
        display_stats=[],
        # ======================================================
        # Arguments for Plot Keys
        # ======================================================
        maxcols=2,
        key_with_legend=None,
        subplot_kwargs={},
        plot_data_kwargs={},
        fig_kwargs={},
        # ======================================================
        # misc.
        # ======================================================
        verbosity=1,
        ):

        # ======================================================
        # get objects
        # ======================================================
        vis_objects = get_vis_objects(
            tensorboard_data=self.tensorboard_data,
            data_filters=data_filters,
            common_settings=common_settings if common_settings else self.common_settings,
            filter_key=filter_key if filter_key else self.filter_key,
            topk=topk,
            filter_kwargs=filter_kwargs,
            verbosity=verbosity,
            )

        # ======================================================
        # display pandas dataframe with relevant data
        # ======================================================
        if verbosity:
            display_metadata(
                vis_objects=vis_objects,
                settings=display_settings if display_settings else self.metadata_settings,
                stats=display_stats if display_stats else self.metadata_stats,
                )

        plot_data_kwargs = copy.deepcopy(plot_data_kwargs)
        fig_kwargs = copy.deepcopy(fig_kwargs)
        subplot_kwargs = copy.deepcopy(subplot_kwargs)
        for k in range(topk):
            plot_settings = copy.deepcopy(self.plot_settings)
            # -----------------------
            # add K to titles for identification
            # -----------------------
            if topk > 1:
                for info in plot_settings:
                    info['title'] = f"{info['title']} (TopK={k})"

                # indicate which settings to use
                plot_data_kwargs['settings_idx'] = k

            plot_keys(
                vis_objects=vis_objects,
                plot_settings=plot_settings,
                maxcols=maxcols,
                subplot_kwargs=subplot_kwargs,
                plot_data_kwargs=plot_data_kwargs,
                fig_kwargs=fig_kwargs,
                key_with_legend=key_with_legend if key_with_legend else self.key_with_legend,
                )



def get_vis_objects(tensorboard_data, data_filters, common_settings, filter_key, topk=1, filter_kwargs={}, verbosity=0):
    # copy data so can reuse
    data_filters = copy.deepcopy(data_filters)
    common_settings = copy.deepcopy(common_settings)
    common_settings = flatten_dict(common_settings, sep=':')
    
    vis_objects = []
    for data_filter in data_filters:
        data_filter['settings'] = flatten_dict(data_filter['settings'], sep=':')
        data_filter['settings'].update(common_settings)

        match = tensorboard_data.filter_topk(
            key=filter_key,
            filters=[data_filter['settings']],
            topk=topk,
            verbose=verbosity,
            **filter_kwargs,
        )[0]

        if match is not None:
            vis_object = VisDataObject(
                tensorboard_data=match,
                **data_filter,
                )
            vis_objects.append(vis_object)

    return vis_objects

def display_metadata(vis_objects, settings=[], stats=[], data_key=None):
    """Display metadata about config (settings) or some stats, e.g. nonzero seeds, number of seeds, etc. (stats).
    
    Args:
        settings (list, optional): config settings
        stats (list, optional): run statistics
    """
    # this enable you to use an empty dictionary to populate settings
    if isinstance(settings, dict):
        settings = flatten_dict(settings, sep=':')
        settings = list(settings.keys())

    settings_df = pd.concat([o.settings_df for o in vis_objects])[settings]
    # display(settings_df)

    if data_key is None:
        data_key = vis_objects[0].tensorboard_data.keys_like('return')[0]
    data_df = pd.concat([o.data_df[data_key]
                            for o in vis_objects])[stats]
    display(pd.concat([settings_df, data_df], axis=1))


# ======================================================
# Plotting functions
# ======================================================
def make_subplots(num_plots, maxcols=2, unit=8, **kwargs):
    #number of subfigures
    ncols = min(num_plots, maxcols)

    if num_plots % 2 == 0:
        nrows = num_plots//2
    else:
        nrows = (num_plots//2)+1

    if not 'figsize' in kwargs:
        height=nrows*unit
        width=ncols*unit
        kwargs['figsize'] = (width, height)

    fig, axs = plt.subplots(nrows, ncols, **kwargs)

    if num_plots > 1:
      axs = axs.ravel()
    else:
      axs = [axs]

    return fig, axs

def plot_keys(
    vis_objects,
    keys=[],
    plot_settings=[],
    maxcols=2,
    subplot_kwargs={},
    plot_data_kwargs={},
    fig_kwargs={},
    key_with_legend=None,
    ):
    if len(keys) > 0 and len(plot_settings) > 0:
        raise RuntimeError("Either only provide keys or plot infos list of dictionaries, where each dict also has a key. Don't provide both")
    # convert, so can use same code for both
    if len(keys):
        plot_settings = [dict(key=k) for k in keys]


    fig, axs = make_subplots(
        num_plots=len(plot_settings),
        maxcols=maxcols,
        **subplot_kwargs,
        )

    # ======================================================
    # plot data
    # ======================================================
    if key_with_legend is None:
        key_with_legend = plot_settings[0]['key']

    for ax, plot_info in zip(axs, plot_settings):
        key = plot_info['key']
        for idx, vis_object in enumerate(vis_objects):
            vis_object.plot_data(ax, key=key,
                datapoint=idx,
                **plot_data_kwargs)

        finish_plotting_ax(
            ax=ax,
            plot_info=plot_info,
            plot_legend=key_with_legend==key,
            **fig_kwargs,
            )

    plt.show()

def finish_plotting_ax(
    ax,
    plot_info,
    title_size=22,
    title_loc='center',
    minor_text_size=18,
    legend_text_size=16,
    grid_kwargs={},
    ysteps=10,
    plot_legend=True,
    ):
    ax.yaxis.label.set_size(minor_text_size)
    ax.xaxis.label.set_size(minor_text_size)
    ax.tick_params(axis='both', which='major', labelsize=minor_text_size)

    if not grid_kwargs:
        grid_kwargs = copy.deepcopy(default_grid_kwargs())
    ax.grid(**grid_kwargs)


    xlim = plot_info.get('xlim', None)
    if xlim:
      ax.set_xlim(*xlim)

    ylim = plot_info.get('xlim', None)
    if ylim:
      ax.set_ylim(*ylim)
      length = ylim[1]-ylim[0]
      step = length/ysteps
      ax.set_yticks(np.arange(ylim[0], ylim[1]+step, step))


    ylabel = plot_info.get("ylabel", None)
    if ylabel:
      ax.set_ylabel(ylabel)

    xlabel = plot_info.get("xlabel", None)
    if xlabel:
      ax.set_xlabel(xlabel)

    title = plot_info.get("title", None)
    if title:
      ax.set_title(title, fontsize=title_size, loc=title_loc)

    if plot_legend:
      ax.legend(prop={'size': legend_text_size})

def default_grid_kwargs():
    return dict(
      linestyle="-.",
      linewidth=.5,
    )





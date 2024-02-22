import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import seaborn as sns
from matplotlib import cm
from matplotlib.ticker import (AutoLocator, AutoMinorLocator)


class Plotters():
    '''
    Utilities class to handle all plots.
    '''
    def __init__(self):

        self.s = 75
        self.marker = 'o'
        self.edgecolors = 'k'
        self.alpha = 1.0
        self.linewidths = 0.5
        self.fontname = "Times New Roman"
        self.ax_linewidth_thick = 1.5
        self.ax_linewidth_thin = 1.0
        self.fontsize = 13
        self.labelsize = 'x-large'
        self.pad = 10
        self.length_major = 9
        self.length_minor = 5
        self.width = 1.5

        cmaps = {
            'Dark2': cm.Dark2,
            'viridis': cm.viridis,
        }
        self.colormap = 'Dark2'

        self.cmap = cmaps[self.colormap]

        self.palette = 'classic'

    def set_palette(self, palette='classic'):
        self.palette = palette

        palette_options = {
            'classic': [
                'r', 'b', 'tab:green', 'deeppink', 'rebeccapurple',
                'darkslategray', 'teal', 'lightseagreen'
            ],
            'autumn': ['darkorange', 'darkred', 'mediumvioletred']
        }

        self.colors = palette_options[self.palette]
        return None

    def ax_formatter(self, axis, smaller=False, crossval=False, xax=True, legend = True):

        labelsize = self.labelsize
        pad = self.pad
        length_major = self.length_major
        length_minor = self.length_minor
        width = self.width

        if smaller == True:
            labelsize = 'medium'
            pad = pad * 0.2
            length_major = length_major * 0.75
            length_minor = length_minor * 0.6
            width = width * 0.6

        if crossval is False:
            if xax is True:
                axis.xaxis.set_major_locator(AutoLocator())
                axis.xaxis.set_minor_locator(AutoMinorLocator())

            axis.yaxis.set_major_locator(AutoLocator())
            axis.yaxis.set_minor_locator(AutoMinorLocator())

        if xax is False:
            width = 1.0
        axis.tick_params(direction='out',
                         pad=pad,
                         length=length_major,
                         width=width,
                         labelsize=labelsize)
        axis.tick_params(which='minor',
                         direction='out',
                         pad=pad,
                         length=length_minor,
                         width=width,
                         labelsize=labelsize)

        for ax in ['top', 'bottom', 'left', 'right']:
            axis.spines[ax].set_linewidth(self.ax_linewidth_thick)
            axis.spines[ax].set_linewidth(self.ax_linewidth_thin)

        if xax is True:
            for tick in axis.get_xticklabels():
                tick.set_fontname(self.fontname)
        for tick in axis.get_yticklabels():
            tick.set_fontname(self.fontname)

        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"
                                      ] + plt.rcParams["font.serif"]
        plt.rcParams['font.size'] = self.fontsize

        if legend is True:
            if xax is True:
                plt.legend(frameon=False)
        plt.tight_layout()

        return axis

    def plot_training_data(self, 
                           clustered_df, 
                           target = True, 
                            xlabel = r'$\rm PaCMAP\ 1$',
                            ylabel = r'$\rm PaCMAP\ 2$',
                           name = 'training_raw'):
        fig1 = plt.figure(figsize=(5, 5))

        clustered_df_denoised = clustered_df.loc[clustered_df.label != -1]

        axs1 = fig1.add_subplot(111) 
        print('Preparing plot...')

        if target is False:
            colors = clustered_df_denoised['label']
            cmap='Set3'
        else:
            colors = clustered_df_denoised['target']
            cmap='plasma'

        axs1.scatter(
            clustered_df_denoised['d0'],
            clustered_df_denoised['d1'],
            #this_cluster['d2'],
            s=10,
            cmap=cmap,
            c=colors,
            edgecolors='k',
            linewidths=0.1
            )

        noise_df = clustered_df.loc[clustered_df.label == -1]

        axs1.scatter(
                    noise_df['d0'],
                    noise_df['d1'],
                    s=7,
                    cmap=cmap,
                    c='k', 
                    edgecolors='k',
                    alpha=0.4,
                    linewidths=0.1,
        )


        axs1.set_xlabel(f"{xlabel}", labelpad=5, fontsize='x-large')
        axs1.set_ylabel(f"{ylabel}", labelpad=5, fontsize='x-large')

        axs1 = self.ax_formatter(axs1, xax=False)

        plt.savefig(
                    f'./reports/images/{name}_data_size_{clustered_df.shape[0]}.png',
                    transparent=True,
                    bbox_inches='tight')

        plt.show()
        plt.show()
        return None

    def plot_elbow(self, distances, threshold, name = 'elbow', save_fig: bool = False):

        fig1 = plt.figure(figsize=(10, 4))

        axs1 = fig1.add_subplot(111) 

        distances = np.sort(distances[distances != 0])

        axs1.plot(
                distances,
                color = 'purple')

        axs1.plot(
                [0, len(distances)],
                [threshold, threshold],
                color = 'r',
                linestyle = '--',
                label = 'knn distance threshold')


        axs1.set_xlabel(r'$\rm Sample$', labelpad=5, fontsize='x-large')
        axs1.set_ylabel(r'$\rm kNN\ distance$', labelpad=5, fontsize='x-large')

        axs1 = self.ax_formatter(axs1, xax=False)
        axs1.legend(frameon=False, ncol=2)

        if save_fig:
            plt.savefig(
                f'./reports/images/{name}.png',
                transparent=True,
                bbox_inches='tight')

        plt.show()
        return None
    
    def plot_pareto(self, res, threshold, name = 'opt_test', save_fig: bool = False):
        fig = plt.figure(figsize=(7, 5))
        axs1 = fig.add_subplot(1, 1, 1)

        axs1.scatter(res.F[:, 1], -res.F[:, 0], color = 'red', edgecolors = 'k')

        axs1.set_xlabel(r'$\rm kNN\ distance$',
                                labelpad=5,
                                fontsize='x-large')

        axs1.set_ylabel(r'$\rm Peak\ power\ density\ (W/cm^{2})$',
                        labelpad=5,
                        fontsize='x-large')

        # axs1.plot([threshold, threshold], [min(-res.F[:, 0]), max(-res.F[:, 0])], label = '2 x kNN threshold')
        # axs1.plot([threshold/2, threshold/2], [min(-res.F[:, 0]), max(-res.F[:, 0])], label = 'kNN threshold')

        
        axs1 = self.ax_formatter(axs1, xax=False)

        axs1.axvspan(min(res.F[:, 1]),
                            threshold,
                            facecolor='lightsteelblue',
                            alpha=0.3)

        axs1.axvspan(threshold,
                            2*threshold,
                            facecolor='cornflowerblue',
                            alpha=0.3)

        axs1.axvspan(2*threshold,
                            axs1.get_xlim()[1],
                            facecolor='royalblue',
                            alpha=0.3)

        axs1.text(min(res.F[:, 1]) + 0.05, max(-res.F[:, 0])*0.98, 'I')
        axs1.text(threshold + 0.05, max(-res.F[:, 0])*0.98, 'II')
        axs1.text(2*threshold + 0.05, max(-res.F[:, 0])*0.98, 'III')

        #plt.legend(frameon=False)
        axs1.set_ylim(min(-res.F[:, 0]), max(-res.F[:, 0])+0.05)
        axs1.set_xlim(min(res.F[:, 1]), max(res.F[:, 1]))

        if save_fig:
            plt.savefig(
                f'./reports/images/{name}.png',
                transparent=True,
                bbox_inches='tight')
        plt.show()

        return None
    
    
    def plot_importance_map(self, data, approach = 'pearson', name='feature'):
        """
            params: DataFrame of square size
            return: heat map
        """

        # create figure
        fig, axs = plt.subplots(figsize=(30, 15))
        # compute correlation
        importance = data.corr(method = approach)
        mask = np.triu(np.ones_like(importance, dtype=bool))
        # plot map
        map = sns.heatmap(importance, mask = mask,  annot = True, fmt = ".2f", center = 0, linewidths = .5, ax = axs, cmap = 'tab10', vmin = -1, vmax = +1)#, cbar_kws={"shrink": 1, "pad": 0.1, "orientation": 'horizontal'})

        axs = self.ax_formatter(axs, xax=False)

        #plt.tight_layout()
        plt.show()

        plt.savefig(
            f'./reports/images/{name}.png',
            transparent=True,
            bbox_inches='tight')


        #plt.show()
        return None

    def plot_pareto(self, 
                    res, 
                    threshold, 
                    xlabel = r'$\rm kNN\ distance$',
                    ylabel = r'$\rm Peak\ power\ density\ (W/cm^{2})$',
                    name = 'opt_test'):
        fig = plt.figure(figsize=(7, 5))
        axs1 = fig.add_subplot(1, 1, 1)

        axs1.scatter(res.F[:, 1], -res.F[:, 0], color = 'red', edgecolors = 'k')

        axs1.set_xlabel(xlabel,
                                labelpad=5,
                                fontsize='x-large')

        axs1.set_ylabel(ylabel,
                        labelpad=5,
                        fontsize='x-large')

        axs1 = self.ax_formatter(axs1, xax=False)

        axs1.axvspan(min(res.F[:, 1]),
                            threshold,
                            facecolor='lightsteelblue',
                            alpha=0.3)

        axs1.axvspan(threshold,
                            2*threshold,
                            facecolor='cornflowerblue',
                            alpha=0.3)

        axs1.axvspan(2*threshold,
                            axs1.get_xlim()[1],
                            facecolor='royalblue',
                            alpha=0.3)

        axs1.text(min(res.F[:, 1]) + 0.05, max(-res.F[:, 0])*0.98, 'I')
        axs1.text(threshold + 0.05, max(-res.F[:, 0])*0.98, 'II')
        axs1.text(2*threshold + 0.05, max(-res.F[:, 0])*0.98, 'III')

        axs1.set_ylim(min(-res.F[:, 0]), max(-res.F[:, 0])+0.05)
        axs1.set_xlim(min(res.F[:, 1]), max(res.F[:, 1]))


        plt.savefig(
            f'./reports/images/{name}.png',
            transparent=True,
            bbox_inches='tight')
        plt.show()

        return None


    def plot_shap_bar(self, 
                      shap_values, 
                      name: str = 'shap_bar'):
        
        shap.plots.bar(shap_values[:, :-1], show = False)

        fig1 = plt.gcf()

        plt.rcParams["figure.figsize"] = (7, 5)

        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"
                                        ] + plt.rcParams["font.serif"]
        plt.rcParams['font.size'] = 13

        plt.savefig(
            f'./reports/images/{name}.png',
            transparent=True,
            bbox_inches='tight')
        plt.show()

        return None

    def plot_shap_bee(self, 
                      shap_values, 
                      training_data, 
                      name: str = 'shap_bee',
                      save_fig: bool = False):

        shap.summary_plot(shap_values[:, :-1], training_data.iloc[:, :-1], show = False)

        fig1 = plt.gcf()

        plt.rcParams["figure.figsize"] = (7, 5)

        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"
                                        ] + plt.rcParams["font.serif"]
        plt.rcParams['font.size'] = 13
        if save_fig:
            plt.savefig(
                f'./reports/images/{name}.png',
                transparent=True,
                bbox_inches='tight')
        plt.show()

        return None
    
    def plot_model_eval(self,
                       predictions: dict = {'train': None, 'test': None, 'val': None},
                        labels: dict = {'train': None, 'test': None, 'val': None},
                        name = 'FigX', 
                        x_label = r'$\rm Voltage_{measured} (V)$', 
                        y_label = r'$\rm Voltage_{predicted} (V)$',
                        llimit = 0,
                        ulimit = 500,
                        shift = 5,
                        fig_size = (6, 5)):
        
        self.colors = ['tab:blue', 'tab:orange', 'tab:green']

        self.sets = [predictions['train'], labels['train'],
                predictions['test'], labels['test'],
                predictions['val'], labels['val']]

        fig = plt.figure(figsize=fig_size)

        axs1 = fig.add_subplot(1, 1, 1)

        axs1.plot([llimit, ulimit], [llimit, ulimit], color='r')

        axs1.scatter(self.sets[0],
                     self.sets[1],
                     s=self.s,
                     marker=self.marker,
                     facecolors=self.colors[0],
                     edgecolors='k',
                     alpha=self.alpha,
                     linewidths=self.linewidths,
                     label='Train')
        
        axs1.scatter(self.sets[4],
                     self.sets[5],
                     marker=self.marker,
                     s=self.s,
                     facecolors=self.colors[1],
                     edgecolors='k',
                     alpha=self.alpha,
                     linewidths=self.linewidths,
                     label='Val')
        
        axs1.scatter(self.sets[2],
                     self.sets[3],
                     s=self.s,
                     marker=self.marker,
                     facecolors=self.colors[2],
                     edgecolors='k',
                     alpha=self.alpha,
                     linewidths=self.linewidths,
                     label='Test')

        axs1.set_xlabel(x_label, 
                        labelpad=5,
                        fontsize='x-large')
        axs1.set_ylabel(y_label, 
                        labelpad=5,
                        fontsize='x-large')

        axs1.set_ylim(llimit, ulimit)
        axs1.set_xlim(llimit, ulimit)

        axs1.legend(frameon=False, loc=0)

        axs1 = self.ax_formatter(axs1)
        plt.show()
        plt.savefig(f'./reports/images/{name}.png',
                transparent=True,
                bbox_inches='tight')

        return None

    def plot_loss(self,
                    train_loss,
                    val_loss,
                    ylabel = r'$\rm Loss$',
                    name = 'loss',
                    fig_size = (5, 3),
                    save_fig: bool = False
                    ):
        fig = plt.figure(figsize=fig_size)
        ax2 = fig.add_subplot(1, 1, 1)
        colors = ['tab:blue', 'tab:orange', 'tab:green']
        ax2.plot(train_loss,
                 label=r'$ \rm Train$',
                 color=colors[0],
                 linestyle='-.',
                 linewidth=1)
        ax2.plot(val_loss,
                 label=r'$ \rm Val$',
                 color=colors[1],
                 linewidth=1)

        ax2.set_xlabel(r'$ \rm Epochs$', labelpad=1)
        ax2.set_ylabel(f"{ylabel}", labelpad=2)
        ax2.legend(frameon=False, loc=0)
        ax2 = self.ax_formatter(ax2)
        plt.show()

        if save_fig:
            plt.savefig(f'./reports/images/loss_{name}_profile.png',
                    transparent=True,
                    bbox_inches='tight')

    def plot_kde(self,
                 data: list,
                 labels: list,
                 colors: list,
                 property: str,
                 fig_size: tuple = (5, 3),
                 save_fig: bool = False) -> None:
        '''
        Plots the kernel density estimation of the data
        '''
    

        fig, axs = plt.subplots(1, dpi=200, figsize=fig_size, facecolor='white')
    
        for data_idx, label_idx, color_idx in zip(data, labels, colors):
            sns.kdeplot(data_idx, ax=axs, label=label_idx, fill=True, alpha=0.25, color=color_idx)

        plt.legend(loc='best', frameon=False, fancybox=False)
        plt.title(f'{property}')
        ymin, ymax = plt.ylim()
        xmin, xmax = plt.xlim()

        plt.show()

        if save_fig:
            plt.savefig(f'reports/images/{property}_kde.png', transparent=True, bbox_inches='tight')


    def plot_pred_distribution(self,
                 fig_size: tuple = (5, 3),
                 format: bool = False,
                 save_fig: bool = False) -> None:
        '''
        Plots the kernel density estimation of the data
        '''

        fig, (axs1, axs2, axs3) = plt.subplots(1, 3, dpi=200, figsize=fig_size, facecolor='white')
        
        sns.kdeplot(self.sets[0], ax=axs1, label='Pred', fill=True, alpha=0.25, color='C10')
        sns.kdeplot(self.sets[1], ax=axs1, label='Targ', fill=True, alpha=0.25, color='C1')
        axs1.set_title('Train')

        sns.kdeplot(self.sets[2], ax=axs2, label='Pred', fill=True, alpha=0.25, color='C10')
        sns.kdeplot(self.sets[3], ax=axs2, label='Targ', fill=True, alpha=0.25, color='C1')
        axs2.set_title('Test')
        

        sns.kdeplot(self.sets[4], ax=axs3, label='Pred', fill=True, alpha=0.25, color='C10')
        sns.kdeplot(self.sets[5], ax=axs3, label='Targ', fill=True, alpha=0.25, color='C1')
        axs3.set_title('Val')

        plt.tight_layout()
        plt.legend(loc='best', frameon=False, fancybox=False)
        plt.show()

        if save_fig:
            plt.savefig(f'reports/images/{property}_kde_predictions_distribution.png', transparent=True, bbox_inches='tight')

    def plot_cross_plot(self, x, y, 
                        x_label = r'$ \rm Original$', y_label = r'$ \rm Imputed$',
                        property = 'EC',
                        fig_size: tuple=(5,3), save_fig: bool=False):
        
        fig, axs = plt.subplots(1, dpi=200, figsize=fig_size, facecolor='white')
        
        colors = ['tab:blue', 'tab:orange', 'tab:green']
        
        axs.scatter(x, y,
                     s=self.s,
                     marker=self.marker,
                     facecolors=colors[0],
                     edgecolors='k',
                     alpha=self.alpha,
                     linewidths=self.linewidths,
                     label='Pred')
        
        axs.plot(x, x,
                 label=r'$ \rm R^{2} = 1$',
                 color='r',
                 linewidth=1)
        
        axs.set_xlabel(x_label + f" {property}", 
                        labelpad=5,
                        fontsize='large')
        
        axs.set_ylabel(y_label + f" {property}", 
                        labelpad=5,
                        fontsize='large')
        
        axs.legend(frameon=False, loc=0)
        
        plt.show()

        if save_fig:
            plt.savefig(f'reports/images/{property}_crossplot.png', transparent=True, bbox_inches='tight')

    def plot_heat_map(self, 
                    data: pd.DataFrame = None, label = r'$ \rm Actual$',
                    property: str = 'original',
                    fig_size = (10, 5), save_fig: bool = False):
        
        fig, ax = plt.subplots(1, figsize=fig_size, facecolor='white')

        # Create the heatmap for the original data
        sns.heatmap(data.corr(), cmap='coolwarm', annot=True, fmt='.2f', linewidths=.5, ax=ax, cbar=False)
        ax.set_title(f'{label}', fontsize='x-large')
        ax.set_xlabel(r'$ \rm Values$', labelpad=5, fontsize='large')

        # Show the plot
        plt.show()
        if save_fig:
                plt.savefig(f'reports/images/{property}_data_single_distibution_comparison.png', transparent=True, bbox_inches='tight')
                


    def plot_heat_2map(self, 
                    original_data: pd.DataFrame = None, original_label = r'$ \rm Actual$',
                    synthetic_data: pd.DataFrame = None, synthetic_label = r'$ \rm Synthetic$', 
                    property: str = 'original_vs_synthetic',
                    fig_size = (10, 5), save_fig: bool = False):
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size, facecolor='white')

        # Create the heatmap for the original data
        sns.heatmap(original_data.corr(), cmap='coolwarm', annot=True, fmt='.2f', linewidths=.5, ax=ax1, cbar=False)
        ax1.set_title(f'{original_label}', fontsize='x-large')
        ax1.set_xlabel(r'$ \rm Values$', labelpad=5, fontsize='large')

        # create the heatmap for the synthetic data
        sns.heatmap(synthetic_data.corr(), cmap='coolwarm', annot=True, fmt='.2f', linewidths=.5, ax=ax2, cbar=False)
        ax2.set_title(f'{synthetic_label}',fontsize='x-large')
        ax2.set_xlabel(r'$ \rm Values$', labelpad=5, fontsize='large')

        # Show the plot
        plt.show()

        if save_fig:
                plt.savefig(f'reports/images/{property}_data_distibution_comparison.png', transparent=True, bbox_inches='tight')
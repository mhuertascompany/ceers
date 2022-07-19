import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import os
from integrated_gradients import integrated_gradients


class GraphPlotter(object):
    """
    Class for organizing all plots produced by
    the training and evaluation of the CNN
    """

    def __init__(self, save_dir, model_id):
        self.save_dir = save_dir
        self.model_id = model_id

    def save_plot(self, filename, directory=None, kwargs={}):
        if directory:
            os.makedirs(os.path.join(self.save_dir, directory), exist_ok=True)
            filename = '{}/{}'.format(directory, filename)
        plt.savefig(os.path.join(self.save_dir, filename + '.png'), **kwargs)
        plt.close()

    def plot_original_maps(self, images, labels, magnitude=None, test=False):
        num_examples = 9
        labels = 10 ** labels
        fig = plt.figure(figsize=(10, 10))
        
        for i in range(num_examples):
            plt.subplot(3, 3, i + 1)
            im = plt.imshow(images[i, :, :], cmap='jet')
            plt.gca().set_title('Re: %.3f' % labels[i] +
                                (', M:  %.3f' % magnitude[i] if magnitude is not None else ''),
                                rotation=0)
            plt.axis('off')

        fig.set_facecolor('w')
        plt.tight_layout()
        self.save_plot('{}Original maps'.format('Test ' if test else ''),
                       kwargs={'dpi': 200})

    def plot_histogram(self, images, labels, test=False):
        plt.hist(labels, bins=100, density=True)
        self.save_plot('Training histogram')
    
    def plot_correlation(self, re, magnitude):
        plt.scatter(re, magnitude)
        plt.xlabel('Log(Effective Radius)')
        plt.ylabel('Magnitude')
        self.save_plot('Correlation_Log')

        plt.scatter(10 ** re, magnitude)
        plt.xlabel('Effective Radius')
        plt.ylabel('Magnitude')
        self.save_plot('Correlation')

    def plot_training_graphs(self, history):
        self.plot_mse_loss_history(history, mode='loss', label='Loss')
        self.plot_mse_loss_history(history, mode='mse', label='MSE')

    def plot_evaluation_results(self, y_true, y_pred, magnitude=None,
                                y_pred_distr=None, mdn=True, logged=True):
        if not logged:
            y_true = 10 ** y_true
            y_pred = 10 ** y_pred 
        
        self.plot_prediction_vs_true(y_true, y_pred, magnitude=magnitude, logged=logged)
        self.plot_residual(y_true, y_pred, logged=logged)

        if mdn:
            y_pred = y_pred_distr.mean().numpy().reshape(-1)
            if not logged:
                y_pred = 10 ** y_pred
            y_pred_std = y_pred_distr.stddev().numpy().reshape(-1)
            self.plot_prediction_vs_true_with_error_bars(y_true, y_pred, y_pred_std,
                                                         magnitude=magnitude, logged=logged)
            self.plot_prediction_vs_true_with_error_bars_smooth(y_true, y_pred, y_pred_std,
                                                                magnitude=magnitude, logged=logged)

    def plot_mse_loss_history(self, history, mode='loss', label='Loss'):
        plt.figure()

        epochs = len(history.history[mode][3:])
        plt.plot(range(epochs), history.history[mode][3:], label=label)
        val_mode = 'val_{}'.format(mode)
        if val_mode in history.history:
            plt.plot(range(epochs), history.history[val_mode][3:],
                     label='Validation {}'.format(label))
        plt.xlabel('Epoch')
        plt.ylabel(mode.capitalize())
        plt.legend(loc='upper right')

        self.save_plot('{}_history'.format(label))

    @staticmethod
    def scatter_predictions_vs_true(y_true, y_pred, magnitude=None):
        color = magnitude
        if color is None:
            color = 'blue'
        scatter_kwargs = {"zorder":100}
        plt.scatter(y_true, y_pred, c=color, **scatter_kwargs)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        _ = plt.plot([y_true.min() - 0.1, y_true.max() + 0.1], 
                     [y_true.min() - 0.1, y_true.max() + 0.1])
        if magnitude is not None:
            plt.colorbar()

    def plot_prediction_vs_true(self, y_true, y_pred, magnitude=None, logged=True):
        plt.figure()
        plt.axes(aspect='equal')
        self.scatter_predictions_vs_true(y_true, y_pred, magnitude=magnitude)
        plt.legend(loc='upper left')
        self.save_plot('Predictions_vs_True{}'.format('_log' if logged else ''))

    def plot_with_median(self, X, Y, color1, color2, log=True, percentiles=True, show=False):
        """
        Plot the running media of the X, Y data with 25th and 75th percentiles,
        if requested
        """
        total_bins = 10
        if log:
            bins = np.geomspace(X.min(), X.max(), total_bins)
        else:
            bins = np.linspace(X.min(), X.max(), total_bins)
    
        delta = bins[1] - bins[0]
        idx = np.digitize(X, bins)
        running_median = [np.median(Y[idx == k]) for k in range(total_bins)]
        running_prc25 = [np.nanpercentile(Y[idx == k], 25) for k in range(total_bins)]
        running_prc75 = [np.nanpercentile(Y[idx == k], 75) for k in range(total_bins)]
        plt.plot(bins - delta / 2, running_median, color2, linestyle='--', lw=4, alpha=.8)
    
        if percentiles:
            plt.fill_between(bins - delta/2, running_prc25, running_median, facecolor=color1, alpha=0.2)
            plt.fill_between(bins - delta/2, running_prc75, running_median, facecolor=color1, alpha=0.2)
        else:
            plt.scatter(X, Y, color=color1, alpha=.2, s=2, zorder=1 if show else 0)
    
        if log:
            plt.xscale('symlog')
    
    def plot_residual(self, y_true, y_pred, logged=True):
        plt.figure()
        self.plot_with_median(y_true, np.abs(y_pred-y_true)/y_true,
                              'blue', 'darkblue', log=False)
        plt.xlabel('True Values')
        plt.ylabel('|Predictions - True|/True')
        self.save_plot('Relative_error{}'.format('_log' if logged else ''))

    def plot_prediction_vs_true_with_error_bars(self, y_true, y_pred, err,
                                                magnitude=None, logged=True):
        plt.figure()
        plt.axes(aspect='equal')
        self.scatter_predictions_vs_true(y_true, y_pred, magnitude=magnitude)
        error_kwargs = {"lw":.5, "zorder":0}
        plt.errorbar(y_true, y_pred, yerr=err, linestyle="None", fmt='o',
                     color='blue', label=r'$\sigma$', 
                     lw=0.5, zorder=0)

        #plt.errorbar(y_true, y_pred, yerr=err, linestyle="None",
        #             fmt=None, marker=None, mew=0, **error_kwargs)

        plt.legend(loc='upper left')
        self.save_plot('Predictions_vs_True_Error_Bars{}'.format('_log' if logged else ''))

    def plot_prediction_vs_true_with_error_bars_smooth(self, y_true, y_pred, err,
                                                       magnitude=None, logged=True):
        sorted_idxs = np.argsort(y_true)
        y_true = y_true[sorted_idxs]
        y_pred = y_pred[sorted_idxs]
        err = np.array(err)[sorted_idxs]

        plt.figure()
        plt.axes(aspect='equal')
        self.scatter_predictions_vs_true(y_true, y_pred, magnitude=magnitude)

        plt.fill_between(y_true, y_pred + err, y_pred - err,
                         alpha=0.2, color='b', label=r'$\sigma$')
        plt.fill_between(y_true, y_pred + 2 * err, y_pred - 2 * err,
                         alpha=0.2, color='b', label=r'$2\sigma$')

        plt.legend(loc='upper left')
        self.save_plot('Predictions_vs_True_Error_Bars_Smooth{}'.format('_log' if logged else ''))



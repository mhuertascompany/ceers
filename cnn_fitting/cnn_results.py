import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import os


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
        plt.savefig(os.path.join(self.save_dir, filename + '_{}.png'.format(self.model_id)), **kwargs)
        plt.close()

    def plot_original_maps(self, images, labels, test=False):
        num_examples = 9
        labels = 10 ** labels
        fig = plt.figure(figsize=(10, 10))
        for i in range(num_examples):
            plt.subplot(3, 3, i + 1)
            im = plt.imshow(images[i, :, :], cmap='jet')
            plt.gca().set_title('Re: %.3f' % labels[i], rotation=0)
            plt.axis('off')

        fig.set_facecolor('w')
        plt.tight_layout()
        self.save_plot('{}Original maps.png'.format('Test ' if test else ''),
                       kwargs={'dpi': 200})

    def plot_training_graphs(self, history):
        self.plot_mse_loss_history(history, mode='loss', label='Loss')
        self.plot_mse_loss_history(history, mode='mse', label='MSE')

    def plot_evaluation_results(self, y_true, y_pred, y_pred_distr=None, mdn=True):
        y_true = 10 ** y_true
        y_pred = 10 ** y_pred
        self.plot_prediction_vs_true(y_true[:128], y_pred[:128])

        if mdn:
            y_pred = y_pred_distr.mean().numpy().reshape(-1)
            y_pred_std = y_pred_distr.stddev().numpy().reshape(-1)
            self.plot_prediction_vs_true_with_error_bars(y_true[:128], y_pred[:128], y_pred_std[:128])
            self.plot_prediction_vs_true_with_error_bars_smooth(y_true[:128], y_pred[:128], y_pred_std[:128])

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
    def scatter_predictions_vs_true(y_true, y_pred):
        plt.scatter(y_true, y_pred, color='b')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        _ = plt.plot([0, 1.5], [0, 1.5])

    def plot_prediction_vs_true(self, y_true, y_pred):
        plt.figure()
        plt.axes(aspect='equal')
        self.scatter_predictions_vs_true(y_true, y_pred)
        plt.legend(loc='upper left')
        self.save_plot('Predictions_vs_True')

    def plot_prediction_vs_true_with_error_bars(self, y_true, y_pred, err):
        plt.figure()
        plt.axes(aspect='equal')
        self.scatter_predictions_vs_true(y_true, y_pred)
        plt.errorbar(y_true, y_pred, yerr=err, linestyle="None", fmt='o',
                     capsize=3, color='blue', capthick=0.5, label=r'$\sigma$')

        plt.legend(loc='upper left')
        self.save_plot('Predictions_vs_True_Error_Bars')

    def plot_prediction_vs_true_with_error_bars_smooth(self, y_true, y_pred, err):
        sorted_idxs = np.argsort(y_true)
        y_true = y_true[sorted_idxs]
        y_pred = y_pred[sorted_idxs]
        err = np.array(err)[sorted_idxs]

        plt.figure()
        plt.axes(aspect='equal')
        self.scatter_predictions_vs_true(y_true, y_pred)

        plt.fill_between(y_true, y_pred + err, y_pred - err,
                         alpha=0.2, color='b', label=r'$\sigma$')
        plt.fill_between(y_true, y_pred + 2 * err, y_pred - 2 * err,
                         alpha=0.2, color='b', label=r'$2\sigma$')

        plt.legend(loc='upper left')
        self.save_plot('Predictions_vs_True_Error_Bars_Smooth')






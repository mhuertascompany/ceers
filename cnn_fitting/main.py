import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.client import device_lib
from contextlib import redirect_stdout

from constants import MDN, INPUT_SHAPE, RESULTS_PATH, BATCHES
from cnn_data_pipeline import input_fn, get_num_examples, get_data, get_data_test
from cnn_model import cnn_model_simple, load_saved_model, CNNModelTemplate
from cnn_results import GraphPlotter


logging.basicConfig(format='%(asctime)s %(message)s',
                    level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger("input_logger")


class CNNModel(object):

    def __init__(self, model_id, model_fn):
        """ Initialize variables required for training and evaluation of model"""

        self.model_id = model_id
        self.model_fn = model_fn

        self.model = None
        self.ds_train = None
        self.ds_val = None
        self.ds_test = None
        self.len_ds_train = self.len_ds_val = self.len_ds_test = 0
        self.n_epochs = 0

        # Create directory for this run
        self.run_dir = os.path.join(RESULTS_PATH, 'structural_fitting_log')
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)

        self.plotter = GraphPlotter(self.run_dir, self.model_id)

        # Save model in dedicated save_models folder
        # (so that it is not copied along with the rest of the results)
        self.model_file_path = RESULTS_PATH + '/saved_models/model.h5'

    def load_datasets(self, dataset_str='structural_fitting'):
        """
        Load the train, validation and test sets and plot some informative graphs
        :return:
        """

        self.ds_train = input_fn('train', dataset_str)
        self.ds_val = input_fn('validation', dataset_str)
        self.ds_test = input_fn('test', dataset_str)

        self.len_ds_train = get_num_examples('train', dataset_str)
        self.len_ds_val = get_num_examples('validation', dataset_str)
        self.len_ds_test = get_num_examples('test', dataset_str)

        # Plot some informative graphs for the input data of the CNN
        #input_plots(self.ds_train, self.run_dir)

    def train_model(self):
        """
        Train the CNN model using the train set and check for
        overfitting with the validation set
        :return:
        """

        log.info('*************** TRAINING *******************')

        tf.compat.v1.reset_default_graph()
        self.model = self.model_fn(INPUT_SHAPE, mdn=MDN).compile()

        with open(self.run_dir + '/modelsummary_{}.txt'.format(self.model_id), 'w') as f:
            with redirect_stdout(f):
                self.model.summary()

        images, y_true = get_data(self.ds_train, batches=1000)
        for i, label in enumerate(['Radius', 'Sersic Idx', 'Ellipticity']):
            self.plotter.plot_histogram(images, y_true[:, i], label=label)
        self.plotter.plot_original_maps(images, y_true)
        
        # Train model
        es = EarlyStopping(monitor='val_loss', patience=50)
        mc = ModelCheckpoint(filepath=self.model_file_path, monitor='val_loss', save_best_only=True)
        history = self.model.fit(self.ds_train,
                                 epochs=2,
                                 steps_per_epoch=self.len_ds_train // BATCHES,
                                 validation_steps=self.len_ds_val // BATCHES,
                                 validation_data=self.ds_val,
                                 callbacks=[es],
                                 use_multiprocessing=True, workers=4)

        self.n_epochs = len(history.history['loss'])
        #for i, label in enumerate(['Radius', 'Sersic Idx', 'Ellipticity']):
        #    self.plotter.plot_training_graphs(history, label)

        self.model.save_weights(self.model_file_path)
        #self.model = load_saved_model(self.model_file_path, mdn=True)
        log.info('Evaluate with training set on best model containing {} examples'.format(self.len_ds_train))
        train_loss, r_loss, s_loss, \
            e_loss, r_mse, s_mse, e_mse = self.model.evaluate(self.ds_train,
                                                              steps=100, verbose=2)
        log.info('Best train Loss: {}, Best train MSE: {}, {}, {}'.format(train_loss, r_mse, s_mse, e_mse))

    def load_saved_model(self):
        model = self.model_fn(INPUT_SHAPE, mdn=MDN).load_saved_model(self.model_file_path)
        return model

    def evaluate_model(self):
        """
        Evaluate the performance of the CNN using the test set
        :return:
        """
        self.model = self.load_saved_model()

        log.info('*************** EVALUATING *******************')
        log.info('Evaluate with test set containing {} examples'.format(self.len_ds_test))
        test_loss, r_loss, s_loss, \
            e_loss, r_mse, s_mse, e_mse = self.model.evaluate(self.ds_test, verbose=2)
        log.info('Test Loss: {}, Test MSE: {}, {}, {}'.format(test_loss, r_mse, s_mse, e_mse))

        with open(self.run_dir + "/Results.txt", "w") as result_file:
            result_file.write("Trained for epochs: %s\n\n"
                              "Test loss, MSE: %s %s %s %s" % (self.n_epochs, test_loss, r_mse, s_mse, e_mse))

        #images, y_true, magnitude = get_data_test(self.ds_train, batches=700)
        #self.plotter.plot_correlation(y_true, magnitude)
        
        images, y_true, magnitude = get_data_test(self.ds_test, batches=500)
        for i, label in enumerate(['Radius', 'Sersic Idx', 'Ellipticity']):
            self.plotter.plot_histogram(images, y_true[:, i], label=label, ds='Test')
        self.plotter.plot_original_maps(images, y_true, magnitude=magnitude, prefix='Test ')

        y_pred_distr = self.model(images)

        for i, label in enumerate(['Radius', 'Sersic Idx', 'Ellipticity']):
            y_pred = y_pred_distr[i].mean().numpy().reshape(-1)

            if i == 0:
                self.plotter.plot_evaluation_results(y_true[:, i], y_pred, magnitude=magnitude,
                                                     y_pred_distr=y_pred_distr[i], mdn=MDN, label=label)
            self.plotter.plot_evaluation_results(y_true[:, i], y_pred, magnitude=magnitude,
                                                 y_pred_distr=y_pred_distr[i], mdn=MDN,
                                                 logged=False, label=label)

    def cross_evaluate_model(self):
        """
        Cross-evaluate the performance of the CNN using the test set
        :return:
        """
        log.info('*************** CROSS EVALUATING *******************')

        ds_test_other = input_fn('test', 'ceers_mocks')
        len_ds_test = get_num_examples('test', dataset_str='ceers_mocks')
        log.info('Evaluate with test set containing {} examples'.format(len_ds_test))

        test_loss, test_mse = self.model.evaluate(ds_test_other, verbose=2)
        log.info('Cross test Loss: {}, Cross test MSE: {}'.format(test_loss, test_mse))

        cross_run_dir = os.path.join(self.run_dir, 'Cross')
        if not os.path.exists(cross_run_dir):
            os.makedirs(cross_run_dir)

        with open(cross_run_dir + "/Results_cross.txt", "w") as result_file:
            result_file.write("Trained for epochs: %s\n\n"
                              "Cross test loss, Cross MSE: %s %s" % (self.n_epochs, test_loss, test_mse))

        # images, y_true, magnitude = get_data_test(self.ds_train, batches=700)
        # self.plotter.plot_correlation(y_true, magnitude)

        images, y_true, magnitude = get_data_test(ds_test_other, batches=50)
        self.plotter.plot_histogram(images, y_true, ds='Cross Test')
        self.plotter.plot_original_maps(images, y_true, magnitude=magnitude, prefix='Cross Test ')

        y_pred = self.model.predict(images).flatten()
        y_pred = np.array(y_pred)

        y_pred_distr = None
        if MDN:
            y_pred_distr = self.model(images)
            y_pred = y_pred_distr.mean().numpy().reshape(-1)

        cross_plotter = GraphPlotter(cross_run_dir, self.model_id)
        cross_plotter.plot_evaluation_results(y_true, y_pred, magnitude=magnitude,
                                             y_pred_distr=y_pred_distr, mdn=MDN)
        cross_plotter.plot_evaluation_results(y_true, y_pred, magnitude=magnitude,
                                             y_pred_distr=y_pred_distr, mdn=MDN,
                                             logged=False)

    def run(self):
        """
        Load the datasets, train and evaluate results of model
        :return:
        """
        dataset_main = 'structural_fitting'
        self.load_datasets(dataset_main)
        self.train_model()
        self.evaluate_model()
        self.cross_evaluate_model()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    log.info(device_lib.list_local_devices())

    with tf.device('/gpu:0'):
        cnn_model = CNNModel(0, CNNModelTemplate)
        cnn_model.run()


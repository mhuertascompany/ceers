"""cnn_fitting dataset."""

import tensorflow_datasets as tfds
from . import structural_fitting


class StructuralFittingTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for cnn_fitting dataset."""
  # TODO(cnn_fitting):
  DATASET_CLASS = structural_fitting.StructuralFitting
  SPLITS = {
      'train': 3,  # Number of fake train example
      'test': 1,  # Number of fake test example
  }

  # If you are calling `download/download_and_extract` with a dict, like:
  #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
  # then the tests needs to provide the fake output paths relative to the
  # fake data directory
  # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}


if __name__ == '__main__':
  tfds.testing.test_main()

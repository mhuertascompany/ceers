""" ceers_mocks dataset. """

import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import os
import sys

from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.nddata import Cutout2D

import custom_split as split

sys.path.append(os.path.abspath('../..'))

from constants import (SPLITS, INPUT_SHAPE, CEERS_MOCK_CATALOG_PATH, CEERS_MOCK_PATH,
                       CEERS_MOCK_MORPH_PATH, CEERS_MOCK_MORPH_IDX_PATH)
from utils import load_catalog_values, load_morphological_values, load_data


_DESCRIPTION = """
    CEERS mock images of galaxies along with their effective radius (in arcsec)
"""

# TODO(cnn_fitting): BibTeX citation
_CITATION = """
"""


class CEERSMocks(tfds.core.GeneratorBasedBuilder):
    """ DatasetBuilder for ceers_mocks dataset. """

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """ Returns the dataset metadata. """

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Tensor(shape=INPUT_SHAPE[:-1], dtype=tf.float32),
                'angular_size': tf.float32,
                'object_id': tf.string,
                'magnitude': tf.float32
            }),
            supervised_keys=('image', 'angular_size'),
            homepage='https://dataset-homepage/',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        split_ids = split.create_custom_split(SPLITS)
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs=dict(
                    split_ids=split_ids['train'],
                )),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs=dict(
                    split_ids=split_ids['validation'],
                )),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs=dict(
                    split_ids=split_ids['test'],
                )),
        ]

    def _generate_examples(self, split_ids):
        """ Yields examples. """

        df = load_catalog_values(CEERS_MOCK_CATALOG_PATH)
        morph_df = load_morphological_values(CEERS_MOCK_MORPH_PATH, CEERS_MOCK_MORPH_IDX_PATH)

        print(df)
        print(morph_df)

        band = 200
        id_200, im_200, z_200, nsel_200 = load_data(0, CEERS_MOCK_PATH, band)
        im_200 = np.expand_dims(im_200, axis=3)

        id_200 = list(id_200)
        i = 0

        for gid in split_ids:
            print ('-----------GALAXY ID', gid)
            g_row = df[df.ID == str(gid)].iloc[0]
            g_morph_row = morph_df[morph_df.ID == gid].iloc[0]

            angular_size = g_morph_row.sersic_rhalf / (g_row.z + 1)
            print(angular_size)
            magnitude = g_row.mf200w

            if angular_size > 2:
                continue

            indexes = [i for i, v in enumerate(id_200) if v.split('_')[0] == str(gid)]
            for index_ in indexes:
                try:
                    img = im_200[index_, :, :, 0]
                    galaxy_rot_id = id_200[index_]
                    if not np.isfinite(angular_size):
                        continue

                    i += 1
                    # Yield with i because in our case object_id will be the same for the 4 different projections
                    yield i, {'image': img.astype("float32"),
                              'angular_size': angular_size.astype("float32"),
                              'object_id': galaxy_rot_id,
                              'magnitude': magnitude}

                except Exception as e:
                    print("Galaxy id not added to the dataset: object_id=", galaxy_rot_id,
                          "angular size ", angular_size, e)


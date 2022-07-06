""" cnn_fitting dataset. """

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

from constants import DATA_PATH, SPLITS


_DESCRIPTION = """
    CEERS images of galaxies along with their effective radius (in arcsec)
    
    Data is extracted from:
     - fits file that we will cut in stamps of 64x64 pixels centered on each galaxy
     (/home/eirini/Documents/PhD/Repos/ceers/data/eirini_data/ceers5_f150w_i2d.fits.gz) 
     - Catalog containing info on their position (ra, dec) and their effective radius 
      (/home/eirini/Documents/PhD/Repos/ceers/data/eirini_data/CEERS_SDR3_SAM_input.fits)
    
    Preprocessing:
     - Remove all galaxies with magnitude > 27.5
     - Remove all images with more than 75% zero pixels
     - Log Effective Radius
"""

# TODO(cnn_fitting): BibTeX citation
_CITATION = """
"""


class StructuralFitting(tfds.core.GeneratorBasedBuilder):
    """ DatasetBuilder for cnn_fitting dataset. """

    VERSION = tfds.core.Version('2.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
      '2.0.0': 'Include both filters'
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """ Returns the dataset metadata. """

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Tensor(shape=(64, 64), dtype=tf.float32),
                'angular_size': tf.float32,
                'object_id': tf.string
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
        """Yields examples."""

        filter_paths = [(f, os.path.join(DATA_PATH,
                                         'ceers5_f{}w_i2d.fits.gz'.format(f)))
                        for f in (150, 200)]

        cat_path = os.path.join(DATA_PATH, 'CEERS_SDR3_SAM_input.fits')

        table = Table.read(cat_path)
        cat = table.to_pandas()

        i = 0
        for filter_v, filter_path in filter_paths:
            ceers = fits.open(filter_path)
            image = ceers[1].data
            w = WCS(ceers[1].header)

            cut = cat.query("NIRCam_F{}W<27".format(filter_v))

            for gid in split_ids:
                c = SkyCoord(cat.ra[gid], cat.dec[gid], unit="deg")
                angular_size = cat.angular_size[gid]
                try:
                    img = Cutout2D(image, c, (64, 64), wcs=w).data
                    if np.where(img == 0.0)[0].size > 0.75 * img.size:
                        continue

                    if img.size != 64*64:
                        continue

                    if np.all(img == img[0, 0]):
                        continue

                    if not np.isfinite(angular_size):
                        continue

                    i += 1
                    # Yield with i because in our case object_id will be the same for the 4 different projections
                    yield i, {'image': img.astype("float32"),
                              'angular_size': angular_size.astype("float32"),
                              'object_id': '{}_{}'.format(gid, filter_v)}

                except Exception as e:
                    print("Galaxy id not added to the dataset: object_id=", gid,
                          "angular size ", angular_size, e)

            ceers.close()

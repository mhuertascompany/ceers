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

sys.path.append(os.path.abspath('../..'))

from constants import DATA_PATH


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
                'image': tfds.features.Tensor(shape=(64, 64), dtype=tf.float32),
                'angular_size': tf.float32,
            }),
            supervised_keys=('image', 'angular_size'),
            homepage='https://dataset-homepage/',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = os.path.join(DATA_PATH, 'ceers5_f150w_i2d.fits.gz')
        print ('----------', path, DATA_PATH)
        cat_path = os.path.join(DATA_PATH, 'CEERS_SDR3_SAM_input.fits')
        return {
            'train': self._generate_examples(path, cat_path),
        }

    def _generate_examples(self, path, cat_path):
        """Yields examples."""

        ceers = fits.open(path)
        image = ceers[1].data
        w = WCS(ceers[1].header)

        table = Table.read(cat_path)
        cat = table.to_pandas()
        cut = cat.query("NIRCam_F150W<27")

        for gid in reversed(cut.index):
            c = SkyCoord(cat.ra[gid], cat.dec[gid], unit="deg")
            angular_size = cat.angular_size[gid]
            try:
                img = Cutout2D(image, c, (64, 64), wcs=w).data
                if np.where(img == 0.0)[0].size > 0.75 * img.size:
                    continue

                if img.size != 64*64:
                    continue

                print('-------------YIELD---------------------')
                # Yield with i because in our case object_id will be the same for the 4 different projections
                yield int(gid), {'image': img.astype("float32"),
                            'angular_size': angular_size.astype("float32")}

            except Exception as e:
                print("Galaxy id not added to the dataset: object_id=", gid,
                      "angular size ", angular_size, e)

        ceers.close()

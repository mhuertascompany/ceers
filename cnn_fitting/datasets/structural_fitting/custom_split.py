import copy
import random
import pickle
import sys
import os

from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.nddata import Cutout2D

sys.path.append(os.path.abspath('../..'))

from constants import GALAXY_IDS_PATH, SPLITS, DATA_PATH, MAGNITUDE_CUT, INPUT_SHAPE


def split_galaxy_ids(ids_file, splits):
    """
    Split the galaxy ids that exist in the file provided
    to the requested splits according to the percentages.

    :param ids_file: The file containing the ids that you wish to split
    :param splits: Ordered dictionary holding as keys the required splits and
    as values the percentage of every split
        e.g {'train': train_percentage}

    :return: Dictionary holding the galaxy ids of each split
        e.g {'train': train_galaxy_ids}
    """

    with open(ids_file, 'rb') as f:
        galaxy_ids = set(pickle.load(f))

    galaxy_splits = {}
    remaining_ids = copy.deepcopy(galaxy_ids)
    for split, percentage in splits.items():
        split_ids = set(random.sample(remaining_ids, int(len(galaxy_ids) * percentage)))
        galaxy_splits[split] = split_ids
        remaining_ids -= split_ids

    # Include the remaining ids (if any) to last split
    if remaining_ids:
        galaxy_splits[split].update(remaining_ids)

    return galaxy_splits


def resolve_galaxy_ids():
    """ Resolve galaxy ids that exist in the fits files
        Use only the 200W filter for that
        (assume the same galaxies exist in the 150W)
    """

    filter_200 = os.path.join(DATA_PATH, 'ceers5_f200w_i2d.fits.gz')
    cat_path = os.path.join(DATA_PATH, 'CEERS_SDR3_SAM_input.fits')

    table = Table.read(cat_path)
    cat = table.to_pandas()

    galaxy_ids = []
    ceers = fits.open(filter_200)
    image = ceers[1].data
    w = WCS(ceers[1].header)

    cut = cat.query("NIRCam_F200W<{}".format(MAGNITUDE_CUT))

    for gid in reversed(cut.index):
        c = SkyCoord(cat.ra[gid], cat.dec[gid], unit="deg")
        try:
            Cutout2D(image, c, INPUT_SHAPE[:-1], wcs=w).data
        except:
            continue
        galaxy_ids.append(gid)

    with open(GALAXY_IDS_PATH, 'wb') as fp:
        pickle.dump(galaxy_ids, fp)


def create_custom_split(splits):
    galaxy_ids_file = GALAXY_IDS_PATH
    # if the galaxy ids have not already been resolved,
    # get them and save them to file
    if not os.path.exists(galaxy_ids_file):
        resolve_galaxy_ids()
    galaxy_splits = split_galaxy_ids(galaxy_ids_file, splits)
    return galaxy_splits


if __name__ == '__main__':
    splits = SPLITS
    create_custom_split(splits)




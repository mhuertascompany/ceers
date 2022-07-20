import copy
import random
import pickle
import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath('../..'))

from constants import SPLITS, CEERS_MOCK_CATALOG_PATH
from utils import load_catalog_values


def split_galaxy_ids(cat_file, splits):
    """
    Split the galaxy ids that exist in the file provided
    to the requested splits according to the percentages.

    :param cat_file: The catalog file containing the ids that you wish to split
    :param splits: Ordered dictionary holding as keys the required splits and
    as values the percentage of every split
        e.g {'train': train_percentage}

    :return: Dictionary holding the galaxy ids of each split
        e.g {'train': train_galaxy_ids}
    """

    df = load_catalog_values(cat_file)
    galaxy_ids = set([int(i) for i in df['ID'].unique()])

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


def create_custom_split(splits):
    galaxy_ids_file = CEERS_MOCK_CATALOG_PATH
    galaxy_splits = split_galaxy_ids(galaxy_ids_file, splits)
    print(galaxy_splits)
    return galaxy_splits


if __name__ == '__main__':
    splits = SPLITS
    create_custom_split(splits)




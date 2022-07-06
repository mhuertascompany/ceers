import os
from collections import OrderedDict

# All parameters needed for running the CNN
MDN = True
INPUT_SHAPE = (64, 64, 1)
BATCHES = 32
DEBUG = True

if DEBUG:
    BASE_PATH = '/home/eirini/Documents/PhD/Repos/ceers/'
else:
    BASE_PATH = '/scratch/eirinia/projects/ceers'

DATA_PATH = os.path.join(BASE_PATH, 'data/eirini_data')
RESULTS_PATH = os.path.join(BASE_PATH, 'cnn_fitting/Results')
GALAXY_IDS_PATH = os.path.join(DATA_PATH, 'galaxy_ids.pkl')

SPLITS = OrderedDict({
    'train': 0.75,
    'validation': 0.1,
    'test': 0.15
})

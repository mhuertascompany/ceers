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

MAGNITUDE_CUT = 27.5
DATA_PATH = os.path.join(BASE_PATH, 'data/eirini_data')
RESULTS_PATH = os.path.join(BASE_PATH, 'cnn_fitting/Results')
GALAXY_IDS_PATH = os.path.join(DATA_PATH, 'galaxy_ids.pkl')

CEERS_MOCK_PATH = os.path.join(DATA_PATH, 'ceers_mocks')
CEERS_MOCK_CATALOG_PATH = os.path.join(CEERS_MOCK_PATH, 'im_all_catalog_64pix.dat')
CEERS_MOCK_MORPH_PATH = os.path.join(CEERS_MOCK_PATH, 'morphological_catalog_F200W_v1.0.hdf5')
CEERS_MOCK_MORPH_IDX_PATH = os.path.join(CEERS_MOCK_PATH, 'indexes_match_morph_catalogs.dat')

SPLITS = OrderedDict({
    'train': 0.75,
    'validation': 0.1,
    'test': 0.15
})

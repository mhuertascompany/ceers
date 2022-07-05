import os

# All parameters needed for running the CNN
MDN = True
INPUT_SHAPE = (64, 64, 1)
BATCHES = 32
DEBUG = False

if DEBUG:
    BASE_PATH = '/home/eirini/Documents/PhD/Repos/ceers/'
else:
    BASE_PATH = '/scratch/eirinia/projects/ceers'

DATA_PATH = os.path.join(BASE_PATH, 'data/eirini_data')
RESULTS_PATH = os.path.join(BASE_PATH, 'Results')

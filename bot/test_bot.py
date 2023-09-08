import logging
import os
import numpy as np
import pandas as pd

zoobot_dir = '/home/ydong-ext/zoobot'

import zoobot
# print(zoobot.__file__)
from zoobot.pytorch.training import finetune
from galaxy_datasets.pytorch.galaxy_datamodule import GalaxyDataModule

logging.basicConfig(level=logging.INFO)

# checkpoint downloaded from Dropbox
checkpoint_loc = 'checkpoints/effnetb0_greyscale_224px.ckpt'

save_dir = 'results/finetune_test_result'

label_cols = ['label']

datamodule = GalaxyDataModule(
    label_cols=label_cols,
    catalog=train_catalog,
    batch_size=4
)

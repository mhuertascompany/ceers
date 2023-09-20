'''
Test Zoobot by visualizing the representations of CEERS dataset using umap
'''


import logging
import os
import re
import numpy as np
import pandas as pd
import albumentations as A
import sys

sys.path.append('/scratch/ydong')

from bars.bot.To3d import To3d
from zoobot.pytorch.training import representations
from zoobot.pytorch.estimators import define_model
from zoobot.pytorch.predictions import predict_on_catalog
from zoobot.shared import load_predictions



logging.basicConfig(level=logging.INFO)

# checkpoint downloaded from Dropbox
checkpoint_loc = 'checkpoints/effnetb0_greyscale_224px.ckpt'

# use my own CEERS demo dataset
data_dir = 'images/demo_F200W'
images = [os.path.join(data_dir,path) for path in os.listdir(data_dir)]
labels = np.random.randint(2,size=len(images))
ids = [int(re.findall(r'\d+',path)[1]) for path in os.listdir(data_dir)]

catalog = pd.DataFrame({'id_str':ids,'file_loc':images,'label':labels})

# zoobot expects id_str and file_loc columns, so add these if needed

# save the representations here
# TODO change this to wherever you'd like
save_dir = 'no_finetune/repr_results'

# assert all([os.path.isfile(x) for x in catalog['file_loc']])

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# can load from either ZoobotTree checkpoint (if trained from scratch)
encoder = define_model.ZoobotTree.load_from_checkpoint(checkpoint_loc).encoder

# convert to simple pytorch lightning model
model = representations.ZoobotEncoder(encoder=encoder, pyramid=False)

label_cols = [f'feat_{n}' for n in range(1280)]
repr_loc = os.path.join(save_dir, 'F200W_representations.hdf5')

accelerator = 'gpu'
batch_size = 16
crop_scale_bounds=(0.7, 0.8)
crop_ratio_bounds=(0.9, 1.1)
resize_after_crop = 224

datamodule_kwargs = {
    'custom_albumentation_transform':A.Compose([
        A.Lambda(image=To3d(),always_apply=True),
        A.Rotate(limit=180, interpolation=1,
            always_apply=True, border_mode=0, value=0),
        A.RandomResizedCrop(
            height=resize_after_crop,  # after crop resize
            width=resize_after_crop,
            scale=crop_scale_bounds,  # crop factor
            ratio=crop_ratio_bounds,  # crop aspect ratio
            interpolation=1,  # This is "INTER_LINEAR" == BILINEAR interpolation. 
            always_apply=True
        ),  # new aspect ratio
        A.VerticalFlip(p=0.5),
])}
trainer_kwargs = {'devices': 1, 'accelerator': accelerator}
predict_on_catalog.predict(
    catalog,
    model,
    n_samples=1,
    label_cols=label_cols,
    save_loc=repr_loc,
    datamodule_kwargs=datamodule_kwargs,
    trainer_kwargs=trainer_kwargs
)

repr_df = load_predictions.single_forward_pass_hdf5s_to_df(repr_loc)
print(repr_df)
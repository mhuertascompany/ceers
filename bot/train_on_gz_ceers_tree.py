import logging
import os
import sys

# sys.path.append('/scratch/ydong')

import pandas as pd
import numpy as np
import albumentations as A
from sklearn.model_selection import train_test_split

from galaxy_datasets.pytorch.galaxy_datamodule import GalaxyDataModule

from zoobot.pytorch.training import finetune
from zoobot.pytorch.predictions import predict_on_catalog
from bot.gz_ceers_schema import gz_ceers_schema
from bot.To3d import To3d

os.environ['CUDA_VISIBLE_DEVICES']="1"

# import signal

# def sigterm_handler(signum, frame):
#     # Custom handling of SIGTERM
#     print("Received SIGTERM, ignoring...")

# Register the signal handler
# signal.signal(signal.SIGTERM, sigterm_handler)

logging.basicConfig(level=logging.INFO)

checkpoint_loc = 'checkpoints/effnetb0_greyscale_224px.ckpt'

save_dir = 'results/finetune_tree_result'

schema = gz_ceers_schema

accelerator = 'gpu'
devices = 1
batch_size = 64
prog_bar = False
max_galaxies = None

catalog = pd.read_csv("bot/match_catalog_F200W.csv")


train_val_catalog, test_catalog = train_test_split(catalog, test_size=0.2)
train_catalog, val_catalog = train_test_split(train_val_catalog, test_size=0.125)

crop_scale_bounds = (0.7, 0.8)
crop_ratio_bounds = (0.9, 1.1)
resize_after_crop = 224     # must match how checkpoint below was trained

datamodule = GalaxyDataModule(
    label_cols=schema.label_cols,
    train_catalog=train_catalog,
    val_catalog=val_catalog,
    test_catalog=test_catalog,
    batch_size=batch_size,
    custom_albumentation_transform=A.Compose([
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
    ]),  
)

model = finetune.FinetuneableZoobotTree(
    checkpoint_loc=checkpoint_loc,
    schema=schema
)

trainer = finetune.get_trainer(save_dir=save_dir, logger=None, accelerator=accelerator)
trainer.fit(model, datamodule)

# now save predictions on test set to evaluate performance
trainer_kwargs = {'devices': 1, 'accelerator': accelerator}
predict_on_catalog.predict(
    test_catalog,
    model,
    n_samples=1,
    label_cols=schema.label_cols,
    save_loc=os.path.join(save_dir, 'demo_tree_predictions.csv'),
    datamodule_kwargs={
        'custom_albumentation_transform':A.Compose([
            A.Lambda(image=To3d(),always_apply=True),
            A.Rotate(limit=180, interpolation=1,
                always_apply=True, border_mode=0, value=0),
            A.RandomResizedCrop(
                height=resize_after_crop,  # after crop resize
                width=resize_after_crop,
                scale=crop_scale_bounds,  # crop factor
                ratio=crop_ratio_bounds,  # crop aspect ratio
                interpolation=1,  # This is "INTER_LINEAR" == BILINEAR interpolation. See: https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html
                always_apply=True
            ),  # new aspect ratio
            A.VerticalFlip(p=0.5),
        ]),
        'batch_size':batch_size
    },
    trainer_kwargs=trainer_kwargs
)
'''
predict p_bar for all F200W images
'''


import os
import albumentations as A
import sys

sys.path.append('/n03data/huertas/python/ceers')

from zoobot.pytorch.training import finetune
from zoobot.pytorch.predictions import predict_on_catalog
from bot.To3d import To3d
from bot.gz_ceers_schema import gz_ceers_schema
from zoobot.pytorch.estimators import define_model
from astropy.table import Table

import numpy as np
import pandas as pd
import re

filter='f277w'


image_dir = f'/n03data/huertas/COSMOS-Web/zoobot/stamps/{filter}'
file_loc = [os.path.join(image_dir,path) for path in os.listdir(image_dir)]
file_loc = [os.path.join(image_dir, path) for path in os.listdir(image_dir) if path.endswith('.jpg')]

# Improved version with error handling and .jpg filtering
ids = []
for path in os.listdir(image_dir):
    if path.endswith('.jpg'):  # Process only files that end with .jpg
        numbers = re.findall(r'\d+', path)
        if len(numbers) > 1:  # Ensure there are at least two groups of digits
            ids.append(int(numbers[1]))
        else:
            print(f"Skipping file with insufficient digit groups: {path}")


#print(ids)
cat_dir = "/n03data/huertas/COSMOS-Web/cats"
cat_name = "COSMOSWeb_master_v3.1.0-sersic-cgs_err-calib_LePhare.fits"
cat_cosmos = Table.read(os.path.join(cat_dir,cat_name), format='fits')
#cat_cosmos = hdu[1].data
names = [name for name in cat_cosmos.colnames if len(cat_cosmos[name].shape) <= 1]
cat=cat_cosmos[names].to_pandas().iloc[ids]
#cat = pd.read_csv(os.path.join(cat_dir,cat_name)).iloc[ids]
cols = cat.columns

# for col in cols:
#     print(col)


pred_cat = cat
#print('number of objects',len(pred_cat))
pred_cat['id_str'] = ids
pred_cat['file_loc'] = file_loc


#checkpoint_loc = '/home/huertas/python/ceers/results/finetune_tree_result/checkpoints/97-v1.ckpt'
#checkpoint_loc = f'/n03data/huertas/CEERS/zoobot/models/finetune_tree_result/{filter}/checkpoints/99_effnet.ckpt'
checkpoint_loc = f'/n03data/huertas/CEERS/zoobot/models/finetune_tree_result/{filter}/checkpoints/99.ckpt' #nano model for f150w
#'results/finetune_tree_result/checkpoints/97-v1.ckpt'
# checkpoint_loc = 'checkpoints/effnetb0_greyscale_224px.ckpt'

save_dir = '/n03data/huertas/COSMOS-Web/cats/'

schema = gz_ceers_schema


batch_size = 64

crop_scale_bounds = (0.7, 0.8)
crop_ratio_bounds = (0.9, 1.1)
resize_after_crop = 224     # must match how checkpoint below was trained

model = finetune.FinetuneableZoobotTree.load_from_checkpoint(checkpoint_loc,schema=schema)
#model = finetune.FinetuneableZoobotTree(checkpoint_loc)
#model = define_model.ZoobotTree.load_from_checkpoint(checkpoint_loc, output_dim=39, question_index_groups=[])

#model = define_model.ZoobotTree.load_from_checkpoint(checkpoint_loc, output_dim=39, question_index_groups=[],question_answer_pairs=schema.question_answer_pairs,dependencies=schema.dependencies)

# now save predictions on test set to evaluate performance
trainer_kwargs = {'devices': 1, 'accelerator': 'gpu'}
predict_on_catalog.predict(
    pred_cat,
    model,
    n_samples=1,  # number of forward passes per galaxy
    label_cols=schema.label_cols,
    save_loc=os.path.join(save_dir, f'bars_COSMOS_3.1_{filter}_m27_nano.csv'),
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
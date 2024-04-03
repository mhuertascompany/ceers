'''
predict p_bar for all F200W images
'''


import os
import albumentations as A


from zoobot.pytorch.training import finetune
from zoobot.pytorch.predictions import predict_on_catalog
from bot.To3d import To3d
from bot.gz_ceers_schema import gz_ceers_schema
from zoobot.pytorch.estimators import define_model

import numpy as np
import pandas as pd
import re

FILTER = 200

image_dir = f'/scratch/ydong/stamps/demo_F{FILTER}W'
file_loc = [os.path.join(image_dir,path) for path in os.listdir(image_dir)]
ids = np.array([int(re.findall(r'\d+',path)[1]) for path in os.listdir(image_dir)])

cat_dir = "/scratch/ydong/cat"
cat_name = "CEERS_DR05_adversarial_asinh_4filters_1122_4class_ensemble_v02_stellar_params_morphflag_delta_10points_DenseBasis_galfit_CLASS_STAR_v052_bug.csv"
cat = pd.read_csv(os.path.join(cat_dir,cat_name)).iloc[ids]
cols = cat.columns

# for col in cols:
#     print(col)


pred_cat = cat
pred_cat['id_str'] = ids
pred_cat['file_loc'] = file_loc

checkpoint_loc = f'results/finetune_tree_result/checkpoints/F{FILTER}W_ckpt/97-v1.ckpt'
# checkpoint_loc = 'checkpoints/effnetb0_greyscale_224px.ckpt'

save_dir = f'bar_estimate/F{FILTER}W_pred'

schema = gz_ceers_schema

batch_size = 64

crop_scale_bounds = (0.7, 0.8)
crop_ratio_bounds = (0.9, 1.1)
resize_after_crop = 224     # must match how checkpoint below was trained

# model = finetune.FinetuneableZoobotTree(
#     checkpoint_loc=checkpoint_loc,
#     schema=schema
# )
model = define_model.ZoobotTree.load_from_checkpoint(checkpoint_loc, output_dim=39, question_index_groups=[])


# now save predictions on test set to evaluate performance
trainer_kwargs = {'devices': 1, 'accelerator': 'gpu'}
predict_on_catalog.predict(
    pred_cat,
    model,
    n_samples=5,  # number of forward passes per galaxy
    label_cols=schema.label_cols,
    save_loc=os.path.join(save_dir, f'full_cat_predictions_F{FILTER}W_0.csv'),
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
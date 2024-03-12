'''
Try running Zoobot. 
'''

import logging
import os
import numpy as np
import pandas as pd
import albumentations as A
import re

zoobot_dir = '/home/ydong-ext/zoobot'


from zoobot.pytorch.training import finetune
from zoobot.pytorch.predictions import predict_on_catalog
from galaxy_datasets.pytorch.galaxy_datamodule import GalaxyDataModule

logging.basicConfig(level=logging.INFO)

# checkpoint downloaded from Dropbox
checkpoint_loc = 'checkpoints/effnetb0_greyscale_224px.ckpt'

save_dir = 'results/finetune_test_result'

num_classes = 2

# build a catalog of images with labels
train_dir = 'images/demo_train'
test_dir = 'images/demo_test'

train_image = [os.path.join(train_dir,path) for path in os.listdir(train_dir)]
test_image = [os.path.join(test_dir,path) for path in os.listdir(test_dir)]

train_label = np.random.randint(2,size=len(train_image))
test_label = np.random.randint(2,size=len(test_image))

train_ids = [int(re.search(r'\d+',path).group()) for path in os.listdir(train_dir)]
test_ids = [int(re.search(r'\d+',path).group()) for path in os.listdir(test_dir)]

train_catalog = pd.DataFrame({'id_str':train_ids,'file_loc':train_image,'label':train_label})
test_catalog = pd.DataFrame({'id_str':test_ids,'file_loc':test_image,'label':test_label})


label_cols = ['label']

class To3d:
    def __init__(self):
        pass

    def __call__(self, image, **kwargs):
        x, y = image.shape
        return image.reshape(x,y,1)

crop_scale_bounds = (0.7, 0.8)
crop_ratio_bounds = (0.9, 1.1)
resize_after_crop = 224

datamodule = GalaxyDataModule(
    label_cols=label_cols,
    catalog=train_catalog,
    batch_size=32,
    custom_albumentation_transform=A.Compose([
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
    num_workers=2
)

model = finetune.FinetuneableZoobotClassifier(
    checkpoint_loc=checkpoint_loc,
    num_classes=2,
    n_layers=0  # only updating the head weights. Set e.g. 1, 2 to finetune deeper. 
)

# retrain to finetune
trainer = finetune.get_trainer(save_dir, devices=1, accelerator='cuda', max_epochs=10)
trainer.fit(model, datamodule)

best_checkpoint = trainer.checkpoint_callback.best_model_path
finetuned_model = finetune.FinetuneableZoobotClassifier.load_from_checkpoint(best_checkpoint)

predict_on_catalog.predict(
    test_catalog,
    finetuned_model,
    n_samples=1,
    label_cols=label_cols,
    save_loc=os.path.join(save_dir, 'demo_finetuned_predictions.csv'),
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
    ])}
    # trainer_kwargs={'accelerator': 'gpu'}
)

import numpy as np
import albumentations as A


class To3d:
    def __init__(self):
        pass

    def __call__(self, image, **kwargs):
        x, y = image.shape
        return image.reshape(x,y,1)


custom_albumentation_transform = A.Compose([
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
])

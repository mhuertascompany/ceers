import numpy as np
import albumentations as A
from PIL import Image


class To3d:
    def __init__(self):
        pass

    def __call__(self, image, **kwargs):
        x, y = image.shape
        return image.reshape(x,y,1)

img = Image.open('/home/ydong-ext/demo_rings/images/102960/102960_1416.jpg')
img = np.array(img)
print(type(img))
print(img.shape)

crop_scale_bounds = (0.7, 0.8)
crop_ratio_bounds = (0.9, 1.1)
resize_after_crop = 224

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

arr = custom_albumentation_transform(image=img)['image']
print(arr.shape)

arr = np.transpose(arr,axes=[2,0,1])
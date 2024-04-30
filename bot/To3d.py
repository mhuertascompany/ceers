'''
Solve the problem that Zoobot can't process 2d images. Convert 2d greyscales to 3d. 
'''

import numpy as np
import albumentations as A


#class To3d:
 #   def __init__(self):
 #       pass

#    def __call__(self, image, **kwargs):
#        x, y = image.shape
#        return image.reshape(x,y,1)




class To3d:
    def __init__(self):
        pass

    def __call__(self, image, **kwargs):
        # Ensure the input image is 2D (grayscale)
        if image.ndim != 2:
            raise ValueError("Input image must be a 2D array")
        
        # Replicate the image across three channels
        return np.stack((image, image, image), axis=-1)

import numpy as np
import matplotlib 
from morpheus.classifier import Classifier
from morpheus.data import example
import matplotlib.pyplot as plt
from astropy.io import fits
import pickle

# open image
hdul = fits.open("/net/diva/scratch-ssd1/mhuertas/data/CEERS/ceers5_f150w_i2d.fits.gz")

image =  hdul[1].data

# convert to e/s
image_electrons = image/hdul[1].header['PHOTMJSR']

print("Classifying...")
classified = Classifier.classify(h=image_electrons, j=image_electrons, v=image_electrons, z=image_electrons)

print("Saving classifier...")
#save classifier
file = open('/net/diva/scratch-ssd1/mhuertas/data/CEERS/classifier_morpheus_ceers5_f150w_i2d.pkl', 'w')
pickle.dump(classified, file)
file.close()

mask = np.zeros_like(image_electrons, np.int)
mask[5:-5, 5:-5] = 1


print("Creating Segmentation Map...")
#create and write segmap
segmap = Classifier.segmap_from_classified(classified, image_electrons, mask=mask)
hdu = fits.PrimaryHDU(segmap)
hdul = fits.HDUList([hdu])
hdul.writeto('/net/diva/scratch-ssd1/mhuertas/data/CEERS/segmap_morpheus_ceers5_f150w_i2d.fits')

print("Creating Catalog...")
catalog = Classifier.catalog_from_classified(classified, image_electrons, segmap)

print("Writing Catalog...")
#write catalog
file = open('/net/diva/scratch-ssd1/mhuertas/data/CEERS/catalog_morpheus_ceers5_f150w_i2d.pkl', 'w')
pickle.dump(catalog, file)
file.close()






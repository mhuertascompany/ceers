from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.python.keras import Model
import imageio
import pandas as pd
import numpy as np
import IPython.display as display
import os
import pickle
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D
from astropy.coordinates import SkyCoord

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from official.vision.image_classification.augment import RandAugment
import matplotlib.pyplot as plt


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"


## load data with CEERS noise
TNG200 = np.load("/scratch/mhuertas/CEERS/TNG50/image_CEERS_64pix_F200W_aug0.npy")
TNG356 = np.load("/scratch/mhuertas/CEERS/TNG50/image_CEERS_64pix_F356W_aug0.npy")
ids = np.load("/scratch/mhuertas/CEERS/TNG50/id_TNG50_64pix_aug0.npy",allow_pickle=True)

X=[]
for stamp in TNG200:
    if np.max(stamp)<=0 or np.count_nonzero(stamp==0)>10:
        continue

    transform = AsinhStretch() + interval
    norm = transform(stamp[32-16:32+16,32-16+32+16])
    X.append(stamp)


TNG_X = tf.convert_to_tensor(X, dtype=tf.float32)
TNG_X = tf.expand_dims(X, -1)


##load models trained on CEERS

data_path = "/scratch/mhuertas/CEERS/"








## create models again .. not very clean I know
def get_network(image_size=32, num_classes=4):
  inputs = keras.Input(shape=(image_size, image_size, 1))
  rot = tf.keras.layers.RandomRotation(0.25)(inputs)
  flip = tf.keras.layers.RandomFlip()(rot)
  conv1 = layers.Conv2D(
        16,
        (2, 2),
        strides=1,
        padding="same",
        activation = "relu"
    )(flip)
  bn1 = layers.BatchNormalization()(conv1)
  conv2 =  layers.Conv2D(
        64,
        (2, 2),
        strides=2,
        padding="same",
        activation = "relu"
    )(bn1)
  mp1 = layers.MaxPool2D((2,2))(conv2)
  bn2 =  layers.BatchNormalization()(mp1)
  conv3 =  layers.Conv2D(
        128,
        (3, 3),
        strides=1,
        padding="same",
        activation = "relu"
    )(bn2)
  mp2 = layers.MaxPool2D((2,2))(conv3)
  bn3 = layers.BatchNormalization()(mp2)
  conv4 =  layers.Conv2D(
        128,
        (2, 2),
        strides=2,
        padding="same",
        activation = "relu"
    )(bn3)
  bn4 = layers.BatchNormalization()(conv4)
  conv5 =  layers.Conv2D(
        128,
        (2, 2),
        strides=2,
        padding="same",
        activation = "relu"
    )(bn4)
  bn5 = layers.BatchNormalization()(conv5)
  #trunk_outputs = layers.GlobalAveragePooling2D()(bn5)
  outputs = layers.Flatten()(bn5) 
  #d1 = layers.Dense(64, activation = "relu")(fl)
  outputs = layers.Dropout(0.4)(outputs)
  #outputs = layers.Dense(num_classes)(dr1)
  return keras.Model(inputs, outputs)

feature_generator = get_network()


class LabelPredictor(Model):
  def __init__(self):
    super(LabelPredictor, self).__init__() 
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(4, activation='softmax')
    self.dr = layers.Dropout(0.4)

  def call(self, feats):  
    feats = self.d1(feats)
    feats = self.dr(feats)
    return self.d2(feats)

label_predictor = LabelPredictor()




filter = 'f200w'
label_predictor.load_weights(data_path+"adversarial_label_irr01_asinh_resnet_"+filter+"v01_0910.weights")
feature_generator.load_weights(data_path+"adversarial__irr01_asinh_resnet_"+filter+"v01_0910.weights")


chunk=1000

sph=[]
dk=[]
irr=[]
bd=[]
    
n=0    
while(n<len(TNG200)):
    if n+chunk>len(TNG200):
        p = label_predictor(feature_generator(TNG200[n:]))
    else:    
        p = label_predictor(feature_generator(TNG200[n:n+chunk]))
    n=n+chunk
    print(len(p))
    sph.append(p[:,0])
    dk.append(p[:,1])
    irr.append(p[:,2])
    bd.append(p[:,3])
    
#df = pd.DataFrame(list(zip(idvec,ravec,decvec,np.concatenate(sph).ravel(),np.concatenate(dk).ravel(),np.concatenate(irr).ravel())),columns =['ID_CEERS', 'ra','dec','sph','disk','irr'])


df = pd.DataFrame(list(zip(ids,np.concatenate(sph).ravel(),np.concatenate(dk).ravel(),np.concatenate(irr).ravel(),np.concatenate(bd).ravel())),columns =['ID_TNG50','sph','disk','irr','bd'])
df.to_csv(data_path+"TNG50_v01_adversarial_irr01_asinh_"+filter+"_0920_4class.csv")






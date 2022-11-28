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

from sklearn.utils import shuffle
import pdb
from astropy.visualization import MinMaxInterval
interval = MinMaxInterval()
from astropy.visualization import AsinhStretch,LogStretch


import os

import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"


## read data

filter = "f444w"
shuffle=True



data_path = "/scratch/mhuertas/CEERS/data_release/"
candels_cat = pd.read_csv(data_path+"cats/CANDELS_morphology.csv")


cat_ceers =   pd.read_hdf(data_path+"cats/ceers_v0.2_photoz_stellar_params_nov17.hdf5",key='data')

ceers_pointings = ["1","2","3","6"]
nir_f200_list=[]
w=[]
cats = []
for c in ceers_pointings:
  nir_f200 = fits.open(data_path+"images/hlsp_ceers_jwst_nircam_nircam"+c+"_"+filter+"_dr0.5_i2d.fits.gz")
  nir_f200_list.append(nir_f200)
  w.append(WCS(nir_f200[1].header))
  cats.append(cat_ceers.query("FIELD=="+c))


wfc3_f160_list=[]
wf160=[]
candels_images = ["hlsp_candels_hst_wfc3_egs-tot-60mas_f160w_v1.0_drz.fits","hlsp_candels_hst_wfc3_gs-tot_f160w_v1.0_drz.fits","hlsp_candels_hst_wfc3_cos-tot_f160w_v1.0_drz.fits","hlsp_candels_hst_wfc3_uds-tot_f160w_v1.0_drz.fits"]
for c in candels_images:
    wfc3_f160 = fits.open(data_path+"images/"+c)
    wfc3_f160_list.append(wfc3_f160)
    wf160.append(WCS(wfc3_f160[0].header))
    wf160[-1].sip = None



fields = ["egs","GDS","COSMOS","UDS"]
X=[]
label=[]

for wfc3_f160,w_c,f in zip(wfc3_f160_list,wf160,fields): 
    sel = candels_cat.query('HMAG<24 and FIELD=='+'"'+f+'"')
    print(len(sel))
    
    for idn,ra,dec,fsph,fdk,firr in zip(sel.RB_ID,sel.RA,sel.DEC,sel.F_SPHEROID,sel.F_DISK,sel.F_IRR):
            
            try:
                
                position = SkyCoord(ra,dec,unit="deg")
                #print(ra,dec)
                
                stamp = Cutout2D(wfc3_f160[0].data,position,32,wcs=w_c)
                
                
                
                if np.max(stamp.data)<=0 or np.count_nonzero(stamp.data==0)>10:
                    continue
                
                
                transform = AsinhStretch() + interval
                #transform = LogStretch() + interval
                norm = transform(stamp.data)
                
                #stamp_name = data_path+"NirCam/CANDELS_stamps/v005/f200fullres/CANDELS-CEERS"+str(idn)+"_f200w_v005.fits"

                if (fsph>=0.66 and fdk<0.66 and firr<0.1):
                    label.append(0)
                    
                    X.append(norm)
                elif ((fsph<0.66 and fdk>=0.66 and firr<0.1)):
                    label.append(1)
                    X.append(norm)
                elif ((firr>=0.1)):
                    label.append(2) 
                    X.append(norm)
                elif ((fsph>0.66 and fdk>0.66 and firr<0.1)):
                    label.append(3)
                    X.append(norm)     


            except:
                continue


if shuffle==True:
    X, label = shuffle(X, label, random_state=0)

X_JWST=[]
idvec=[]
fullvec=[]
fieldvec=[]
ravec=[]
decvec=[]
for nir_f200,w_v,cat in zip(nir_f200_list,w,cats):
  
  sel = cat.query('gf_F200W_MAG<26 and gf_F200W_MAG>0')
  #print(cat)  
  for full,idn, field, ra,dec in zip(sel.fullname,sel.id, sel.FIELD, sel.RA,sel.DEC):
          try:
              position = SkyCoord(ra,dec,unit="deg")
            #print(ra,dec)
              stamp = Cutout2D(nir_f200['SCI'].data,position,32,wcs=w_v)
            
              if np.max(stamp.data)<=0 or np.count_nonzero(stamp.data==0)>10:
                  continue
            
              transform = AsinhStretch() + interval
              norm = transform(stamp.data)  
            #pdb.set_trace()
            #stamp_name = data_path+"NirCam/CANDELS_stamps/v005/f200fullres/CANDELS-CEERS"+str(idn)+"_f200w_v005.fits"
              X_JWST.append(norm)
              idvec.append(idn)
              fullvec.append(full)
              fieldvec.append(field) 
              ravec.append(ra)
              decvec.append(dec)  
            #if (fsph>0.66 and fdk<0.66 and firr<0.1):
            #    label.append(1)
            #else:
            #    label.append(0)
          
            
          except:
              continue





train_s=len(X)*4//5
test_s=len(X)*1//5

print(train_s)
print(test_s)



CANDELS_X = tf.convert_to_tensor(X[0:train_s], dtype=tf.float32)
CANDELS_X = tf.expand_dims(CANDELS_X, -1)
#CANDELS_X = tf.tile(CANDELS_X, [1,1,1,3])
print(tf.shape(CANDELS_X))
label_candels = tf.one_hot(label[0:train_s], 4).numpy()

CANDELS_X_t = tf.convert_to_tensor(X[train_s:train_s+test_s], dtype=tf.float32)
CANDELS_X_t = tf.expand_dims(CANDELS_X_t, -1)
#CANDELS_X = tf.tile(CANDELS_X, [1,1,1,3])
print(tf.shape(CANDELS_X_t))
label_candels_t = tf.one_hot(label[train_s:train_s+test_s], 4).numpy()

JWST_X = tf.convert_to_tensor(X_JWST, dtype=tf.float32)
JWST_X = tf.expand_dims(JWST_X, -1)
label_JWST = np.zeros(len(JWST_X))
#JWST_X = tf.tile(JWST_X, [1,1,1,3])
label_JWST = tf.one_hot(np.zeros(len(JWST_X)), 4).numpy()




all_train_domain_images = np.vstack((CANDELS_X, JWST_X))
channel_mean = all_train_domain_images.mean((0,1,2))

train_ds = tf.data.Dataset.from_tensor_slices(
    (CANDELS_X, label_candels)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((CANDELS_X_t, label_candels_t)).batch(32)



mnist_m_train_ds = tf.data.Dataset.from_tensor_slices((JWST_X,tf.cast(label_JWST, tf.int8))).batch(32)
mnist_m_test_ds = tf.data.Dataset.from_tensor_slices((JWST_X,tf.cast(label_JWST, tf.int8))).batch(32)


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

class DomainPredictor(Model):
  def __init__(self):
    super(DomainPredictor, self).__init__()   
    self.d3 = Dense(128, activation='relu')
    self.d4 = Dense(64, activation='relu')
    self.d5 = Dense(2, activation='softmax')
    self.dr = layers.Dropout(0.4)

  def call(self, feats):
    feats = self.d3(feats)
    feats = self.d4(feats)
    feats = self.dr(feats)
    return self.d5(feats)

domain_predictor = DomainPredictor()

x_train_domain_labels = np.ones([len(label_candels)])
mnist_m_train_domain_labels = np.zeros([len(label_JWST)])
all_train_domain_labels = np.hstack((x_train_domain_labels, mnist_m_train_domain_labels))
all_train_domain_labels = tf.one_hot(np.zeros(len(all_train_domain_labels)), 2).numpy()
domain_train_ds = tf.data.Dataset.from_tensor_slices(
    (all_train_domain_images, tf.cast(all_train_domain_labels, tf.int8))).shuffle(60000).batch(32)

loss_object = tf.keras.losses.CategoricalCrossentropy()
d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
f_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

m_test_loss = tf.keras.metrics.Mean(name='m_test_loss')
m_test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='m_test_accuracy')

conf_train_loss = tf.keras.metrics.Mean(name='c_train_loss')
conf_train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='c_train_accuracy')

@tf.function
def train_step(images, labels, images2, domains,alpha):
    
  """
  i. images = batch of source images
  ii. labels = corresponding labels
  iii. images2 = batch of source and target images
  iv. domains = corresponding domain labels
  v. alpha = weight attributed to the domain loss
  """
    
  ## Update the generator and the classifier
   
  with tf.GradientTape(persistent=True) as tape:
     
    features = feature_generator(images)
    l_predictions = label_predictor(features)
    #pdb.set_trace()
    #print(l_predictions.eval())
    features = feature_generator(images2)
    #pdb.set_trace()
    d_predictions = domain_predictor(features)
    label_loss = loss_object(labels, l_predictions)
    domain_loss = loss_object(domains, d_predictions)
    
  f_gradients_on_label_loss = tape.gradient(label_loss, feature_generator.trainable_variables)
  f_gradients_on_domain_loss = tape.gradient(domain_loss, feature_generator.trainable_variables)    
  f_gradients = [f_gradients_on_label_loss[i] - alpha*f_gradients_on_domain_loss[
      i] for i in range(len(f_gradients_on_domain_loss))]

    
  l_gradients = tape.gradient(label_loss, label_predictor.trainable_variables)

  f_optimizer.apply_gradients(zip(f_gradients+l_gradients, 
                                  feature_generator.trainable_variables+label_predictor.trainable_variables)) 
    
    
  ## Update the discriminator: Comment this bit to complete all updates in one step. Asynchronous updating 
  ## seems to work a bit better, with better accuracy and stability, but may take longer to train    
  with tf.GradientTape() as tape:
    features = feature_generator(images2)
    d_predictions = domain_predictor(features)
    #print(d_predictions)
    domain_loss = loss_object(domains, d_predictions)
  #####
   
  d_gradients = tape.gradient(domain_loss, domain_predictor.trainable_variables)  
  d_gradients = [alpha*i for i in d_gradients]
  d_optimizer.apply_gradients(zip(d_gradients, domain_predictor.trainable_variables))
  
    
    
  train_loss(label_loss)
  print(label_loss)  
  train_accuracy(labels, l_predictions)
  conf_train_loss(domain_loss)
  conf_train_accuracy(domains, d_predictions)

@tf.function
def test_step(mnist_images, labels, mnist_m_images, labels2):
  #pdb.set_trace()  
  features = feature_generator(mnist_images)
  predictions = label_predictor(features)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

  features = feature_generator(mnist_m_images)
  predictions = label_predictor(features)
  t_loss = loss_object(labels2, predictions)
    
  m_test_loss(t_loss)
  m_test_accuracy(labels2, predictions)


def reset_metrics():
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
    m_test_loss.reset_states()
    m_test_accuracy.reset_states()


EPOCHS = 50

alpha = 1
for epoch in range(EPOCHS):
  reset_metrics()
  
  for domain_data, label_data in zip(domain_train_ds, train_ds):
    
    try:
      train_step(label_data[0], label_data[1], domain_data[0], domain_data[1], alpha=alpha)
     
    #End of the smaller dataset
    except ValueError: 
      pass
    
  for test_data, m_test_data in zip(test_ds,mnist_m_test_ds):
    test_step(test_data[0], test_data[1], m_test_data[0], m_test_data[1])
  
  template = 'Epoch {}, Train Accuracy: {}, Domain Accuracy: {}, Source Test Accuracy: {}, Target Test Accuracy: {}'
  print (template.format(epoch+1,
                         train_accuracy.result()*100,
                         conf_train_accuracy.result()*100,
                         test_accuracy.result()*100,
                         m_test_accuracy.result()*100,))


label_predictor.save_weights(data_path+"models/adversarial_asinh_resnet_"+filter+"vDR05_1122_shuffle.weights")
feature_generator.save_weights(data_path+"models/adversarial_asinh_resnet_"+filter+"vDR05_1122_shuffle.weights")


chunk=1000

sph=[]
dk=[]
irr=[]
bd=[]
    
n=0    
while(n<len(JWST_X)):
    if n+chunk>len(JWST_X):
        p = label_predictor(feature_generator(JWST_X[n:]))
    else:    
        p = label_predictor(feature_generator(JWST_X[n:n+chunk]))
    n=n+chunk
    print(len(p))
    sph.append(p[:,0])
    dk.append(p[:,1])
    irr.append(p[:,2])
    bd.append(p[:,3])
    
#df = pd.DataFrame(list(zip(idvec,ravec,decvec,np.concatenate(sph).ravel(),np.concatenate(dk).ravel(),np.concatenate(irr).ravel())),columns =['ID_CEERS', 'ra','dec','sph','disk','irr'])


df = pd.DataFrame(list(zip(fullvec,idvec,fieldvec,ravec,decvec,np.concatenate(sph).ravel(),np.concatenate(dk).ravel(),np.concatenate(irr).ravel(),np.concatenate(bd).ravel())),columns =['fullname','id','FIELD', 'ra','dec','sph','disk','irr','bd'])
df.to_csv(data_path+"cats/CEERS_DR05_adversarial_asinh_"+filter+"_1122_4class_shuffle.csv")



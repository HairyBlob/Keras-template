# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import nibabel as nib
from scipy import ndimage
from intensity_normalization.normalize import zscore
from tensorflow_addons.layers import InstanceNormalization

class TensorBoardImage(tf.keras.callbacks.Callback):
    def __init__(self, val_data):
        super().__init__() 
        #Il faut lui fournir l'accès aux données de validation
        self.validation_data = val_data
        
    #On définit une fonction qui log les résultats à la fin de chaque epoch
    def on_epoch_end(self, epoch, logs={}):
        #changer le nom ici à chaque nouvel entraînement
        writer = tf.summary.create_file_writer('./logs')
        #Comme les données sont 3D on doit sélectionner une slice (70) et on display 3 des MRI en couleur (0:3)
        #Comme on peut logger 3 de large sur tensorboard on prend les 3 premiers exemples
        input_img = self.validation_data[0][0:3,:,:,70,0:3]
        label_img = self.validation_data[1][0:3,:,:,70,0:3]
        output_img = self.model.predict(self.validation_data[0][0:3])[:,:,:,70,:]
        
        with writer.as_default():
            tf.summary.image(name = 'input', data = input_img, step = epoch)
            tf.summary.image(name = 'label', data = label_img, step = epoch)
            tf.summary.image(name = 'output', data = output_img, step = epoch)
            tf.summary.image(name = 'seg', data = np.round(output_img), step = epoch)
        writer.flush()

        return

#Fonction de crop hard-codée et bien dégueu, svp ne pas imiter
def crop(flair, T1, T1CE, T2, seg):
    #On assume que le centre de masse est relativement constant entre les patients
    CM = ndimage.measurements.center_of_mass(T1.astype(np.float32))
    off = 0
    if int(CM[2])+74 > flair.shape[2] or int(CM[2])-74 < 0:
        if int(CM[2])+74 > flair.shape[2]:
            off = flair.shape[2] - (int(CM[2])+74)
        elif int(CM[2])-74 < 0:
            off = -(int(CM[2])-74)

    return flair[int(CM[0])-82+off:int(CM[0])+73+off, int(CM[1])-107+off:int(CM[1])+87+off,int(CM[2])-74+off:int(CM[2])+74+off], T1[int(CM[0])-82+off:int(CM[0])+73+off, int(CM[1])-107+off:int(CM[1])+87+off,int(CM[2])-74+off:int(CM[2])+74+off], T1CE[int(CM[0])-82+off:int(CM[0])+73+off, int(CM[1])-107+off:int(CM[1])+87+off,int(CM[2])-74+off:int(CM[2])+74+off], T2[int(CM[0])-82+off:int(CM[0])+73+off, int(CM[1])-107+off:int(CM[1])+87+off,int(CM[2])-74+off:int(CM[2])+74+off], seg[int(CM[0])-82+off:int(CM[0])+73+off, int(CM[1])-107+off:int(CM[1])+87+off,int(CM[2])-74+off:int(CM[2])+74+off]

#Fonction principale de data loading pour partitioner en train/validation. 
#La fonction de load pour les données test n'est pas écrite
def get_T1_data(CV=True, fold=0, foldNBR = 10, resample = 0):

    train_MRI = []

    test_MRI = []

    train_seg = []
    test_seg = []
    seg = 0
    T1 = 0
    T2 = 0
    T1CE= 0
    flair = 0
    i = 0
    
    for dirpath, dirnames, filenames in os.walk('D:\BRATS'):
        for filename in [f for f in filenames if (f.endswith("t1.nii.gz") or f.endswith("t2.nii.gz") or f.endswith("seg.nii.gz") or f.endswith("t1ce.nii.gz") or f.endswith("flair.nii.gz"))]:
            img = nib.load(os.path.join(dirpath,filename))
            if (i + fold) % foldNBR == 0 and CV:
                if filename.endswith("seg.nii.gz"):
                    #On binarise la segmentation, si on veut segmenter les 4 classes ce n'est pas nécessaire
                    seg = nib.Nifti1Image(img.get_fdata().astype(bool).astype(int), img.affine)
                    
                if filename.endswith("flair.nii.gz"):
                    #Le package intensity-normalization fait la normalisation pour nous
                    flair = zscore.zscore_normalize(img, nib.Nifti1Image(img.get_fdata().astype(bool).astype(int), img.affine))
                    
                if filename.endswith("t1ce.nii.gz"):
                    T1CE = zscore.zscore_normalize(img, nib.Nifti1Image(img.get_fdata().astype(bool).astype(int), img.affine))
                    
                if filename.endswith("t1.nii.gz"):
                    T1 = zscore.zscore_normalize(img, nib.Nifti1Image(img.get_fdata().astype(bool).astype(int), img.affine))
                    
                if filename.endswith("t2.nii.gz"):
                    T2 = zscore.zscore_normalize(img, nib.Nifti1Image(img.get_fdata().astype(bool).astype(int), img.affine))
                    #On crop les 5 images de la même façon
                    flair, T1, T1CE, T2, seg = crop(flair.get_fdata(), T1.get_fdata(), T1CE.get_fdata(), T2.get_fdata(), seg.get_fdata())
                    #On reshape pour avoir le format attendu par le network
                    seg = np.reshape(seg, (155, 194, 148, 1))
                    T1 = np.reshape(T1, (155, 194, 148, 1))
                    T2 = np.reshape(T2, (155, 194, 148, 1))
                    T1CE = np.reshape(T1CE, (155, 194, 148, 1))
                    flair = np.reshape(flair, (155, 194, 148, 1))
                    combined = np.concatenate((T1,T2,T1CE,flair), axis = -1)
                    
                    #IMPORTANT On cast le tout en float16 pour limiter l'espace utilisé
                    test_seg.append(seg.astype(np.float16))
                    test_MRI.append(combined.astype(np.float16))
                    
                    i += 1
                    
            else:
                if filename.endswith("seg.nii.gz"):
                    seg = nib.Nifti1Image(img.get_fdata().astype(bool).astype(int), img.affine)
                    
                if filename.endswith("flair.nii.gz"):
                    flair = zscore.zscore_normalize(img, nib.Nifti1Image(img.get_fdata().astype(bool).astype(int), img.affine))
                    
                if filename.endswith("t1ce.nii.gz"):
                    T1CE = zscore.zscore_normalize(img, nib.Nifti1Image(img.get_fdata().astype(bool).astype(int), img.affine))
                    
                if filename.endswith("t1.nii.gz"):
                    T1 = zscore.zscore_normalize(img, nib.Nifti1Image(img.get_fdata().astype(bool).astype(int), img.affine))
                    
                if filename.endswith("t2.nii.gz"):
                    T2 = zscore.zscore_normalize(img, nib.Nifti1Image(img.get_fdata().astype(bool).astype(int), img.affine))
                    
                    flair, T1, T1CE, T2, seg = crop(flair.get_fdata(), T1.get_fdata(), T1CE.get_fdata(), T2.get_fdata(), seg.get_fdata())
                    seg = np.reshape(seg, (155, 194, 148, 1))
                    T1 = np.reshape(T1, (155, 194, 148, 1))
                    T2 = np.reshape(T2, (155, 194, 148, 1))
                    T1CE = np.reshape(T1CE, (155, 194, 148, 1))
                    flair = np.reshape(flair, (155, 194, 148, 1))
                    combined = np.concatenate((T1,T2,T1CE,flair), axis = -1)
                    
                    train_seg.append(seg.astype(np.float16))
                    train_MRI.append(combined.astype(np.float16))
                    
                    i += 1
                        
    return np.array(train_MRI), np.array(train_seg), np.array(test_MRI), np.array(test_seg)

BATCH_SIZE = 1
dim_1 = 155
dim_2 = 194
dim_3 = 148

train_MRI, train_seg, test_MRI, test_seg = get_T1_data()

#Vérifions la taille du dataset
print(test_MRI.shape)
print(train_MRI.shape)

tbi_callback = TensorBoardImage((test_MRI, test_seg))

def show_batch(image_batch, label_batch):
  ax = plt.subplot(2,1,1)
  plt.imshow(image_batch[:,:,70,0].astype(np.float32))
  ax = plt.subplot(2,1,2)
  plt.imshow(label_batch[:,:,70,0].astype(np.float32))
  plt.axis('off')
  plt.show()

#Vérifions que les données sont encore belles
show_batch(test_MRI[1],test_seg[1])

#Le modèle est défini comme une suite d'opérations séquentielles. Notez l'activation sigmoid en sortie 
model = tf.keras.Sequential([
  tf.keras.layers.Conv3D(4, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', input_shape=(dim_1,dim_2,dim_3, 4), padding='same'),
  InstanceNormalization(),
  tf.keras.layers.Conv3D(4, (3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same'),
  InstanceNormalization(),
  tf.keras.layers.Conv3D(4, (3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same'),
  InstanceNormalization(),
  tf.keras.layers.Conv3D(4, (3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same'),
  InstanceNormalization(),
  tf.keras.layers.Conv3D(1, (3, 3, 3), strides=(1, 1, 1), activation='sigmoid', padding='same')
])


base_learning_rate = 0.0001
#Si on veut loader un modèle déjà entraîné
#model = tf.keras.models.load_model('FCNN.h5', compile=False)

def dice_loss(y_true, y_pred):
    dice = tf.reduce_mean(tf.math.multiply(y_true, y_pred))/(tf.reduce_mean(y_true) + tf.reduce_mean(y_pred))*2
    return 1 - dice

def weighted_bce(y_true, y_pred, weight):
  weights = (y_true * weight) + 1.
  #From logits = False puisqu'on a appliqué une sigmoide
  bce = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits = False)
  weighted_bce = tf.reduce_mean(tf.expand_dims(bce, axis = -1) * weights)
  return weighted_bce

#Si on veut optimiser la combinaison des fonctions de cout
def combined(y_true, y_pred):
    return dice_loss(y_true, y_pred) + weighted_bce(y_true, y_pred)

model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=dice_loss,
              metrics=['accuracy', dice_loss])

model.summary() #Permet de voir le modèle en ligne de commande

len(model.trainable_variables)

# ### Train the model
# 
# 

history = model.fit(x = train_MRI, y = train_seg, #On peut passer les arrays numpy en input
                    epochs= 70,
                    batch_size = 1, #Petite batch_size en raison de la taille des images
                    validation_data = (test_MRI, test_seg), 
                    verbose = 2,
                    callbacks=[tbi_callback]) #mettre les hooks pour le tensorboard ici

model.save('FCNN_BCE_4.h5') # Sauvons ce résultat
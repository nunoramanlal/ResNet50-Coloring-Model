import glob
import numpy as np
import tensorflow as tf
import random as rn
from model_utils import generate_mobilenet_model
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import load_img, img_to_array
import keras
from skimage.io import imsave, imread
from keras.applications.mobilenet import preprocess_input
import os
import skimage

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
tf.set_random_seed(1234)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf.DeviceCountEntry
session_conf.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(graph=tf.get_default_graph(),config=session_conf))

_TRAINING_IMAGES = '../../../../home/nramanlal/data-nramanlal-lfreixinho/resized_images_sf/'
#_TRAINING_IMAGES = '../resized_images_sf2/'
_MODEL_NAME = 'mobilenet_model_color.h5'
_MODEL_WEIGHTS_NAME = 'weights_mobilenet_model_color.h5'

def get_images(path):
    images = []
    data = os.listdir(path)
    for entry in data:
        images.append(entry.rstrip())
    return images[:70000]
	
def generator(X, img_path, bc):
	while 1:
		new_X1 = np.zeros((bc, 224, 224, 1))
		new_X2 = np.zeros((bc, 224, 224, 3))
		new_Y1 = np.zeros((bc, 224, 224, 2))
		count = 0
		for entry in X:
			if count < bc:
				x_b = imread(img_path + str(entry))
				x_b = np.expand_dims(x_b, axis=0)
				x_b = np.array(x_b)

				grayscaled_rgb = gray2rgb(rgb2gray(x_b))  # convert to 3 channeled grayscale image
				grayscaled_rgb = np.array(grayscaled_rgb)*255
				grayscaled_rgb = preprocess_input(grayscaled_rgb)

				lab_batch = rgb2lab(x_b)  # convert to LAB colorspace #usar o grayscaled_rgb
				X_batch = lab_batch[:, :, :, 0]  # extract L from LAB
				X_batch = X_batch.reshape(X_batch.shape + (1,))  # reshape into (batch, IMAGE_SIZE, IMAGE_SIZE, 1)
				X_batch = 2 * X_batch / 100 - 1.  # normalize the batch
				Y_batch = lab_batch[:, :, :, 1:] / 127  # extract AB from LAB

				new_X1[count,:] = X_batch
				new_X2[count,:] = grayscaled_rgb
				new_Y1[count,:] = Y_batch
				count+=1
			else:
				yield ([new_X1, new_X2], new_Y1)
				count = 0
				new_X1 = np.zeros((bc, 224, 224, 1))
				new_X2 = np.zeros((bc, 224, 224, 3))
				new_Y1 = np.zeros((bc, 224, 224, 2))

		if(np.count_nonzero(new_X1) != 0):
			yield ([new_X1, new_X2], new_Y1)

training_images = get_images(_TRAINING_IMAGES)

model = generate_mobilenet_model()
model.load_weights(_MODEL_WEIGHTS_NAME)

checkpoint = ModelCheckpoint(_MODEL_WEIGHTS_NAME, monitor='loss',
                             save_best_only=False, save_weights_only=True)

model.fit_generator(generator(training_images, _TRAINING_IMAGES, 32),
                    steps_per_epoch=len(training_images)/10,
                    epochs=100,
                    verbose=1,
                    callbacks=[checkpoint],
                    )

model.save(_MODEL_NAME)

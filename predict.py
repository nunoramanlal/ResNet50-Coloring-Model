from keras.preprocessing.image import img_to_array, load_img
from model_utils import generate_mobilenet_model
import os
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.io import imsave, imread
import numpy as np
from keras.utils.generic_utils import CustomObjectScope
from keras.models import load_model
import keras
from keras import backend as K
import skimage

_MODEL_WEIGHTS_NAME = 'weights_mobilenet_model_color.h5'
#_TARGET_PATH = 'colored_guardian_noFilter/'
#_IMAGE_PATH = 'Guardian/'
_IMAGE_PATH = '../old_resized_images_ny/'
_TARGET_PATH = 'news/'
# shows the minimum value of the AB channels
def y_true_min(yt, yp):
    return K.min(yt)

# shows the maximum value of the RGB AB channels
def y_true_max(yt, yp):
    return K.max(yt)

# shows the minimum value of the predicted AB channels
def y_pred_min(yt, yp):
    return K.min(yp)

# shows the maximum value of the predicted AB channels
def y_pred_max(yt, yp):
    return K.max(yp)

def postprocess_output(X_lab, y, img):
    '''
    This is a helper function for test time to convert and save the
    the processed image into the 'results' directory.

    Args:
        X_lab: L channel extracted from the grayscale image
        y: AB channels predicted by the colorizer network
        image_size: output image size
    '''
    y *= 127.  # scale the predictions to [-127, 127]
    X_lab = (X_lab + 1) * 50.  # scale the L channel to [0, 100]

    image_size = 224

    for i in range(len(y)):
        cur = np.zeros((image_size, image_size, 3))
        cur[:, :, 0] = X_lab[i, :, :, 0]
        cur[:, :, 1:] = y[i]
        imsave(_TARGET_PATH+img, lab2rgb(cur))

		
def sepia(image):
	
	# Convert image to float
	# (Avoids 8bit unsigned int problems)
	img = image.astype(float) / 256.0
	# Emulate Sepia filter by 'Dampening' colors
	# Example of the math is seen below:
	# R = .393*r + .769*g + .189&b
	# G = .349*r + .686*g + .168*b
	# B = .272*r + .534*g + .131*b
	sepia_filter = np.array([[.393, .769, .189], [.349, .686, .168], [.272, .534, .131]])
	scale = 0.01
	sepia_img = img.dot(sepia_filter.T * scale)
	sepia_img /= sepia_img.max()
	sepia_img = skimage.img_as_ubyte(sepia_img)
	return sepia_img 
	#variance_generator = lambda i,j,h: 0.25*(i+j)/1022. + 0.001
	#variances = np.fromfunction(variance_generator,(224,224,3))
	#sepia_img = skimage.util.random_noise(sepia_img, mode='gaussian', seed=1)
	#sepia_img = skimage.util.random_noise(sepia_img, mode='localvar', local_vars=variances)
	
	#return skimage.img_as_ubyte(skimage.color.rgb2gray(sepia_img))
	
	return sepia_img
	
model = generate_mobilenet_model()
model.load_weights(_MODEL_WEIGHTS_NAME)

def get_images_names(path):
    images = os.listdir(path)
    final = []
    for image in images:
        final.append(image.rstrip())
    return images

images_names = get_images_names(_IMAGE_PATH)

for img in images_names:
	x_b = img_to_array(load_img(_IMAGE_PATH+img, target_size= (224,224)))
	x_b = np.expand_dims(x_b, axis=0)
	x_b = x_b/255 #conversion to a range of -1 to 1. Explanation saved.

	grayscaled_rgb = gray2rgb(rgb2gray(x_b))  # convert to 3 channeled grayscale image

	X_lab = rgb2lab(grayscaled_rgb)[:, :, :, 0]
	X_lab = X_lab.reshape(X_lab.shape + (1,))
	X_lab = 2 * X_lab / 100 - 1.

	predictions = model.predict([X_lab,grayscaled_rgb], steps=1, verbose = 1)

	postprocess_output(X_lab , predictions, img)




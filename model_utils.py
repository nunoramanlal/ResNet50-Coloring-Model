from keras.layers import Conv2D, Input, Reshape, RepeatVector, concatenate, UpSampling2D, Flatten, Conv2DTranspose
from keras.models import Model
from keras.applications import mobilenet
from keras import backend as K
from keras.optimizers import Adam
import keras
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

def generate_mobilenet_model(img_size=224):
	'''
	Creates a Colorizer model. Note the difference from the report
	- https://github.com/baldassarreFe/deep-koalarization/blob/master/report.pdf

	I use a long skip connection network to speed up convergence and
	boost the output quality.
	'''
	# encoder model
	encoder_ip = Input(shape=(img_size, img_size, 1))
	encoder1 = Conv2D(64, (3, 3), padding='same', activation='relu', strides=(2, 2))(encoder_ip)
	encoder = Conv2D(128, (3, 3), padding='same', activation='relu')(encoder1)
	encoder2 = Conv2D(128, (3, 3), padding='same', activation='relu', strides=(2, 2))(encoder)
	encoder = Conv2D(256, (3, 3), padding='same', activation='relu')(encoder2)
	encoder = Conv2D(256, (3, 3), padding='same', activation='relu', strides=(2, 2))(encoder)
	encoder = Conv2D(512, (3, 3), padding='same', activation='relu')(encoder)
	encoder = Conv2D(512, (3, 3), padding='same', activation='relu')(encoder)
	encoder = Conv2D(256, (3, 3), padding='same', activation='relu')(encoder)

	# input fusion
	# Decide the image shape at runtime to allow prediction on
	# any size image, even if training is on 128x128
	batch, height, width, channels = K.int_shape(encoder)

	feature_extraction_model = keras.applications.resnet50.ResNet50(
										include_top=True, 
										weights='imagenet', 
										input_tensor=None, 
										input_shape=None, 
										pooling=None, 
										classes=1000)

	resnet_activations = Model(feature_extraction_model.input, feature_extraction_model.layers[-3].output)

	inp = Input(shape = (img_size,img_size,3))
	resnet_model_features = resnet_activations(inp)
	x = keras.layers.Conv2D(1000, (1, 1), padding='same', name='conv_preds')(resnet_model_features)
	a = Flatten()(x)

	fusion = RepeatVector(height * width)(a)
	fusion = Reshape((height, width, 1000))(fusion)
	fusion = concatenate([encoder, fusion], axis=-1)

	'''fusion = encoder'''
	fusion = Conv2D(256, (1, 1), padding='same', activation='relu')(fusion)

	# decoder model
	decoder = Conv2D(128, (3, 3), padding='same', activation='relu')(fusion)
	decoder = UpSampling2D()(decoder)
	decoder = concatenate([decoder, encoder2], axis=-1)
	decoder = Conv2D(64, (3, 3), padding='same', activation='relu')(decoder)
	decoder = Conv2D(64, (3, 3), padding='same', activation='relu')(decoder)
	decoder = UpSampling2D()(decoder)
	decoder = concatenate([decoder, encoder1], axis=-1)
	decoder = Conv2D(32, (3, 3), padding='same', activation='relu')(decoder)
	decoder = Conv2D(2, (3, 3), padding='same', activation='tanh')(decoder)
	decoder = UpSampling2D((2, 2))(decoder)

	model = Model([encoder_ip, inp], decoder, name='Colorizer')
	opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
	model.compile(optimizer=opt, loss='mse', metrics=[y_true_max,
														 y_true_min,
														 y_pred_max,
														 y_pred_min])

	print("Colorization model built and compiled")
	return model


if __name__ == '__main__':
    model = generate_mobilenet_model()
    model.summary()

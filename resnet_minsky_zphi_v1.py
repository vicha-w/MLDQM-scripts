import keras
from keras.layers import Input, Conv2D, ReLU, UpSampling2D, Add
from keras.models import Model
import uproot
import numpy as np
import sklearn.preprocessing
import skimage.measure
import pickle
import argparse
import os

VARIATION_NAME = "zphi1_resnet_v1"
os.system("mkdir {}".format(VARIATION_NAME))

TRAINING_EPOCHS = 50
ENCODER_FEATURES = 30

parser = argparse.ArgumentParser()
parser.add_argument("training_fraction", type=float)
args = parser.parse_args()

Bchain_zphi1 = []
Bchain_zphi1_runnr = []
Bchain_zphi1_lumisection = []
for i in range(1, 46):
    cache = uproot.open("/nfs/public/vwachira/Pixel2D_test/ZeroBias_2017B_DataFrame_2D_{}.root".format(i))["lumisections"]
    cache_isGoodLumi, cache_zphi1, cache_undead_count_zphi1, cache_runnr = cache.arrays(["isGoodLumi", "hist_zphi1", "undead_count_zphi1", "runnr"], outputtype=tuple)
    # We will now use Reference Run 297178 to train and test the AE, with 1379 lumisections present.
    #cache_zphi1 = cache_zphi1[(cache_isGoodLumi == 1) & (cache_undead_count_zphi1 >= 20000)]
    cache_zphi1 = cache_zphi1[cache_runnr==297178]
    if i == 1: Bchain_zphi1 = np.copy(cache_zphi1)
    else: Bchain_zphi1 = np.concatenate((Bchain_zphi1, cache_zphi1), axis=0)
Bchain_zphi1 = np.asarray(Bchain_zphi1)
print("Dataset has {} lumisections.".format(len(Bchain_zphi1)))
print()
print("TO START")
print("PRESS ENTER KEY")
input()

Bchain_zphi1_2D = np.reshape(Bchain_zphi1, (-1, 202, 302))[:, 1:201, 80:220]

#Bchain_zphi1_2D_count = np.sum(np.reshape(Bchain_zphi1_2D != 0, (-1, 200*140)), axis=1)
#Bchain_zphi1_2D = Bchain_zphi1_2D[np.flip(Bchain_zphi1_2D_count.argsort())]
np.random.shuffle(Bchain_zphi1_2D)

Bchain_zphi1_2D = np.reshape(Bchain_zphi1_2D, (-1, 200*140))
Bchain_zphi1_2D = sklearn.preprocessing.normalize(Bchain_zphi1_2D, norm="max")
Bchain_zphi1_2D = np.reshape(Bchain_zphi1_2D, (-1, 200, 140, 1))
#Bchain_zphi1_2D = skimage.measure.block_reduce(Bchain_zphi1_2D, block_size=(1, 5, 5, 1), func=np.mean)

#input_shape=(40, 28, 1)
input_shape=(200, 140, 1)

training_fraction = args.training_fraction
training_lumisections = int(len(Bchain_zphi1_2D)*training_fraction)
Bchain_zphi1_2D_train = Bchain_zphi1_2D[:training_lumisections]
Bchain_zphi1_2D_test  = Bchain_zphi1_2D[training_lumisections:]

def residual_block_enc(x, filter_number, kernel_size, strides, padding="valid"):
    y = Conv2D(filter_number, kernel_size, padding=padding, strides=strides)(x)
    y = ReLU()(y)
    y = Conv2D(filter_number, kernel_size, padding="same")(y)
    
    x = Conv2D(filter_number, kernel_size, padding=padding, strides=strides)(x)
    
    out = Add()([x, y])
    return out

def residual_block_dec(x, filter_number, kernel_size, upsampling_kernel):
    if upsampling_kernel > 1: 
        y = UpSampling2D(upsampling_kernel, interpolation="bilinear")(x)
        y = Conv2D(filter_number, kernel_size, padding="same")(y)
    else:
        y = Conv2D(filter_number, kernel_size, padding="same")(x)
    y = ReLU()(y)
    y = Conv2D(filter_number, kernel_size, padding="same")(y)
    
    x = UpSampling2D(upsampling_kernel, interpolation="bilinear")(x)
    x = Conv2D(filter_number, kernel_size, padding="same")(x)
    
    out = Add()([x, y])
    return out

def base_model():
    input_layer = Input(shape=input_shape)
    resnet_layer = residual_block_enc(input_layer, 60, (2, 2), 1, padding="same")
    resnet_layer = residual_block_enc(resnet_layer, 60, (2, 2), 2)
    resnet_layer = residual_block_enc(resnet_layer, 60, (2, 2), 1, padding="same")
    resnet_layer = residual_block_enc(resnet_layer, 60, (2, 2), 2)
    resnet_layer = residual_block_enc(resnet_layer, 1, (2, 2), 1, padding="same")

    resnet_layer = residual_block_dec(resnet_layer, 60, (2, 2), 1)
    resnet_layer = residual_block_dec(resnet_layer, 60, (2, 2), 2)
    resnet_layer = residual_block_dec(resnet_layer, 60, (2, 2), 1)
    resnet_layer = residual_block_dec(resnet_layer, 60, (2, 2), 2)
    resnet_layer = residual_block_dec(resnet_layer, 1, (2, 2), 1)
    
    resnet_model = Model(inputs = [input_layer], outputs = [resnet_layer])
    resnet_model.compile(loss = "mean_squared_error", optimizer="adadelta")
    return resnet_model

resnet_model = base_model()

print("Starting training. O(∩_∩)O")
history=resnet_model.fit(Bchain_zphi1_2D_train, Bchain_zphi1_2D_train, epochs=TRAINING_EPOCHS, verbose=True, validation_data=(Bchain_zphi1_2D_test, Bchain_zphi1_2D_test))
resnet_model.save("{1}/resnet_minsky_zphi1_{0:.2f}_v5.hdf5".format(args.training_fraction, VARIATION_NAME))
with open("{1}/resnet_minsky_zphi1_{0:.2f}_history_v5.p".format(args.training_fraction, VARIATION_NAME), "wb") as history_pickle:
    pickle.dump(history.history, history_pickle)

print("Evaluating. \(￣︶￣*\))")
train_predictions_cache = resnet_model.predict(Bchain_zphi1_2D_train)
resnet_train_losses = keras.losses.mse(keras.backend.constant(Bchain_zphi1_2D_train), keras.backend.constant(train_predictions_cache))
resnet_train_losses = keras.backend.eval(resnet_train_losses)
resnet_train_losses = np.mean(np.reshape(resnet_train_losses, (-1, 40*28)), axis=1) # WRONG

test_predictions_cache = resnet_model.predict(Bchain_zphi1_2D_test)
resnet_test_losses = keras.losses.mse(keras.backend.constant(Bchain_zphi1_2D_test), keras.backend.constant(test_predictions_cache))
resnet_test_losses = keras.backend.eval(resnet_test_losses)
resnet_test_losses = np.mean(np.reshape(resnet_test_losses, (-1, 40*28)), axis=1) # WRONG

with open("{1}/train_loss_{0:.2f}.p".format(args.training_fraction, VARIATION_NAME), "wb") as train_loss_pickle:
    pickle.dump(resnet_train_losses, train_loss_pickle)
with open("{1}/test_loss_{0:.2f}.p".format(args.training_fraction, VARIATION_NAME), "wb") as test_loss_pickle:
    pickle.dump(resnet_test_losses, test_loss_pickle)
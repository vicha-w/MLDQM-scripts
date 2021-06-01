import keras
from keras.layers import Input, Conv2D, ReLU, Conv2DTranspose, Dense, Add, Flatten, Reshape
from keras.models import Model
import uproot
import numpy as np
import sklearn.preprocessing
import pickle
import argparse
import os
import pandas as pd

VARIATION_NAME = "zphi1_resnet_v2"
os.system("mkdir {}".format(VARIATION_NAME))

TRAINING_EPOCHS = 50
ENCODER_FEATURES = 30

parser = argparse.ArgumentParser()
parser.add_argument("training_fraction", type=float)
args = parser.parse_args()

Bchain_zphi1_refrun = []

Bchain_zphi1_allgood = []
Bchain_zphi1_allgood_runnr = []
Bchain_zphi1_allgood_lumisection = []
Bchain_zphi1_allgood_fileloc = []

Bchain_zphi1_allbad = []
Bchain_zphi1_allbad_runnr = []
Bchain_zphi1_allbad_lumisection = []
Bchain_zphi1_allbad_fileloc = []

for i in range(1, 46):
    cache = uproot.open("/nfs/public/vwachira/Pixel2D_test/ZeroBias_2017B_DataFrame_2D_{}.root".format(i))["lumisections"]
    cache_isGoodLumi, cache_zphi1, cache_undead_count_zphi1, cache_runnr, cache_lumisection = cache.arrays(["isGoodLumi", "hist_zphi1", "undead_count_zphi1", "runnr", "lumisection"], outputtype=tuple)
    # We will now use Reference Run 297178 to train and test the AE, with 1379 lumisections present.
    #cache_zphi1 = cache_zphi1[(cache_isGoodLumi == 1) & (cache_undead_count_zphi1 >= 20000)]
    cache_zphi1_refrun = cache_zphi1[cache_runnr==297178]
    if i == 1: Bchain_zphi1_refrun = np.copy(cache_zphi1_refrun)
    else: Bchain_zphi1_refrun = np.concatenate((Bchain_zphi1_refrun, cache_zphi1_refrun), axis=0)

    cache_zphi1_allgood = cache_zphi1[(cache_isGoodLumi == 1)]
    cache_runnr_allgood = cache_runnr[(cache_isGoodLumi == 1)]
    cache_lumisection_allgood = cache_lumisection[(cache_isGoodLumi == 1)]
    cache_fileloc_allgood = np.array([i]*len(cache_runnr_allgood))
    if i== 1: 
        Bchain_zphi1_allgood = np.copy(cache_zphi1_allgood)
        Bchain_zphi1_allgood_runnr = np.copy(cache_runnr_allgood)
        Bchain_zphi1_allgood_lumisection = np.copy(cache_lumisection_allgood)
        Bchain_zphi1_allgood_fileloc = np.copy(cache_fileloc_allgood)
    else: 
        Bchain_zphi1_allgood = np.concatenate((Bchain_zphi1_allgood, cache_zphi1_allgood), axis=0)
        Bchain_zphi1_allgood_runnr = np.concatenate((Bchain_zphi1_allgood_runnr, cache_runnr_allgood), axis=0)
        Bchain_zphi1_allgood_lumisection = np.concatenate((Bchain_zphi1_allgood_lumisection, cache_lumisection_allgood), axis=0)
        Bchain_zphi1_allgood_fileloc = np.concatenate((Bchain_zphi1_allgood_fileloc, cache_fileloc_allgood), axis=0)

    cache_zphi1_allbad = cache_zphi1[(cache_isGoodLumi == 0)]
    cache_runnr_allbad = cache_runnr[(cache_isGoodLumi == 0)]
    cache_lumisection_allbad = cache_lumisection[(cache_isGoodLumi == 0)]
    cache_fileloc_allbad = np.array([i]*len(cache_runnr_allbad))
    if i== 1: 
        Bchain_zphi1_allbad = np.copy(cache_zphi1_allbad)
        Bchain_zphi1_allbad_runnr = np.copy(cache_runnr_allbad)
        Bchain_zphi1_allbad_lumisection = np.copy(cache_lumisection_allbad)
        Bchain_zphi1_allbad_fileloc = np.copy(cache_fileloc_allbad)
    else: 
        Bchain_zphi1_allbad = np.concatenate((Bchain_zphi1_allbad, cache_zphi1_allbad), axis=0)
        Bchain_zphi1_allbad_runnr = np.concatenate((Bchain_zphi1_allbad_runnr, cache_runnr_allbad), axis=0)
        Bchain_zphi1_allbad_lumisection = np.concatenate((Bchain_zphi1_allbad_lumisection, cache_lumisection_allbad), axis=0)
        Bchain_zphi1_allbad_fileloc = np.concatenate((Bchain_zphi1_allbad_fileloc, cache_fileloc_allbad), axis=0)

Bchain_zphi1_refrun = np.asarray(Bchain_zphi1_refrun)
Bchain_zphi1_allgood = np.asarray(Bchain_zphi1_allgood)
Bchain_zphi1_allbad = np.asarray(Bchain_zphi1_allbad)

#print("Dataset has {} lumisections.".format(len(Bchain_zphi1_refrun)))
#print()
#print("TO START")
#print("PRESS ENTER KEY")
#input()

Bchain_zphi1_2D = np.reshape(Bchain_zphi1_refrun, (-1, 202, 302))[:, 1:201, 80:220]

#Bchain_zphi1_2D_count = np.sum(np.reshape(Bchain_zphi1_2D != 0, (-1, 200*140)), axis=1)
#Bchain_zphi1_2D = Bchain_zphi1_2D[np.flip(Bchain_zphi1_2D_count.argsort())]
np.random.shuffle(Bchain_zphi1_2D)

Bchain_zphi1_2D = np.reshape(Bchain_zphi1_2D, (-1, 200*140))
Bchain_zphi1_2D = sklearn.preprocessing.normalize(Bchain_zphi1_2D, norm="max")
Bchain_zphi1_2D = np.reshape(Bchain_zphi1_2D, (-1, 200, 140, 1))
#Bchain_zphi1_2D = skimage.measure.block_reduce(Bchain_zphi1_2D, block_size=(1, 5, 5, 1), func=np.mean)

Bchain_zphi1_2D_allgood = np.reshape(Bchain_zphi1_allgood, (-1, 202, 302))[:, 1:201, 80:220]
Bchain_zphi1_2D_allgood = np.reshape(Bchain_zphi1_2D_allgood, (-1, 200*140))
Bchain_zphi1_2D_allgood = sklearn.preprocessing.normalize(Bchain_zphi1_2D_allgood, norm="max")
Bchain_zphi1_2D_allgood = np.reshape(Bchain_zphi1_2D_allgood, (-1, 200, 140, 1))

Bchain_zphi1_2D_allbad = np.reshape(Bchain_zphi1_allbad, (-1, 202, 302))[:, 1:201, 80:220]
Bchain_zphi1_2D_allbad = np.reshape(Bchain_zphi1_2D_allbad, (-1, 200*140))
Bchain_zphi1_2D_allbad = sklearn.preprocessing.normalize(Bchain_zphi1_2D_allbad, norm="max")
Bchain_zphi1_2D_allbad = np.reshape(Bchain_zphi1_2D_allbad, (-1, 200, 140, 1))

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

def residual_block_dec(x, filter_number, kernel_size, strides, padding="valid"):
    y = Conv2DTranspose(filter_number, kernel_size, strides=strides, padding=padding)(x)
    y = ReLU()(y)
    y = Conv2DTranspose(filter_number, kernel_size, padding="same")(y)
    
    x = Conv2DTranspose(filter_number, kernel_size, strides=strides, padding=padding)(x)
    
    out = Add()([x, y])
    return out

def residual_block_dense(x, outsize):
    y = Dense(outsize)(x)
    y = ReLU()(y)
    y = Dense(outsize)(y)
    
    x = Dense(outsize)(x)
    
    out = Add()([x, y])
    return out

def base_model():
    input_layer = Input(shape=input_shape)
    resnet_layer = residual_block_enc(input_layer, 128, 3, 2, padding="same")
    resnet_layer = residual_block_enc(resnet_layer, 128, 3, 1, padding="same")
    resnet_layer = residual_block_enc(resnet_layer, 128, 3, 1, padding="same")
    resnet_layer = residual_block_enc(resnet_layer, 128, 3, 2, padding="same")
    resnet_layer = residual_block_enc(resnet_layer, 128, 3, 1, padding="same")
    resnet_layer = residual_block_enc(resnet_layer, 1, 3, 1, padding="same")
    
    resnet_layer = residual_block_dec(resnet_layer, 128, 3, 1, padding="same")
    resnet_layer = residual_block_dec(resnet_layer, 128, 3, 1, padding="same")
    resnet_layer = residual_block_dec(resnet_layer, 128, 3, 2, padding="same")
    resnet_layer = residual_block_dec(resnet_layer, 128, 3, 1, padding="same")
    resnet_layer = residual_block_dec(resnet_layer, 128, 3, 1, padding="same")
    resnet_layer = residual_block_dec(resnet_layer, 1, 3, 2, padding="same")
    
    resnet_model = Model(inputs = [input_layer], outputs = [resnet_layer])
    resnet_model.compile(loss = "mse", optimizer="adadelta")
    return resnet_model

resnet_model = base_model()

print("Starting training. O(∩_∩)O")
history=resnet_model.fit(Bchain_zphi1_2D_train, Bchain_zphi1_2D_train, epochs=TRAINING_EPOCHS, verbose=True, validation_data=(Bchain_zphi1_2D_test, Bchain_zphi1_2D_test))
resnet_model.save("{1}/resnet_minsky_zphi1_{0:.2f}_v5.hdf5".format(args.training_fraction, VARIATION_NAME))
with open("{1}/resnet_minsky_zphi1_{0:.2f}_history_v5.p".format(args.training_fraction, VARIATION_NAME), "wb") as history_pickle:
    pickle.dump(history.history, history_pickle)

print("Evaluating. \(￣︶￣*\))")
train_predictions_cache = resnet_model.predict(Bchain_zphi1_2D_train)
resnet_train_losses = keras.losses.mse(Bchain_zphi1_2D_train, train_predictions_cache)
resnet_train_losses = np.mean(np.reshape(resnet_train_losses, (-1, 200*140)), axis=1)

test_predictions_cache = resnet_model.predict(Bchain_zphi1_2D_test)
resnet_test_losses = keras.losses.mse(Bchain_zphi1_2D_test, test_predictions_cache)
resnet_test_losses = np.mean(np.reshape(resnet_test_losses, (-1, 200*140)), axis=1)

with open("{1}/train_loss_{0:.2f}.p".format(args.training_fraction, VARIATION_NAME), "wb") as train_loss_pickle:
    pickle.dump(resnet_train_losses, train_loss_pickle)
with open("{1}/test_loss_{0:.2f}.p".format(args.training_fraction, VARIATION_NAME), "wb") as test_loss_pickle:
    pickle.dump(resnet_test_losses, test_loss_pickle)

del Bchain_zphi1_2D_train
del Bchain_zphi1_2D_test

good_predictions_cache = resnet_model.predict(Bchain_zphi1_2D_allgood)
resnet_good_losses = keras.losses.mse(Bchain_zphi1_2D_allgood, good_predictions_cache)
resnet_good_losses = np.mean(np.reshape(resnet_good_losses, (-1, 200*140)), axis=1)

del Bchain_zphi1_2D_allgood

bad_predictions_cache = resnet_model.predict(Bchain_zphi1_2D_allbad)
resnet_bad_losses = keras.losses.mse(Bchain_zphi1_2D_allbad, bad_predictions_cache)
resnet_bad_losses = np.mean(np.reshape(resnet_bad_losses, (-1, 200*140)), axis=1)

del Bchain_zphi1_2D_allbad

resnet_goodlumi_df = pd.DataFrame()
resnet_goodlumi_df["runnr"] = Bchain_zphi1_allgood_runnr
resnet_goodlumi_df["lumisection"] = Bchain_zphi1_allgood_lumisection
resnet_goodlumi_df["fileloc"] = Bchain_zphi1_allgood_fileloc
resnet_goodlumi_df["MSE"] = resnet_good_losses

resnet_badlumi_df = pd.DataFrame()
resnet_badlumi_df["runnr"] = Bchain_zphi1_allbad_runnr
resnet_badlumi_df["lumisection"] = Bchain_zphi1_allbad_lumisection
resnet_badlumi_df["fileloc"] = Bchain_zphi1_allbad_fileloc
resnet_badlumi_df["MSE"] = resnet_bad_losses

with open("{1}/good_loss_{0:.2f}.p".format(args.training_fraction, VARIATION_NAME), "wb") as good_loss_pickle:
    pickle.dump(resnet_goodlumi_df, good_loss_pickle)
with open("{1}/bad_loss_{0:.2f}.p".format(args.training_fraction, VARIATION_NAME), "wb") as bad_loss_pickle:
    pickle.dump(resnet_badlumi_df, bad_loss_pickle)
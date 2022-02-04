import keras
from keras.layers import Input, Conv2D, ReLU, Conv2DTranspose, Dense, Add, Reshape
from keras.layers import MaxPooling2D, UpSampling2D
from keras.models import Model
import tensorflow as tf
import uproot
import numpy as np
import pickle
import argparse
import os
import pandas as pd
import time

parser = argparse.ArgumentParser()
parser.add_argument("--training_fraction", type=float, default=0.5)
parser.add_argument("--histogram", default="zphi1")
#parser.add_argument("--middle_layer", type=int, default=1750)
args = parser.parse_args()

TRAINING_EPOCHS = 75
ENCODER_FEATURES = 30

USE_HISTOGRAM = args.histogram
VARIATION_NAME = f"{USE_HISTOGRAM}_resnet_v5"
os.system("mkdir {}".format(VARIATION_NAME))

Bchain_zphi_refrun = []
Bchain_zphi_refrun_rate = []

Bchain_zphi_all = []
Bchain_zphi_all_isgoodlumi = []
Bchain_zphi_all_runnr = []
Bchain_zphi_all_lumisection = []
Bchain_zphi_all_fileloc = []
Bchain_zphi_all_rate = []
Bchain_zphi_all_undeadcount = []
Bchain_zphi_all_tibtid = []

datasetrate_df = pd.read_json("datasetrate_2017B.json")

oms_era_df = pd.read_json("era_2017B.json")

training_time_dict = {}

start_time = time.time()
os.system("mkdir {}".format(VARIATION_NAME))
for i in range(1, 46):
    cache = uproot.open("/nfs/public/vwachira/Pixel2D_test/ZeroBias_2017B_DataFrame_2D_{}.root".format(i))["lumisections"]
    cache_isGoodLumi, cache_zphi, cache_runnr, cache_lumisection = cache.arrays(["isGoodLumi", f"hist_{USE_HISTOGRAM}", "runnr", "lumisection"], outputtype=tuple)
    cache_undead_zphi1, cache_undead_zphi2, cache_undead_zphi3, cache_undead_zphi4 = cache.arrays([f"undead_count_zphi{i}" for i in range(1, 5)], outputtype=tuple)
    [cache_undead_current] = cache.arrays([f"undead_count_{USE_HISTOGRAM}"], outputtype=tuple)

    cache_rate = []
    cache_tibtid = []
    for runnr, lumisection in zip(cache_runnr, cache_lumisection):
        rate_series = datasetrate_df[(datasetrate_df["run_number"] == runnr) & (datasetrate_df["first_lumisection_number"] == lumisection)]["rate"].to_list()
        if len(rate_series) != 1: cache_rate.append(np.nan)
        else: cache_rate.append(rate_series[0])
        oms_series = oms_era_df[(oms_era_df["run_number"]==runnr) & (oms_era_df["lumisection_number"]==lumisection)]["tibtid_ready"].to_list()
        if len(oms_series) != 1: cache_tibtid.append(False)
        else: cache_tibtid.append(oms_series[0])

    cache_rate = np.asarray(cache_rate)
    cache_tibtid = np.asarray(cache_tibtid)

    # Now we are removing isGoodLumi requirement (getting rid of human labels)
    # and active pixel count rate.

    #cache_zphi1_refrun = cache_zphi1[(cache_isGoodLumi == 1) & (~np.isnan(cache_rate)) & (cache_rate >= 75) & (cache_undead_count_zphi1 >= 20000)]
    #cache_zphi1_refrun_rate = cache_rate[(cache_isGoodLumi == 1) & (~np.isnan(cache_rate)) & (cache_rate >= 75) & (cache_undead_count_zphi1 >= 20000)]

    cache_zphi_refrun = cache_zphi[(~np.isnan(cache_rate)) & (cache_rate >= 75) & (cache_tibtid) & (cache_undead_zphi1 >= 24000) & (cache_undead_zphi2 >= 24000) & (cache_undead_zphi3 >= 24000) & (cache_undead_zphi4 >= 24000)]
    cache_zphi_refrun_rate = cache_rate[(~np.isnan(cache_rate)) & (cache_rate >= 75) & (cache_tibtid) & (cache_undead_zphi1 >= 24000) & (cache_undead_zphi2 >= 24000) & (cache_undead_zphi3 >= 24000) & (cache_undead_zphi4 >= 24000)]
    if i == 1: 
        Bchain_zphi_refrun = np.copy(cache_zphi_refrun)
        Bchain_zphi_refrun_rate = np.copy(cache_zphi_refrun_rate)
    else: 
        Bchain_zphi_refrun = np.concatenate((Bchain_zphi_refrun, cache_zphi_refrun), axis=0)
        Bchain_zphi_refrun_rate = np.concatenate((Bchain_zphi_refrun_rate, cache_zphi_refrun_rate), axis=0)
        
    if i == 1:
        Bchain_zphi_all = np.copy(cache_zphi)
        Bchain_zphi_all_isgoodlumi = np.copy(cache_isGoodLumi)
        Bchain_zphi_all_runnr = np.copy(cache_runnr)
        Bchain_zphi_all_lumisection = np.copy(cache_lumisection)
        Bchain_zphi_all_fileloc = np.array([i]*len(cache_runnr))
        Bchain_zphi_all_rate = np.copy(cache_rate)
        Bchain_zphi_all_undeadcount = np.copy(cache_undead_current)
        Bchain_zphi_all_tibtid = np.copy(cache_tibtid)
    else:
        Bchain_zphi_all = np.concatenate((Bchain_zphi_all, cache_zphi), axis=0)
        Bchain_zphi_all_isgoodlumi = np.concatenate((Bchain_zphi_all_isgoodlumi, cache_isGoodLumi), axis=0)
        Bchain_zphi_all_runnr = np.concatenate((Bchain_zphi_all_runnr, cache_runnr), axis=0)
        Bchain_zphi_all_lumisection = np.concatenate((Bchain_zphi_all_lumisection, cache_lumisection), axis=0)
        Bchain_zphi_all_fileloc = np.concatenate((Bchain_zphi_all_fileloc, np.array([i]*len(cache_runnr))), axis=0)
        Bchain_zphi_all_rate = np.concatenate((Bchain_zphi_all_rate, cache_rate), axis=0)
        Bchain_zphi_all_undeadcount = np.concatenate((Bchain_zphi_all_undeadcount, cache_undead_current), axis=0)
        Bchain_zphi_all_tibtid = np.concatenate((Bchain_zphi_all_tibtid, cache_tibtid), axis=0)

Bchain_zphi_refrun = np.asarray(Bchain_zphi_refrun)

#print("Dataset has {} lumisections.".format(len(Bchain_zphi_refrun)))
#print()
#print("TO START")
#print("PRESS ENTER KEY")
#input()

Bchain_zphi_2D = np.reshape(Bchain_zphi_refrun, (-1, 202, 302))[:, 1:201, 80:220]

#Bchain_zphi_2D_count = np.sum(np.reshape(Bchain_zphi_2D != 0, (-1, 200*140)), axis=1)
#Bchain_zphi_2D = Bchain_zphi_2D[np.flip(Bchain_zphi_2D_count.argsort())]

Bchain_zphi_2D = np.reshape(Bchain_zphi_2D, (-1, 200*140))
#Bchain_zphi_2D = sklearn.preprocessing.normalize(Bchain_zphi_2D, norm="max")
Bchain_zphi_2D = np.asarray([hist/rate for hist, rate in zip(Bchain_zphi_2D, Bchain_zphi_refrun_rate)])
Bchain_zphi_2D = np.reshape(Bchain_zphi_2D, (-1, 200, 140, 1))
#Bchain_zphi_2D = skimage.measure.block_reduce(Bchain_zphi_2D, block_size=(1, 5, 5, 1), func=np.mean)

Bchain_zphi_all = np.asarray(Bchain_zphi_all)
Bchain_zphi_2D_all = np.reshape(Bchain_zphi_all, (-1, 202, 302))[:, 1:201, 80:220]
Bchain_zphi_2D_all = np.reshape(Bchain_zphi_2D_all, (-1, 200*140))
Bchain_zphi_2D_all = np.asarray([hist/rate for hist, rate in zip(Bchain_zphi_2D_all, Bchain_zphi_all_rate)])
Bchain_zphi_2D_all = np.reshape(Bchain_zphi_2D_all, (-1, 200, 140, 1))

#input_shape=(40, 28, 1)
input_shape=(200, 140, 1)

np.random.shuffle(Bchain_zphi_2D)
training_fraction = args.training_fraction
training_lumisections = int(len(Bchain_zphi_2D)*training_fraction)
Bchain_zphi_2D_train = Bchain_zphi_2D[:training_lumisections]
Bchain_zphi_2D_test  = Bchain_zphi_2D[training_lumisections:]

using_regularizer = tf.keras.regularizers.l2(l=1e-4)

def residual_block_enc(x, filter_number, kernel_size, strides, padding="valid"):
    y = Conv2D(filter_number, kernel_size, padding=padding, strides=strides, kernel_regularizer=using_regularizer)(x)
    y = ReLU()(y)
    y = Conv2D(filter_number, kernel_size, padding="same", kernel_regularizer=using_regularizer)(y)
    
    x = Conv2D(filter_number, kernel_size, padding=padding, strides=strides, kernel_regularizer=using_regularizer)(x)
    
    out = Add()([x, y])
    return out

def residual_block_dec(x, filter_number, kernel_size, strides, padding="valid"):
    y = Conv2DTranspose(filter_number, kernel_size, strides=strides, padding=padding, kernel_regularizer=using_regularizer)(x)
    y = ReLU()(y)
    y = Conv2DTranspose(filter_number, kernel_size, padding="same", kernel_regularizer=using_regularizer)(y)
    
    x = Conv2DTranspose(filter_number, kernel_size, strides=strides, padding=padding, kernel_regularizer=using_regularizer)(x)
    
    out = Add()([x, y])
    return out

def base_model():
    input_layer = Input(shape=input_shape)
    resnet_layer = residual_block_enc(input_layer, 16, 3, 1, padding="same")
    resnet_layer = MaxPooling2D((2, 2))(resnet_layer)
    resnet_layer = residual_block_enc(resnet_layer, 32, 3, 1, padding="same")
    resnet_layer = MaxPooling2D((2, 2))(resnet_layer)
    resnet_layer = residual_block_enc(resnet_layer, 64, 3, 1, padding="same")
    resnet_layer = MaxPooling2D((5, 5))(resnet_layer)
    resnet_layer = residual_block_enc(resnet_layer, 128, 3, 1, padding="same")

    resnet_layer = Reshape((70*128,))(resnet_layer)
    resnet_layer = Dense(1000, activation="relu")(resnet_layer)
    resnet_layer = Dense(70*64, activation="relu")(resnet_layer)
    resnet_layer = Reshape((10, 7, 64))(resnet_layer)
    
    resnet_layer = residual_block_dec(resnet_layer, 64, 3, 1, padding="same")
    resnet_layer = UpSampling2D((5, 5))(resnet_layer)
    resnet_layer = residual_block_dec(resnet_layer, 32, 3, 1, padding="same")
    resnet_layer = UpSampling2D((2, 2))(resnet_layer)
    resnet_layer = residual_block_dec(resnet_layer, 16, 3, 1, padding="same")
    resnet_layer = UpSampling2D((2, 2))(resnet_layer)
    resnet_layer = residual_block_dec(resnet_layer, 1, 3, 1, padding="same")
    
    resnet_model = Model(inputs = [input_layer], outputs = [resnet_layer])
    resnet_model.compile(loss="mse", optimizer="adam")
    return resnet_model

resnet_model = base_model()

#variant_suffix = "{0:.2f}_{1}".format(args.training_fraction, args.middle_layer)
variant_suffix = "{0:.2f}".format(args.training_fraction)

print("Starting training. O(∩_∩)O")
callbacks = [
    keras.callbacks.EarlyStopping(monitor="loss", patience=3),
    #keras.callbacks.ModelCheckpoint(filepath="{1}/resnet_minsky_zphi_{0}_{{epoch:03d}}.hdf5".format(variant_suffix, VARIATION_NAME)),
    keras.callbacks.ModelCheckpoint(filepath="{1}/resnet_minsky_{2}_{0}_best.hdf5".format(variant_suffix, VARIATION_NAME, USE_HISTOGRAM), save_best_only=True)
]
history = resnet_model.fit(Bchain_zphi_2D_train, Bchain_zphi_2D_train, epochs=TRAINING_EPOCHS, verbose=True, validation_data=(Bchain_zphi_2D_test, Bchain_zphi_2D_test), batch_size=100, callbacks=callbacks)
#resnet_model.save("{1}/resnet_minsky_zphi_{0}.hdf5".format(variant_suffix, VARIATION_NAME))
with open("{1}/resnet_minsky_{2}_{0}_history.p".format(variant_suffix, VARIATION_NAME, USE_HISTOGRAM), "wb") as history_pickle:
    pickle.dump(history.history, history_pickle)

resnet_model = keras.models.load_model("{1}/resnet_minsky_{2}_{0}_best.hdf5".format(variant_suffix, VARIATION_NAME, USE_HISTOGRAM))

print("Evaluating. \(￣︶￣*\))")
train_predictions_cache = resnet_model.predict(Bchain_zphi_2D_train)
resnet_train_losses = keras.losses.mse(Bchain_zphi_2D_train, train_predictions_cache)
resnet_train_losses = np.mean(np.reshape(resnet_train_losses, (-1, 200*140)), axis=1)

test_predictions_cache = resnet_model.predict(Bchain_zphi_2D_test)
resnet_test_losses = keras.losses.mse(Bchain_zphi_2D_test, test_predictions_cache)
resnet_test_losses = np.mean(np.reshape(resnet_test_losses, (-1, 200*140)), axis=1)

with open("{1}/train_loss_{0}.p".format(variant_suffix, VARIATION_NAME), "wb") as train_loss_pickle:
    pickle.dump(resnet_train_losses, train_loss_pickle)
with open("{1}/test_loss_{0}.p".format(variant_suffix, VARIATION_NAME), "wb") as test_loss_pickle:
    pickle.dump(resnet_test_losses, test_loss_pickle)

del Bchain_zphi_2D_train
del Bchain_zphi_2D_test
del train_predictions_cache
del test_predictions_cache
del Bchain_zphi_all

all_predictions_cache = resnet_model.predict(Bchain_zphi_2D_all)
resnet_all_losses = keras.losses.mse(Bchain_zphi_2D_all, all_predictions_cache)
all_mean_MSE = np.mean(np.reshape(resnet_all_losses, (-1, 200*140)), axis=1)
all_max_MSE  = np.max(np.reshape(resnet_all_losses, (-1, 200*140)), axis=1)
all_std_MSE  = np.std(np.reshape(resnet_all_losses, (-1, 200*140)), axis=1)

resnet_alllumi_df = pd.DataFrame()
resnet_alllumi_df["runnr"] = Bchain_zphi_all_runnr
resnet_alllumi_df["lumisection"] = Bchain_zphi_all_lumisection
resnet_alllumi_df["fileloc"] = Bchain_zphi_all_fileloc
resnet_alllumi_df["isGoodLumi"] = Bchain_zphi_all_isgoodlumi
resnet_alllumi_df["rate"] = Bchain_zphi_all_rate
resnet_alllumi_df["undead_count"] = Bchain_zphi_all_undeadcount
resnet_alllumi_df["tibtid"] = Bchain_zphi_all_tibtid
resnet_alllumi_df["MSE_mean"] = all_mean_MSE
resnet_alllumi_df["MSE_max"] = all_max_MSE
resnet_alllumi_df["MSE_std"] = all_std_MSE

with open("{1}/all_loss_{0}.p".format(variant_suffix, VARIATION_NAME), "wb") as all_loss_pickle:
    pickle.dump(resnet_alllumi_df, all_loss_pickle)

end_time = time.time()
training_time_dict[USE_HISTOGRAM] = end_time - start_time
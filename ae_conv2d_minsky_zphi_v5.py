import uproot
import tensorflow as tf
import keras
from keras.layers import Input, Conv2D, UpSampling2D, Reshape, Dense, Dropout
from keras.models import Model
import numpy as np
import sklearn.preprocessing
import skimage.measure
import pickle
import argparse
import os

os.system("mkdir zphi1_v5")

TRAINING_EPOCHS = 500
ENCODER_FEATURES = 30

parser = argparse.ArgumentParser()
parser.add_argument("training_fraction", type=float)
args = parser.parse_args()

def msetop10(y_true, y_pred):
    top_values, _ = tf.nn.top_k(keras.backend.flatten(keras.backend.square(y_pred - y_true)), k=4*28, sorted=True)
    mean=keras.backend.mean(top_values, axis=-1)
    return mean

#Bchain = pyr.TChain("lumisections")
#for i in range(1, 46):
#    Bchain.Add("$SHARE_DIR/vwachira/Pixel2D_test/ZeroBias_2017B_DataFrame_2D_{}.root".format(i))
#
#Bchain_xyp1 = root_numpy.tree2array(Bchain, "hist_xyp1", "isGoodLumi==1")
#Bchain_xyp1_all = root_numpy.tree2array(Bchain, "hist_xyp1")

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
Bchain_zphi1_2D = skimage.measure.block_reduce(Bchain_zphi1_2D, block_size=(1, 5, 5, 1), func=np.mean)

input_shape = (40, 28, 1)

training_fraction = args.training_fraction
training_lumisections = int(len(Bchain_zphi1_2D)*training_fraction)
Bchain_zphi1_2D_train = Bchain_zphi1_2D[:training_lumisections]
Bchain_zphi1_2D_test  = Bchain_zphi1_2D[training_lumisections:]

def base_model(activation_func, use_top10 = False):
    input_layer = Input(shape=input_shape)
    ae_model_layer = Conv2D(60, (4, 4), activation=activation_func, padding="valid", strides=4)(input_layer)
    ae_model_layer = Dropout(0.5)(ae_model_layer)
    ae_model_layer = Conv2D(60, (5, 1), activation=activation_func, padding="valid", strides=(5, 1))(ae_model_layer)

    ae_model_layer = Reshape((840, ))(ae_model_layer)
    ae_model_layer = Dropout(0.5)(ae_model_layer)
    ae_model_layer = Dense(ENCODER_FEATURES, activation=activation_func)(ae_model_layer)
    ae_model_layer = Dropout(0.5)(ae_model_layer)
    ae_model_layer = Dense(840, activation=activation_func)(ae_model_layer)
    ae_model_layer = Reshape((2, 7, 60))(ae_model_layer)

    ae_model_layer = UpSampling2D((5, 1), interpolation="bilinear")(ae_model_layer)
    ae_model_layer = Dropout(0.5)(ae_model_layer)
    ae_model_layer = Conv2D(60, (3, 3), activation=activation_func, padding="same")(ae_model_layer)
    ae_model_layer = Dropout(0.5)(ae_model_layer)
    ae_model_layer = UpSampling2D((4, 4), interpolation="bilinear")(ae_model_layer)
    ae_model_layer = Conv2D(1, (4, 4), activation=activation_func, padding="same")(ae_model_layer)
    ae_model = Model(inputs = [input_layer], outputs = [ae_model_layer])
    ae_model.compile(loss = (msetop10 if use_top10 else "mse"), optimizer="adadelta")
    return ae_model

ae_models = {}
model_names = ["sigmoid", "relu", "sigmoid_top10", "relu_top10"]

ae_models["sigmoid"] = base_model("sigmoid", use_top10=False)
ae_models["relu"] = base_model("relu", use_top10=False)
ae_models["sigmoid_top10"] = base_model("sigmoid", use_top10=True)
ae_models["relu_top10"] = base_model("relu", use_top10=True)

history = {}
ae_train_losses = {}
ae_test_losses = {}
for model_name in model_names:
    print("Starting {} training. O(∩_∩)O".format(model_name))
    history[model_name] = ae_models[model_name].fit(Bchain_zphi1_2D_train, Bchain_zphi1_2D_train, epochs=TRAINING_EPOCHS, verbose=True, validation_data=(Bchain_zphi1_2D_test, Bchain_zphi1_2D_test))
    ae_models[model_name].save("zphi1_v5/ae_2dmodel_minsky_zphi1_{}_{:.2f}_v5.hdf5".format(model_name, args.training_fraction))
    with open("zphi1_v5/ae_2dmodel_minsky_zphi1_{}_{:.2f}_history_v5.p".format(model_name, args.training_fraction), "wb") as history_pickle:
        pickle.dump(history[model_name].history, history_pickle)

    print("Evaluating {}. \(￣︶￣*\))".format(model_name))
    train_predictions_cache = ae_models[model_name].predict(Bchain_zphi1_2D_train)
    train_losses_cache = keras.losses.mse(keras.backend.constant(Bchain_zphi1_2D_train), keras.backend.constant(train_predictions_cache))
    train_losses_cache = keras.backend.eval(train_losses_cache)
    train_losses_cache = np.mean(np.reshape(train_losses_cache, (-1, 40*28)), axis=1)
    ae_train_losses[model_name] = train_losses_cache
    
    test_predictions_cache = ae_models[model_name].predict(Bchain_zphi1_2D_test)
    test_losses_cache = keras.losses.mse(keras.backend.constant(Bchain_zphi1_2D_test), keras.backend.constant(test_predictions_cache))
    test_losses_cache = keras.backend.eval(test_losses_cache)
    test_losses_cache = np.mean(np.reshape(test_losses_cache, (-1, 40*28)), axis=1)
    ae_test_losses[model_name] = test_losses_cache

    with open("zphi1_v5/train_loss_{}_{:.2f}.p".format(model_name, args.training_fraction), "wb") as train_loss_pickle:
        pickle.dump(ae_train_losses[model_name], train_loss_pickle)
    with open("zphi1_v5/test_loss_{}_{:.2f}.p".format(model_name, args.training_fraction), "wb") as test_loss_pickle:
        pickle.dump(ae_test_losses[model_name], test_loss_pickle)
    
    print("Extracting {}. (*￣3￣)╭".format(model_name))
    extractor = keras.Model(inputs=ae_models[model_name].inputs, outputs=[ae_models[model_name].layers[4].output])
    encoded_train = np.reshape(extractor.predict(Bchain_zphi1_2D_train), (-1, ENCODER_FEATURES))
    encoded_test = np.reshape(extractor.predict(Bchain_zphi1_2D_test), (-1, ENCODER_FEATURES))
    
    with open("zphi1_v5/train_encoded_{}_{:.2f}.p".format(model_name, args.training_fraction), "wb") as train_encoded_pickle:
        pickle.dump(encoded_train, train_encoded_pickle)
    with open("zphi1_v5/test_encoded_{}_{:.2f}.p".format(model_name, args.training_fraction), "wb") as test_encoded_pickle:
        pickle.dump(encoded_test, test_encoded_pickle)

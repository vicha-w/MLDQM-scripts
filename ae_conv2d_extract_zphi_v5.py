import uproot
import keras
import numpy as np
import sklearn.preprocessing
import skimage.measure
import pickle
import argparse
import os
import pandas as pd

os.system("mkdir zphi1_v5")

ENCODER_FEATURES = 30

parser = argparse.ArgumentParser()
parser.add_argument("training_fraction", type=float)
args = parser.parse_args()

#Bchain = pyr.TChain("lumisections")
#for i in range(1, 46):
#    Bchain.Add("$SHARE_DIR/vwachira/Pixel2D_test/ZeroBias_2017B_DataFrame_2D_{}.root".format(i))
#
#Bchain_xyp1 = root_numpy.tree2array(Bchain, "hist_xyp1", "isGoodLumi==1")
#Bchain_xyp1_all = root_numpy.tree2array(Bchain, "hist_xyp1")

Bchain_zphi1 = []
Bchain_zphi1_runnr = []
Bchain_zphi1_lumisection = []
Bchain_zphi1_isGoodLumi = []
for i in range(1, 46):
    cache = uproot.open("/nfs/public/vwachira/Pixel2D_test/ZeroBias_2017B_DataFrame_2D_{}.root".format(i))["lumisections"]
    cache_isGoodLumi, cache_runnr, cache_lumisection, cache_zphi1, cache_undead_count_zphi1 = cache.arrays(["isGoodLumi", "runnr", "lumisection", "hist_zphi1", "undead_count_zphi1"], outputtype=tuple)
    #cache_zphi1 = cache_zphi1[cache_isGoodLumi == 1]
    #cache_runnr = cache_runnr[cache_isGoodLumi == 1]
    #cache_lumisection = cache_lumisection[cache_isGoodLumi == 1]
    #cache_zphi1 = cache_zphi1[cache_undead_count_zphi1 >= 20000]
    #cache_runnr = cache_runnr[cache_undead_count_zphi1 >= 20000]
    #cache_lumisection = cache_lumisection[cache_undead_count_zphi1 >= 20000]
    #cache_isGoodLumi = cache_isGoodLumi[cache_undead_count_zphi1 >= 20000]
    if i == 1: 
        Bchain_zphi1 = np.copy(cache_zphi1)
        Bchain_zphi1_runnr = np.copy(cache_runnr)
        Bchain_zphi1_lumisection = np.copy(cache_lumisection)
        Bchain_zphi1_isGoodLumi = np.copy(cache_isGoodLumi)
    else: 
        Bchain_zphi1 = np.concatenate((Bchain_zphi1, cache_zphi1), axis=0)
        Bchain_zphi1_runnr = np.concatenate((Bchain_zphi1_runnr, cache_runnr), axis=0)
        Bchain_zphi1_lumisection = np.concatenate((Bchain_zphi1_lumisection, cache_lumisection), axis=0)
        Bchain_zphi1_isGoodLumi = np.concatenate((Bchain_zphi1_isGoodLumi, cache_isGoodLumi), axis=0)

Bchain_zphi1 = np.asarray(Bchain_zphi1)
Bchain_zphi1_runnr = np.asarray(Bchain_zphi1_runnr)
Bchain_zphi1_lumisection = np.asarray(Bchain_zphi1_lumisection)
Bchain_zphi1_isGoodLumi = np.asarray(Bchain_zphi1_isGoodLumi)

Bchain_zphi1_2D = np.reshape(Bchain_zphi1, (-1, 202, 302))[:, 1:201, 80:220]
Bchain_zphi1_pixelcount = np.sum(np.reshape(Bchain_zphi1_2D != 0, (-1, 200*140)), axis=1)
Bchain_zphi1_2D = np.reshape(Bchain_zphi1_2D, (-1, 200*140))
Bchain_zphi1_2D = sklearn.preprocessing.normalize(Bchain_zphi1_2D, norm="max")
Bchain_zphi1_2D = np.reshape(Bchain_zphi1_2D, (-1, 200, 140, 1))
Bchain_zphi1_2D = skimage.measure.block_reduce(Bchain_zphi1_2D, block_size=(1, 5, 5, 1), func=np.mean)

input_shape = (40, 28, 1)

ae_models = {}
model_names = ["sigmoid", "relu", "sigmoid_top10", "relu_top10"]

MSE_loss_df = pd.DataFrame()
MSE_loss_df["runnr"] = Bchain_zphi1_runnr
MSE_loss_df["lumisection"] = Bchain_zphi1_lumisection
MSE_loss_df["pixelcount"] = Bchain_zphi1_pixelcount
MSE_loss_df["isGoodLumi"] = Bchain_zphi1_isGoodLumi

MSE_loss_df["input_hist"] = [str(line) for line in np.reshape(Bchain_zphi1_2D, (-1, 40, 28))]

for model_name in model_names:
    ae_models[model_name] = keras.models.load_model("zphi1_v5/ae_2dmodel_minsky_zphi1_{}_{:.2f}_v5.hdf5".format(model_name, args.training_fraction), compile=False)

    print("Extracting {}. (*￣3￣)╭".format(model_name))
    extractor = keras.Model(inputs=ae_models[model_name].inputs, outputs=[ae_models[model_name].layers[6].output])
    encoded_all = np.reshape(extractor.predict(Bchain_zphi1_2D), (-1, ENCODER_FEATURES))
    print(Bchain_zphi1_isGoodLumi.shape)
    print(encoded_all.shape)
    MSE_loss_df["encoded_vector_{}".format(model_name)] = [str(line) for line in encoded_all]

    output_hist_all = np.reshape(ae_models[model_name].predict(Bchain_zphi1_2D), (-1, 40, 28))
    MSE_loss_df["output_{}".format(model_name)] = [str(line) for line in output_hist_all]

with open("zphi1_v5/encoded_{:.2f}_df.p".format(args.training_fraction), "wb") as dataframe_pickle:
    pickle.dump(MSE_loss_df, dataframe_pickle)
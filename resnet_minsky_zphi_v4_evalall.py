import keras
import uproot
import numpy as np
import pickle
import argparse
import os
import pandas as pd

VARIATION_NAME = "zphi_resnet_v4"
os.system("mkdir {}".format(VARIATION_NAME))

parser = argparse.ArgumentParser()
parser.add_argument("--training_fraction", type=float, default=0.5)
#parser.add_argument("--middle_layer", type=int, default=1750)
args = parser.parse_args()

Bchain_zphi1_all = []
Bchain_zphi1_all_undeadcount = []
Bchain_zphi2_all = []
Bchain_zphi2_all_undeadcount = []
Bchain_zphi3_all = []
Bchain_zphi3_all_undeadcount = []
Bchain_zphi4_all = []
Bchain_zphi4_all_undeadcount = []

Bchain_zphi_all_isgoodlumi = []
Bchain_zphi_all_runnr = []
Bchain_zphi_all_lumisection = []
Bchain_zphi_all_fileloc = []
Bchain_zphi_all_rate = []

datasetrate_df = pd.read_json("datasetrate_2017B.json")

for i in range(1, 46):
    cache = uproot.open("/nfs/public/vwachira/Pixel2D_test/ZeroBias_2017B_DataFrame_2D_{}.root".format(i))["lumisections"]
    cache_isGoodLumi, cache_zphi1, cache_zphi2, cache_zphi3, cache_zphi4, cache_undead_count_zphi1, cache_undead_count_zphi2, cache_undead_count_zphi3, cache_undead_count_zphi4, cache_runnr, cache_lumisection = cache.arrays(["isGoodLumi", "hist_zphi1", "hist_zphi2", "hist_zphi3", "hist_zphi4", "undead_count_zphi1", "undead_count_zphi2", "undead_count_zphi3", "undead_count_zphi4", "runnr", "lumisection"], outputtype=tuple)

    cache_rate = []
    for runnr, lumisection in zip(cache_runnr, cache_lumisection):
        rate_series = datasetrate_df[(datasetrate_df["run_number"] == runnr) & (datasetrate_df["first_lumisection_number"] == lumisection)]["rate"].to_list()
        if len(rate_series) != 1: cache_rate.append(np.nan)
        else: cache_rate.append(rate_series[0])
    cache_rate = np.asarray(cache_rate)

    if i == 1:
        Bchain_zphi1_all = np.copy(cache_zphi1)
        Bchain_zphi1_all_undeadcount = np.copy(cache_undead_count_zphi1)
        Bchain_zphi2_all = np.copy(cache_zphi2)
        Bchain_zphi2_all_undeadcount = np.copy(cache_undead_count_zphi2)
        Bchain_zphi3_all = np.copy(cache_zphi3)
        Bchain_zphi3_all_undeadcount = np.copy(cache_undead_count_zphi3)
        Bchain_zphi4_all = np.copy(cache_zphi4)
        Bchain_zphi4_all_undeadcount = np.copy(cache_undead_count_zphi4)

        Bchain_zphi_all_isgoodlumi = np.copy(cache_isGoodLumi)
        Bchain_zphi_all_runnr = np.copy(cache_runnr)
        Bchain_zphi_all_lumisection = np.copy(cache_lumisection)
        Bchain_zphi_all_fileloc = np.array([i]*len(cache_runnr))
        Bchain_zphi_all_rate = np.copy(cache_rate)
    else:
        Bchain_zphi1_all = np.concatenate((Bchain_zphi1_all, cache_zphi1), axis=0)
        Bchain_zphi1_all_undeadcount = np.concatenate((Bchain_zphi1_all_undeadcount, cache_undead_count_zphi1), axis=0)
        Bchain_zphi2_all = np.concatenate((Bchain_zphi2_all, cache_zphi2), axis=0)
        Bchain_zphi2_all_undeadcount = np.concatenate((Bchain_zphi2_all_undeadcount, cache_undead_count_zphi2), axis=0)
        Bchain_zphi3_all = np.concatenate((Bchain_zphi3_all, cache_zphi3), axis=0)
        Bchain_zphi3_all_undeadcount = np.concatenate((Bchain_zphi3_all_undeadcount, cache_undead_count_zphi3), axis=0)
        Bchain_zphi4_all = np.concatenate((Bchain_zphi4_all, cache_zphi4), axis=0)
        Bchain_zphi4_all_undeadcount = np.concatenate((Bchain_zphi4_all_undeadcount, cache_undead_count_zphi4), axis=0)

        Bchain_zphi_all_isgoodlumi = np.concatenate((Bchain_zphi_all_isgoodlumi, cache_isGoodLumi), axis=0)
        Bchain_zphi_all_runnr = np.concatenate((Bchain_zphi_all_runnr, cache_runnr), axis=0)
        Bchain_zphi_all_lumisection = np.concatenate((Bchain_zphi_all_lumisection, cache_lumisection), axis=0)
        Bchain_zphi_all_fileloc = np.concatenate((Bchain_zphi_all_fileloc, np.array([i]*len(cache_runnr))), axis=0)
        Bchain_zphi_all_rate = np.concatenate((Bchain_zphi_all_rate, cache_rate), axis=0)

Bchain_zphi1_all = np.asarray(Bchain_zphi1_all)
Bchain_zphi2_all = np.asarray(Bchain_zphi2_all)
Bchain_zphi3_all = np.asarray(Bchain_zphi3_all)
Bchain_zphi4_all = np.asarray(Bchain_zphi4_all)

Bchain_zphi1_2D = np.reshape(Bchain_zphi1_all, (-1, 202, 302))[:, 1:201, 80:220]
Bchain_zphi1_2D = np.reshape(Bchain_zphi1_2D, (-1, 200*140))
Bchain_zphi1_2D = np.asarray([hist/rate for hist, rate in zip(Bchain_zphi1_2D, Bchain_zphi_all_rate)])
Bchain_zphi1_2D = np.reshape(Bchain_zphi1_2D, (-1, 200, 140, 1))

Bchain_zphi2_2D = np.reshape(Bchain_zphi2_all, (-1, 202, 302))[:, 1:201, 80:220]
Bchain_zphi2_2D = np.reshape(Bchain_zphi2_2D, (-1, 200*140))
Bchain_zphi2_2D = np.asarray([hist/rate for hist, rate in zip(Bchain_zphi2_2D, Bchain_zphi_all_rate)])
Bchain_zphi2_2D = np.reshape(Bchain_zphi2_2D, (-1, 200, 140, 1))

Bchain_zphi3_2D = np.reshape(Bchain_zphi3_all, (-1, 202, 302))[:, 1:201, 80:220]
Bchain_zphi3_2D = np.reshape(Bchain_zphi3_2D, (-1, 200*140))
Bchain_zphi3_2D = np.asarray([hist/rate for hist, rate in zip(Bchain_zphi3_2D, Bchain_zphi_all_rate)])
Bchain_zphi3_2D = np.reshape(Bchain_zphi3_2D, (-1, 200, 140, 1))

Bchain_zphi4_2D = np.reshape(Bchain_zphi4_all, (-1, 202, 302))[:, 1:201, 80:220]
Bchain_zphi4_2D = np.reshape(Bchain_zphi4_2D, (-1, 200*140))
Bchain_zphi4_2D = np.asarray([hist/rate for hist, rate in zip(Bchain_zphi4_2D, Bchain_zphi_all_rate)])
Bchain_zphi4_2D = np.reshape(Bchain_zphi4_2D, (-1, 200, 140, 1))

variant_suffix = "{0:.2f}".format(args.training_fraction)
#model1_path = "{1}/resnet_minsky_zphi1_{0}_best.hdf5".format(variant_suffix, VARIATION_NAME)
model1_path = f"zphi1_resnet_v4/resnet_minsky_zphi1_{variant_suffix}_best.hdf5"
model2_path = f"zphi2_resnet_v4/resnet_minsky_zphi2_{variant_suffix}_best.hdf5"
model3_path = f"zphi3_resnet_v4/resnet_minsky_zphi3_{variant_suffix}_best.hdf5"
model4_path = f"zphi4_resnet_v4/resnet_minsky_zphi4_{variant_suffix}_best.hdf5"

input_shape=(200, 140, 1)
resnet_zphi1_model = keras.models.load_model(model1_path)
resnet_zphi2_model = keras.models.load_model(model2_path)
resnet_zphi3_model = keras.models.load_model(model3_path)
resnet_zphi4_model = keras.models.load_model(model4_path)
print(f"Model loaded: {model1_path}")
print(f"Model loaded: {model2_path}")
print(f"Model loaded: {model3_path}")
print(f"Model loaded: {model4_path}")

print("Evaluating. \(￣︶￣*\))")

UPPER_HIST_BOUND = 0.5
HIST_BINS = 50
raw_bins = np.linspace(0., 0.5, HIST_BINS+1)
use_bins = [raw_bins, np.array([np.inf])]
use_bins = np.concatenate(use_bins)

zphi1_predictions_cache = resnet_zphi1_model.predict(Bchain_zphi1_2D)
resnet_zphi1_losses = keras.losses.mse(Bchain_zphi1_2D, zphi1_predictions_cache)
zphi1_mean_MSE = np.mean(np.reshape(resnet_zphi1_losses, (-1, 200*140)), axis=1)
zphi1_max_MSE  = np.max(np.reshape(resnet_zphi1_losses, (-1, 200*140)), axis=1)
zphi1_std_MSE  = np.std(np.reshape(resnet_zphi1_losses, (-1, 200*140)), axis=1)
zphi1_hist_MSE = np.asarray([np.histogram(mse_plot, bins=use_bins)[0] \
                             for mse_plot in np.reshape(resnet_zphi1_losses, (-1, 200*140))])
del zphi1_predictions_cache
del resnet_zphi1_losses

zphi2_predictions_cache = resnet_zphi2_model.predict(Bchain_zphi2_2D)
resnet_zphi2_losses = keras.losses.mse(Bchain_zphi2_2D, zphi2_predictions_cache)
zphi2_mean_MSE = np.mean(np.reshape(resnet_zphi2_losses, (-1, 200*140)), axis=1)
zphi2_max_MSE  = np.max(np.reshape(resnet_zphi2_losses, (-1, 200*140)), axis=1)
zphi2_std_MSE  = np.std(np.reshape(resnet_zphi2_losses, (-1, 200*140)), axis=1)
zphi2_hist_MSE = np.asarray([np.histogram(mse_plot, bins=use_bins)[0] \
                             for mse_plot in np.reshape(resnet_zphi2_losses, (-1, 200*140))])
del zphi2_predictions_cache
del resnet_zphi2_losses

zphi3_predictions_cache = resnet_zphi3_model.predict(Bchain_zphi3_2D)
resnet_zphi3_losses = keras.losses.mse(Bchain_zphi3_2D, zphi3_predictions_cache)
zphi3_mean_MSE = np.mean(np.reshape(resnet_zphi3_losses, (-1, 200*140)), axis=1)
zphi3_max_MSE  = np.max(np.reshape(resnet_zphi3_losses, (-1, 200*140)), axis=1)
zphi3_std_MSE  = np.std(np.reshape(resnet_zphi3_losses, (-1, 200*140)), axis=1)
zphi3_hist_MSE = np.asarray([np.histogram(mse_plot, bins=use_bins)[0] \
                             for mse_plot in np.reshape(resnet_zphi3_losses, (-1, 200*140))])
del zphi3_predictions_cache
del resnet_zphi3_losses

zphi4_predictions_cache = resnet_zphi4_model.predict(Bchain_zphi4_2D)
resnet_zphi4_losses = keras.losses.mse(Bchain_zphi4_2D, zphi4_predictions_cache)
zphi4_mean_MSE = np.mean(np.reshape(resnet_zphi4_losses, (-1, 200*140)), axis=1)
zphi4_max_MSE  = np.max(np.reshape(resnet_zphi4_losses, (-1, 200*140)), axis=1)
zphi4_std_MSE  = np.std(np.reshape(resnet_zphi4_losses, (-1, 200*140)), axis=1)
zphi4_hist_MSE = np.asarray([np.histogram(mse_plot, bins=use_bins)[0] \
                             for mse_plot in np.reshape(resnet_zphi4_losses, (-1, 200*140))])
del zphi4_predictions_cache
del resnet_zphi4_losses

resnet_alllumi_df = pd.DataFrame()
resnet_alllumi_df["runnr"] = Bchain_zphi_all_runnr
resnet_alllumi_df["lumisection"] = Bchain_zphi_all_lumisection
resnet_alllumi_df["fileloc"] = Bchain_zphi_all_fileloc
resnet_alllumi_df["isGoodLumi"] = Bchain_zphi_all_isgoodlumi
resnet_alllumi_df["rate"] = Bchain_zphi_all_rate
resnet_alllumi_df["zphi1_undead_count"] = Bchain_zphi1_all_undeadcount
resnet_alllumi_df["zphi2_undead_count"] = Bchain_zphi2_all_undeadcount
resnet_alllumi_df["zphi3_undead_count"] = Bchain_zphi3_all_undeadcount
resnet_alllumi_df["zphi4_undead_count"] = Bchain_zphi4_all_undeadcount
resnet_alllumi_df["zphi1_MSE_mean"] = zphi1_mean_MSE
resnet_alllumi_df["zphi1_MSE_max"]  = zphi1_max_MSE
resnet_alllumi_df["zphi1_MSE_std"]  = zphi1_std_MSE
resnet_alllumi_df["zphi2_MSE_mean"] = zphi2_mean_MSE
resnet_alllumi_df["zphi2_MSE_max"]  = zphi2_max_MSE
resnet_alllumi_df["zphi2_MSE_std"]  = zphi2_std_MSE
resnet_alllumi_df["zphi3_MSE_mean"] = zphi3_mean_MSE
resnet_alllumi_df["zphi3_MSE_max"]  = zphi3_max_MSE
resnet_alllumi_df["zphi3_MSE_std"]  = zphi3_std_MSE
resnet_alllumi_df["zphi4_MSE_mean"] = zphi4_mean_MSE
resnet_alllumi_df["zphi4_MSE_max"]  = zphi4_max_MSE
resnet_alllumi_df["zphi4_MSE_std"]  = zphi4_std_MSE

for bin_number in range(HIST_BINS+1):
    resnet_alllumi_df[f"zphi1_MSE_bin{bin_number:02d}"] = zphi1_hist_MSE[:, bin_number]
for bin_number in range(HIST_BINS+1):
    resnet_alllumi_df[f"zphi2_MSE_bin{bin_number:02d}"] = zphi2_hist_MSE[:, bin_number]
for bin_number in range(HIST_BINS+1):
    resnet_alllumi_df[f"zphi3_MSE_bin{bin_number:02d}"] = zphi3_hist_MSE[:, bin_number]
for bin_number in range(HIST_BINS+1):
    resnet_alllumi_df[f"zphi4_MSE_bin{bin_number:02d}"] = zphi4_hist_MSE[:, bin_number]

#for anomaly_num in range(1, 40):
#    print(f"Calculating anomaly pattern {anomaly_num}")
#    Bchain_zphi1_2D_anomalous = Bchain_zphi1_2D.copy()
#    if anomaly_num <= 10: Bchain_zphi1_2D_anomalous[:, 20*(anomaly_num-1):20*anomaly_num, :, :] = 0
#    elif anomaly_num <= 24: Bchain_zphi1_2D_anomalous[:, :, 10*(anomaly_num-11):10*(anomaly_num-10), :] = 0
#    elif anomaly_num <= 28: Bchain_zphi1_2D_anomalous[:, 50*(anomaly_num-25):50*(anomaly_num-24), :, :] = 0
#    elif anomaly_num <= 39: Bchain_zphi1_2D_anomalous[:, :, 10*(anomaly_num-29):10*(anomaly_num-29)+40, :] = 0
#    
#    anomaly_predictions_cache = resnet_model.predict(Bchain_zphi1_2D_anomalous)
#    resnet_anomalous_losses = keras.losses.mse(Bchain_zphi1_2D_anomalous, anomaly_predictions_cache)
#    anomalous_mean_MSE = np.mean(np.reshape(resnet_anomalous_losses, (-1, 200*140)), axis=1)
#    anomalous_max_MSE  = np.max(np.reshape(resnet_anomalous_losses, (-1, 200*140)), axis=1)
#    anomalous_std_MSE  = np.std(np.reshape(resnet_anomalous_losses, (-1, 200*140)), axis=1)
#    resnet_alllumi_df[f"MSE_mean_anomaly{anomaly_num:02d}"] = anomalous_mean_MSE
#    resnet_alllumi_df[f"MSE_max_anomaly{anomaly_num:02d}"] = anomalous_max_MSE
#    resnet_alllumi_df[f"MSE_std_anomaly{anomaly_num:02d}"] = anomalous_std_MSE
#
#    del Bchain_zphi1_2D_anomalous
#    del anomaly_predictions_cache
#    del resnet_anomalous_losses

with open("zphiall_losses_{0}.p".format(variant_suffix, VARIATION_NAME), "wb") as all_loss_pickle:
    pickle.dump(resnet_alllumi_df, all_loss_pickle)
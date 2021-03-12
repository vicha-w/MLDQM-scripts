import ROOT as pyr
import argparse
import pandas as pd
import numpy as np
import re
from array import array

parser = argparse.ArgumentParser()
parser.add_argument("name")

args = parser.parse_args()
source_path = args.name
if "2017B" in source_path:
    dest_path = re.sub("/eos/project/c/cmsml4dc/ML_2020/UL2017_Data/DF2017B_2D_Complete/", "/eos/home-v/vwachira/Pixel2D/", source_path)
if "2017C" in source_path:
    dest_path = re.sub("/eos/project/c/cmsml4dc/ML_2020/UL2017_Data/DF2017C_2D_Complete/", "/eos/home-v/vwachira/Pixel2D/", source_path)
if "2017D" in source_path:
    dest_path = re.sub("/eos/project/c/cmsml4dc/ML_2020/UL2017_Data/DF2017D_2D_Complete/", "/eos/home-v/vwachira/Pixel2D/", source_path)
if "2017E" in source_path:
    dest_path = re.sub("/eos/project/c/cmsml4dc/ML_2020/UL2017_Data/DF2017E_2D_Complete/", "/eos/home-v/vwachira/Pixel2D/", source_path)
if "2017F" in source_path:
    dest_path = re.sub("/eos/project/c/cmsml4dc/ML_2020/UL2017_Data/DF2017F_2D_Complete/", "/eos/home-v/vwachira/Pixel2D/", source_path)
dest_path = re.sub(".csv", ".root", dest_path)

golden_retriever = open("/afs/cern.ch/user/v/vwachira/CMSSW_11_0_1/src/condensed_GOLDEN_2017.txt", "r")
bad_retriever    = open("/afs/cern.ch/user/v/vwachira/CMSSW_11_0_1/src/condensed_BAD_2017.txt", "r")
golden_lines = golden_retriever.readlines()
bad_lines = bad_retriever.readlines()

golden_lumi = []
bad_lumi = []
for line in golden_lines[1:]:
    golden_lumi.append((int(line.split()[0]), int(line.split()[1])))
for line in bad_lines[1:]:
    bad_lumi.append((int(line.split()[0]), int(line.split()[1])))

condensed_file = pyr.TFile(dest_path, "RECREATE")
lumisections_tree = pyr.TTree("lumisections", "Summary from each lumisection")

runnr = array('i', [0])
lumisection = array('i', [0])
isGoodLumi = array('i', [0])
undead_count_xyp1 = array('i', [0])
undead_count_xyp2 = array('i', [0])
undead_count_xyp3 = array('i', [0])
undead_count_xym1 = array('i', [0])
undead_count_xym2 = array('i', [0])
undead_count_xym3 = array('i', [0])
undead_count_zphi1 = array('i', [0])
undead_count_zphi2 = array('i', [0])
undead_count_zphi3 = array('i', [0])
undead_count_zphi4 = array('i', [0])
hist_xyp1  = array('H', [0]*202*202)
hist_xyp2  = array('H', [0]*202*202)
hist_xyp3  = array('H', [0]*202*202)
hist_xym1  = array('H', [0]*202*202)
hist_xym2  = array('H', [0]*202*202)
hist_xym3  = array('H', [0]*202*202)
hist_zphi1 = array('H', [0]*302*202)
hist_zphi2 = array('H', [0]*302*202)
hist_zphi3 = array('H', [0]*302*202)
hist_zphi4 = array('H', [0]*302*202)

raw_hitpoints_xyp1 = np.zeros((202, 202))
raw_hitpoints_xyp2 = np.zeros((202, 202))
raw_hitpoints_xyp3 = np.zeros((202, 202))
raw_hitpoints_xym1 = np.zeros((202, 202))
raw_hitpoints_xym2 = np.zeros((202, 202))
raw_hitpoints_xym3 = np.zeros((202, 202))

raw_hitpoints_zphi1 = np.zeros((302, 202))
raw_hitpoints_zphi2 = np.zeros((302, 202))
raw_hitpoints_zphi3 = np.zeros((302, 202))
raw_hitpoints_zphi4 = np.zeros((302, 202))

hist_hitpoints_xyp1 = pyr.TH2I("hist_hitpoints_xyp1", "Hit points", 202, 0, 202, 202, 0, 202)
hist_hitpoints_xyp2 = pyr.TH2I("hist_hitpoints_xyp2", "Hit points", 202, 0, 202, 202, 0, 202)
hist_hitpoints_xyp3 = pyr.TH2I("hist_hitpoints_xyp3", "Hit points", 202, 0, 202, 202, 0, 202)
hist_hitpoints_xym1 = pyr.TH2I("hist_hitpoints_xym1", "Hit points", 202, 0, 202, 202, 0, 202)
hist_hitpoints_xym2 = pyr.TH2I("hist_hitpoints_xym2", "Hit points", 202, 0, 202, 202, 0, 202)
hist_hitpoints_xym3 = pyr.TH2I("hist_hitpoints_xym3", "Hit points", 202, 0, 202, 202, 0, 202)

hist_hitpoints_zphi1 = pyr.TH2I("hist_hitpoints_zphi1", "Hit points", 302, 0, 302, 202, 0, 202)
hist_hitpoints_zphi2 = pyr.TH2I("hist_hitpoints_zphi2", "Hit points", 302, 0, 302, 202, 0, 202)
hist_hitpoints_zphi3 = pyr.TH2I("hist_hitpoints_zphi3", "Hit points", 302, 0, 302, 202, 0, 202)
hist_hitpoints_zphi4 = pyr.TH2I("hist_hitpoints_zphi4", "Hit points", 302, 0, 302, 202, 0, 202)

lumisections_tree.Branch("runnr", runnr, "runnr/I")
lumisections_tree.Branch("lumisection", lumisection, "lumisection/I")
lumisections_tree.Branch("isGoodLumi", isGoodLumi, "isGoodLumi/I")
lumisections_tree.Branch("undead_count_xyp1", undead_count_xyp1, "undead_count_xyp1/I")
lumisections_tree.Branch("undead_count_xyp2", undead_count_xyp2, "undead_count_xyp2/I")
lumisections_tree.Branch("undead_count_xyp3", undead_count_xyp3, "undead_count_xyp3/I")
lumisections_tree.Branch("undead_count_xym1", undead_count_xym1, "undead_count_xym1/I")
lumisections_tree.Branch("undead_count_xym2", undead_count_xym2, "undead_count_xym2/I")
lumisections_tree.Branch("undead_count_xym3", undead_count_xym3, "undead_count_xym3/I")
lumisections_tree.Branch("undead_count_zphi1", undead_count_zphi1, "undead_count_zphi1/I")
lumisections_tree.Branch("undead_count_zphi2", undead_count_zphi2, "undead_count_zphi2/I")
lumisections_tree.Branch("undead_count_zphi3", undead_count_zphi3, "undead_count_zphi3/I")
lumisections_tree.Branch("undead_count_zphi4", undead_count_zphi4, "undead_count_zphi4/I")
lumisections_tree.Branch("hist_xyp1", hist_xyp1, "hist_xyp1[40804]/s")
lumisections_tree.Branch("hist_xyp2", hist_xyp2, "hist_xyp2[40804]/s")
lumisections_tree.Branch("hist_xyp3", hist_xyp3, "hist_xyp3[40804]/s")
lumisections_tree.Branch("hist_xym1", hist_xym1, "hist_xym1[40804]/s")
lumisections_tree.Branch("hist_xym2", hist_xym2, "hist_xym2[40804]/s")
lumisections_tree.Branch("hist_xym3", hist_xym3, "hist_xym3[40804]/s")
lumisections_tree.Branch("hist_zphi1", hist_zphi1, "hist_zphi1[61004]/s")
lumisections_tree.Branch("hist_zphi2", hist_zphi2, "hist_zphi2[61004]/s")
lumisections_tree.Branch("hist_zphi3", hist_zphi3, "hist_zphi3[61004]/s")
lumisections_tree.Branch("hist_zphi4", hist_zphi4, "hist_zphi4[61004]/s")

data = pd.read_csv(args.name)

all_lumisections = []
for i, e in data[["fromrun", "fromlumi"]].drop_duplicates().iterrows():
    all_lumisections.append((int(e["fromrun"]), int(e["fromlumi"])))

for fromrun, fromlumi in all_lumisections:
    undead_count_xyp1[0] = 0
    undead_count_xyp2[0] = 0
    undead_count_xyp3[0] = 0
    undead_count_xym1[0] = 0
    undead_count_xym2[0] = 0
    undead_count_xym3[0] = 0
    undead_count_zphi1[0] = 0
    undead_count_zphi2[0] = 0
    undead_count_zphi3[0] = 0
    undead_count_zphi4[0] = 0
    for i in range(40804):
        hist_xyp1[i] = 0
        hist_xyp2[i] = 0
        hist_xyp3[i] = 0
        hist_xym1[i] = 0
        hist_xym2[i] = 0
        hist_xym3[i] = 0
    for i in range(61004):
        hist_zphi1[i] = 0
        hist_zphi2[i] = 0
        hist_zphi3[i] = 0
        hist_zphi4[i] = 0

    runnr[0] = fromrun
    lumisection[0] = fromlumi
    if (fromrun, fromlumi) in golden_lumi:
        isGoodLumi[0] = 1
    elif (fromrun, fromlumi) in bad_lumi:
        isGoodLumi[0] = 0
    else: isGoodLumi[0] = -1

    print("Processing Run: {} Lumi: {}".format(fromrun, fromlumi))
    print(runnr[0])
    print(lumisection[0])
    print(isGoodLumi[0])

    lumidata = data.loc[(data["fromrun"] == fromrun) & (data["fromlumi"] == fromlumi)]

    if len(lumidata.loc[lumidata["hname"] == "clusterposition_xy_ontrack_PXDisk_+1"]["histo"]) == 0:
        undead_count_xyp1[0] = -1
    elif len(lumidata.loc[lumidata["hname"] == "clusterposition_xy_ontrack_PXDisk_+1"]["histo"]) != 1:
        undead_count_xyp1[0] = -2
    elif len(lumidata.loc[lumidata["hname"] == "clusterposition_xy_ontrack_PXDisk_+1"]["histo"]) == 1:
        cache_line = lumidata.loc[lumidata["hname"] == "clusterposition_xy_ontrack_PXDisk_+1"]["histo"].iloc[0]
        for i, num in zip(range(40804), eval(cache_line)): hist_xyp1[i] = num
        cache_hist = np.split(np.asarray(eval(cache_line)), 202)
        undead_count_xyp1[0] = np.count_nonzero(cache_hist)
        #print(undead_count_xyp1[0])
        raw_hitpoints_xyp1 += cache_hist

    if len(lumidata.loc[lumidata["hname"] == "clusterposition_xy_ontrack_PXDisk_+2"]["histo"]) == 0:
        undead_count_xyp2[0] = -1
    elif len(lumidata.loc[lumidata["hname"] == "clusterposition_xy_ontrack_PXDisk_+2"]["histo"]) != 1:
        undead_count_xyp2[0] = -2
    elif len(lumidata.loc[lumidata["hname"] == "clusterposition_xy_ontrack_PXDisk_+2"]["histo"]) == 1:
        cache_line = lumidata.loc[lumidata["hname"] == "clusterposition_xy_ontrack_PXDisk_+2"]["histo"].iloc[0]
        for i, num in zip(range(40804), eval(cache_line)): hist_xyp2[i] = num
        cache_hist = np.split(np.asarray(eval(cache_line)), 202)
        undead_count_xyp2[0] = np.count_nonzero(cache_hist)
        #print(undead_count_xyp2[0])
        raw_hitpoints_xyp2 += cache_hist

    
    if len(lumidata.loc[lumidata["hname"] == "clusterposition_xy_ontrack_PXDisk_+3"]["histo"]) == 0:
        undead_count_xyp3[0] = -1
    elif len(lumidata.loc[lumidata["hname"] == "clusterposition_xy_ontrack_PXDisk_+3"]["histo"]) != 1:
        undead_count_xyp3[0] = -2
    elif len(lumidata.loc[lumidata["hname"] == "clusterposition_xy_ontrack_PXDisk_+3"]["histo"]) == 1:
        cache_line = lumidata.loc[lumidata["hname"] == "clusterposition_xy_ontrack_PXDisk_+3"]["histo"].iloc[0]
        for i, num in zip(range(40804), eval(cache_line)): hist_xyp3[i] = num
        cache_hist = np.split(np.asarray(eval(cache_line)), 202)
        undead_count_xyp3[0] = np.count_nonzero(cache_hist)
        print(undead_count_xyp3[0])
        raw_hitpoints_xyp3 += cache_hist

    
    if len(lumidata.loc[lumidata["hname"] == "clusterposition_xy_ontrack_PXDisk_-1"]["histo"]) == 0:
        undead_count_xym1[0] = -1
    elif len(lumidata.loc[lumidata["hname"] == "clusterposition_xy_ontrack_PXDisk_-1"]["histo"]) != 1:
        undead_count_xym1[0] = -2
    elif len(lumidata.loc[lumidata["hname"] == "clusterposition_xy_ontrack_PXDisk_-1"]["histo"]) == 1:
        cache_line = lumidata.loc[lumidata["hname"] == "clusterposition_xy_ontrack_PXDisk_-1"]["histo"].iloc[0]
        for i, num in zip(range(40804), eval(cache_line)): hist_xym1[i] = num
        cache_hist = np.split(np.asarray(eval(cache_line)), 202)
        undead_count_xym1[0] = np.count_nonzero(cache_hist)
        #print(undead_count_xym1[0])
        raw_hitpoints_xyp1 += cache_hist

    if len(lumidata.loc[lumidata["hname"] == "clusterposition_xy_ontrack_PXDisk_-2"]["histo"]) == 0:
        undead_count_xym2[0] = -1
    elif len(lumidata.loc[lumidata["hname"] == "clusterposition_xy_ontrack_PXDisk_-2"]["histo"]) != 1:
        undead_count_xym2[0] = -2
    elif len(lumidata.loc[lumidata["hname"] == "clusterposition_xy_ontrack_PXDisk_-2"]["histo"]) == 1:
        cache_line = lumidata.loc[lumidata["hname"] == "clusterposition_xy_ontrack_PXDisk_-2"]["histo"].iloc[0]
        for i, num in zip(range(40804), eval(cache_line)): hist_xym2[i] = num
        cache_hist = np.split(np.asarray(eval(cache_line)), 202)
        undead_count_xym2[0] = np.count_nonzero(cache_hist)
        #print(undead_count_xym2[0])
        raw_hitpoints_xyp2 += cache_hist

    
    if len(lumidata.loc[lumidata["hname"] == "clusterposition_xy_ontrack_PXDisk_-3"]["histo"]) == 0:
        undead_count_xym3[0] = -1
    elif len(lumidata.loc[lumidata["hname"] == "clusterposition_xy_ontrack_PXDisk_-3"]["histo"]) != 1:
        undead_count_xym3[0] = -2
    elif len(lumidata.loc[lumidata["hname"] == "clusterposition_xy_ontrack_PXDisk_-3"]["histo"]) == 1:
        cache_line = lumidata.loc[lumidata["hname"] == "clusterposition_xy_ontrack_PXDisk_-3"]["histo"].iloc[0]
        for i, num in zip(range(40804), eval(cache_line)): hist_xym3[i] = num
        cache_hist = np.split(np.asarray(eval(cache_line)), 202)
        undead_count_xym3[0] = np.count_nonzero(cache_hist)
        #print(undead_count_xym3[0])
        raw_hitpoints_xyp3 += cache_hist

    if len(lumidata.loc[lumidata["hname"] == "clusterposition_zphi_ontrack_PXLayer_1"]["histo"]) == 0:
        undead_count_zphi1[0] = -1
    elif len(lumidata.loc[lumidata["hname"] == "clusterposition_zphi_ontrack_PXLayer_1"]["histo"]) != 1:
        undead_count_zphi1[0] = -2
    elif len(lumidata.loc[lumidata["hname"] == "clusterposition_zphi_ontrack_PXLayer_1"]["histo"]) == 1:
        cache_line = lumidata.loc[lumidata["hname"] == "clusterposition_zphi_ontrack_PXLayer_1"]["histo"].iloc[0]
        for i, num in zip(range(61004), eval(cache_line)): hist_zphi1[i] = num
        cache_hist = np.split(np.asarray(eval(cache_line)), 302)
        undead_count_zphi1[0] = np.count_nonzero(cache_hist)
        raw_hitpoints_zphi1 += cache_hist
        
    if len(lumidata.loc[lumidata["hname"] == "clusterposition_zphi_ontrack_PXLayer_2"]["histo"]) == 0:
        undead_count_zphi2[0] = -1
    elif len(lumidata.loc[lumidata["hname"] == "clusterposition_zphi_ontrack_PXLayer_2"]["histo"]) != 1:
        undead_count_zphi2[0] = -2
    elif len(lumidata.loc[lumidata["hname"] == "clusterposition_zphi_ontrack_PXLayer_2"]["histo"]) == 1:
        cache_line = lumidata.loc[lumidata["hname"] == "clusterposition_zphi_ontrack_PXLayer_2"]["histo"].iloc[0]
        for i, num in zip(range(61004), eval(cache_line)): hist_zphi2[i] = num
        cache_hist = np.split(np.asarray(eval(cache_line)), 302)
        undead_count_zphi2[0] = np.count_nonzero(cache_hist)
        raw_hitpoints_zphi2 += cache_hist

    if len(lumidata.loc[lumidata["hname"] == "clusterposition_zphi_ontrack_PXLayer_3"]["histo"]) == 0:
        undead_count_zphi3[0] = -1
    elif len(lumidata.loc[lumidata["hname"] == "clusterposition_zphi_ontrack_PXLayer_3"]["histo"]) != 1:
        undead_count_zphi3[0] = -2
    elif len(lumidata.loc[lumidata["hname"] == "clusterposition_zphi_ontrack_PXLayer_3"]["histo"]) == 1:
        cache_line = lumidata.loc[lumidata["hname"] == "clusterposition_zphi_ontrack_PXLayer_3"]["histo"].iloc[0]
        for i, num in zip(range(61004), eval(cache_line)): hist_zphi3[i] = num
        cache_hist = np.split(np.asarray(eval(cache_line)), 302)
        undead_count_zphi3[0] = np.count_nonzero(cache_hist)
        raw_hitpoints_zphi3 += cache_hist
        
    if len(lumidata.loc[lumidata["hname"] == "clusterposition_zphi_ontrack_PXLayer_4"]["histo"]) == 0:
        undead_count_zphi4[0] = -1
    elif len(lumidata.loc[lumidata["hname"] == "clusterposition_zphi_ontrack_PXLayer_4"]["histo"]) != 1:
        undead_count_zphi4[0] = -2
    elif len(lumidata.loc[lumidata["hname"] == "clusterposition_zphi_ontrack_PXLayer_4"]["histo"]) == 1:
        cache_line = lumidata.loc[lumidata["hname"] == "clusterposition_zphi_ontrack_PXLayer_4"]["histo"].iloc[0]
        for i, num in zip(range(61004), eval(cache_line)): hist_zphi4[i] = num
        cache_hist = np.split(np.asarray(eval(cache_line)), 302)
        undead_count_zphi4[0] = np.count_nonzero(cache_hist)
        raw_hitpoints_zphi4 += cache_hist
    
    lumisections_tree.Fill()

for i in range(202):
    for j in range(202):
        hist_hitpoints_xyp1.SetBinContent(i+1, j+1, raw_hitpoints_xyp1[i][j])
        hist_hitpoints_xyp2.SetBinContent(i+1, j+1, raw_hitpoints_xyp2[i][j])
        hist_hitpoints_xyp3.SetBinContent(i+1, j+1, raw_hitpoints_xyp3[i][j])
        hist_hitpoints_xym1.SetBinContent(i+1, j+1, raw_hitpoints_xym1[i][j])
        hist_hitpoints_xym2.SetBinContent(i+1, j+1, raw_hitpoints_xym2[i][j])
        hist_hitpoints_xym3.SetBinContent(i+1, j+1, raw_hitpoints_xym3[i][j])

for i in range(302):
    for j in range(202):
        hist_hitpoints_zphi1.SetBinContent(i+1, j+1, raw_hitpoints_zphi1[i][j])
        hist_hitpoints_zphi2.SetBinContent(i+1, j+1, raw_hitpoints_zphi2[i][j])
        hist_hitpoints_zphi3.SetBinContent(i+1, j+1, raw_hitpoints_zphi3[i][j])
        hist_hitpoints_zphi4.SetBinContent(i+1, j+1, raw_hitpoints_zphi4[i][j])

condensed_file.cd()
lumisections_tree.Write()
hist_hitpoints_xyp1.Write()
hist_hitpoints_xyp2.Write()
hist_hitpoints_xyp3.Write()
hist_hitpoints_xym1.Write()
hist_hitpoints_xym2.Write()
hist_hitpoints_xym3.Write()
hist_hitpoints_zphi1.Write()
hist_hitpoints_zphi2.Write()
hist_hitpoints_zphi3.Write()
hist_hitpoints_zphi4.Write()
condensed_file.Close()

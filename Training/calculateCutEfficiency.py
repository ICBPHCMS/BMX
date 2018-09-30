import ROOT
import sys
import os
import time
import numpy
import math

def baseSelection(tree,isSignal):
    #at least one b hypothesis
    if (tree.nBToKmumu==0):
        return False
    
    #an additional tag muon
    if (tree.Muon_sel_index<0):
        return False
        
    if (tree.BToKmumu_sel_index<0):
        return False
    
    #signal defined to be fully matched to gen (no product missing after reconstruction)
    #if (isSignal and tree.BToKmumu_gen_index<0):
    #    return False
        
    if (tree.BToKmumu_kaon_pt[tree.BToKmumu_sel_index]<=1. or math.fabs(tree.BToKmumu_kaon_eta[tree.BToKmumu_sel_index])>=2.4):
        return False
        
    if (tree.BToKmumu_mu1_pt[tree.BToKmumu_sel_index]<=1. or math.fabs(tree.BToKmumu_mu1_pt[tree.BToKmumu_sel_index])>=2.4):
        return False
        
    if (tree.BToKmumu_mu2_pt[tree.BToKmumu_sel_index]<=1. or math.fabs(tree.BToKmumu_mu2_pt[tree.BToKmumu_sel_index])>=2.4):
        return False
        
    if (tree.BToKmumu_mu1_charge[tree.BToKmumu_sel_index]*tree.BToKmumu_mu2_charge[tree.BToKmumu_sel_index]>0):
        return False
        
    #veto B mass window for background to reject signal in data
    if (not isSignal and tree.BToKmumu_mass[tree.BToKmumu_sel_index]<5.6):
        return False
    
    return True
    
def signalSelection(tree,isSignal):
    if (tree.BToKmumu_CL_vtx[tree.BToKmumu_sel_index]<=0.1):
        return False
    if (tree.BToKmumu_cosAlpha[tree.BToKmumu_sel_index]<=0.999):
        return False
    if (tree.BToKmumu_Lxy[tree.BToKmumu_sel_index]<=6):
        return False
    if (tree.BToKmumu_pt[tree.BToKmumu_sel_index]<=10.):
        return False
    if (tree.BToKmumu_kaon_pt[tree.BToKmumu_sel_index]<=1.5):
        return False
        
    return True

def calculateEfficiency(chain,baseSelection,signalSelection,isSignal=False):

    nEvents = min(5000000,chain.GetEntries())
    
    nPassBaseSelection = 0
    nPassSignalSelection = 0
    
    for ientry in range(nEvents):
        if ientry%1000==0:
            print "processing ... %i/%i"%(ientry,nEvents)
        chain.GetEntry(ientry)
        if not baseSelection(chain,isSignal):
            continue
        nPassBaseSelection+=1
        if signalSelection(chain,isSignal):
            nPassSignalSelection+=1
    return nEvents,nPassBaseSelection,nPassSignalSelection
    
   
    
    
signalChain = ROOT.TChain("Events")
signalChain.Add("/vols/cms/tstreble/BPH/BToKmumu_ntuple/BToKmumu_18_08_14_new/*.root")

backgroundChain = ROOT.TChain("Events")
backgroundChain.Add("/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking1_2018A_18_08_14_new/BToKmumuNtuple*.root")
#backgroundChain.Add("/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking2_2018A_18_08_14_new/BToKmumuNtuple*.root")
#backgroundChain.Add("/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking3_2018A_18_08_14_new/BToKmumuNtuple*.root")
#backgroundChain.Add("/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking4_2018A_18_08_14_new/BToKmumuNtuple*.root")
#backgroundChain.Add("/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking5_2018A_18_08_14_new/BToKmumuNtuple*.root")
#backgroundChain.Add("/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking6_2018A_18_08_14_new/BToKmumuNtuple*.root")

#backgroundChain.Add("/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking1_2018B_18_08_14_new/BToKmumuNtuple*.root")
#backgroundChain.Add("/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking2_2018B_18_08_14_new/BToKmumuNtuple*.root")
#backgroundChain.Add("/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking3_2018B_18_08_14_new/BToKmumuNtuple*.root")
#backgroundChain.Add("/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking4_2018B_18_08_14_new/BToKmumuNtuple*.root")
#backgroundChain.Add("/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking5_2018B_18_08_14_new/BToKmumuNtuple*.root")
#backgroundChain.Add("/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking6_2018B_18_08_14_new/BToKmumuNtuple*.root")


nEvents,nPassBaseSelection,nPassSignalSelection = calculateEfficiency(
    signalChain,
    baseSelection=baseSelection,
    signalSelection=signalSelection,
    isSignal=True
)
print "signal",nEvents,nPassBaseSelection,nPassSignalSelection,100.*nPassBaseSelection/nEvents,"%",100.*nPassSignalSelection/nPassBaseSelection,"%"
'''
nEvents,nPassBaseSelection,nPassSignalSelection = calculateEfficiency(
    backgroundChain,
    baseSelection=baseSelection,
    signalSelection=signalSelection,
    isSignal=False
)
print "background",nEvents,nPassBaseSelection,nPassSignalSelection,100.*nPassBaseSelection/nEvents,"%",100.*nPassSignalSelection/nPassBaseSelection,"%"
'''



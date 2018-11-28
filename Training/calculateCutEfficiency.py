import ROOT
import sys
import os
import time
import numpy
import math
import re
import uproot

class Chain(object):
    def __init__(self,fileList):
        self._fileList = fileList
        self._nEvents = []
        self._sumEvents = 0
        self._currentFile = None
        self._currentTree = None
        self._currentEntry = None
        self._currentOffset = None
        self._buffer = {}
        for i,f in enumerate(self._fileList[:]):
            try:
                print i,'/',len(self._fileList),'...',f
                rootFile = uproot.open(f)
                tree = rootFile["Events"]
                nevents = len(tree)
                self._sumEvents += nevents
                self._nEvents.append(nevents)
            except:
                print "Error - cannot open file: ",f
                self._fileList.remove(f)
            
    def GetEntries(self):
        return self._sumEvents
            
    def GetEntry(self,i):
        if self._currentTree!=None and (i-self._currentOffset)<len(self._currentTree) and (i-self._currentOffset)>=0:
            self._currentEntry = i-self._currentOffset
        else:
            del self._currentTree
            self._currentTree = None
            del self._buffer
            self._buffer = {}
            s = 0
            i = i%self._sumEvents #loop
            for e in range(len(self._nEvents)):
                if s<=i and (s+self._nEvents[e])>i:
                    print "opening",self._fileList[e]
                    self._currentFile = uproot.open(self._fileList[e])
                    self._currentTree = self._currentFile["Events"]
                    self._currentOffset = s
                    self._currentEntry = i-self._currentOffset
                    break
                s+=self._nEvents[e]
                    
    def __getattr__(self,k):
        if not self._buffer.has_key(k):
            self._buffer[k] = self._currentTree[k].array()
            #print "loading branch ",k," with entries ",len(self._buffer[k])," and shape ",self._buffer[k][self._currentEntry].shape,", (tree len=",len(self._currentTree),")"
        if self._currentEntry>=len(self._buffer[k]):
            print "Error - buffer for branch '",k,"' only ",len(self._buffer[k])," but requested entry ",self._currentEntry," (tree len=",len(self._currentTree),")"
            return 0
        return self._buffer[k][self._currentEntry]



def combinationSelectionMu(tree,icombination):
        
    if (tree.BToKmumu_kaon_pt[icombination]<=1. or math.fabs(tree.BToKmumu_kaon_eta[icombination])>=2.5):
        return False
        
    if (tree.BToKmumu_mu1_pt[icombination]<=1. or math.fabs(tree.BToKmumu_mu1_eta[icombination])>=2.5):
        return False
        
    if (tree.BToKmumu_mu2_pt[icombination]<=1. or math.fabs(tree.BToKmumu_mu2_eta[icombination])>=2.5):
        return False
        
    if (tree.BToKmumu_mu1_charge[icombination]*tree.BToKmumu_mu2_charge[icombination]>0):
        return False
        
    if (tree.BToKmumu_CL_vtx[icombination]<0.001):
        return False
        
    return True
    
def combinationSelectionEle(tree,icombination):
        
    if (tree.BToKee_kaon_pt[icombination]<=1. or math.fabs(tree.BToKee_kaon_eta[icombination])>=2.5):
        return False
        
    if (tree.BToKee_ele1_pt[icombination]<=1. or math.fabs(tree.BToKee_ele1_eta[icombination])>=2.5):
        return False
        
    if (tree.BToKee_ele2_pt[icombination]<=1. or math.fabs(tree.BToKee_ele2_eta[icombination])>=2.5):
        return False
        
    if (tree.BToKee_ele1_charge[icombination]*tree.BToKee_ele2_charge[icombination]>0):
        return False
        
    if (tree.BToKee_CL_vtx[icombination]<0.001):
        return False
        
    return True

def baseSelectionMu(tree,isSignal):
    #at least one b hypothesis
    if (tree.nBToKmumu==0):
        return False
    
    #an additional tag muon
    if (tree.Muon_sel_index<0):
        return False
        
    if (tree.BToKmumu_sel_index<0):
        return False
    
    #signal defined to be fully matched to gen (no product missing after reconstruction)
    if (isSignal and tree.BToKmumu_gen_index<0):
        return False
        
    if (not combinationSelectionMu(tree,tree.BToKmumu_sel_index)):
        return False
        
    #veto B mass window for background to reject signal in data
    if (not isSignal and tree.BToKmumu_mass[tree.BToKmumu_sel_index]>4.5 and tree.BToKmumu_mass[tree.BToKmumu_sel_index]<5.6):
        return False
        
    if (not isSignal and tree.BToKmumu_mass[tree.BToKmumu_sel_index]<3):
        return False
        
    return True
    
    
def baseSelectionEle(tree,isSignal):
    #at least one b hypothesis
    if (tree.nBToKee==0):
        return False
    
    #an additional tag muon
    if (tree.Muon_sel_index<0):
        return False
        
    if (tree.BToKee_sel_index<0):
        return False
    
    #signal defined to be fully matched to gen (no product missing after reconstruction)
    if (isSignal and tree.BToKee_gen_index<0):
        return False
        
    if (not combinationSelectionEle(tree,tree.BToKee_sel_index)):
        return False
        
    #veto B mass window for background to reject signal in data
    if (not isSignal and tree.BToKee_mass[tree.BToKee_sel_index]>5. and tree.BToKee_mass[tree.BToKee_sel_index]<5.6):
        return False
        
    return True
    
def signalSelectionMu(tree,isSignal):
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
    
def signalSelectionEle(tree,isSignal):
    if (tree.BToKee_CL_vtx[tree.BToKee_sel_index]<=0.1):
        return False
    if (tree.BToKee_cosAlpha[tree.BToKee_sel_index]<=0.999):
        return False
    if (tree.BToKee_Lxy[tree.BToKee_sel_index]<=6):
        return False
    if (tree.BToKee_pt[tree.BToKee_sel_index]<=10.):
        return False
    if (tree.BToKee_kaon_pt[tree.BToKee_sel_index]<=1.5):
        return False
        
    return True
    

def calculateEfficiency(chain,baseSelection,signalSelection,combinationSelection,nCombinations,combinationCLVtx,genCombination,isSignal=False):

    nEvents = min(5000000,chain.GetEntries())
    
    nPassBaseSelection = 0
    nPassSignalSelection = 0
    
    if isSignal:
        combIndexDict = {}
    
    for ientry in range(nEvents):
        if ientry%1000==0:
            print "processing ... %i/%i"%(ientry,nEvents)
        chain.GetEntry(ientry)
        if not baseSelection(chain,isSignal):
            continue
        nPassBaseSelection+=1
        
        if isSignal:
            selectedCombIndicesCLvtxPairs = []
            
            for icomb in range(nCombinations(chain)):
                if (combinationSelection(chain,icomb)):
                    #if (chain.BToKmumu_CL_vtx[icomb]<0.001):
                    #    continue
                    selectedCombIndicesCLvtxPairs.append([icomb,combinationCLVtx(chain,icomb)])
            selectedCombIndicesCLvtxPairsSorted = sorted(selectedCombIndicesCLvtxPairs,key=lambda elem: -elem[1])
            selectedCombIndicesSorted = map(lambda x:x[0],selectedCombIndicesCLvtxPairsSorted)
            genIndex = -1
            for icombSorted,icombIndex in enumerate(selectedCombIndicesSorted):
                if icombIndex==int(genCombination(chain)):
                    genIndex = icombSorted
                    break
            if not combIndexDict.has_key(genIndex):
                combIndexDict[genIndex] = 0
            combIndexDict[genIndex]+=1
        
        if signalSelection(chain,isSignal):
            nPassSignalSelection+=1
            
    if isSignal:
        culum = 0
        s = sum(combIndexDict.values())
        for icomb in sorted(combIndexDict.keys()):
            if icomb==-1:
                continue
            culum+=combIndexDict[icomb]
            print icomb,1.*culum/s
    return nEvents,nPassBaseSelection,nPassSignalSelection
    
   
'''
backgroundFiles = []

for folder in [
    "/vols/cms/vc1116/BParking/ntuPROD/data_BToKmumuNtuple/PR27_BToKmumu/A1",
    "/vols/cms/vc1116/BParking/ntuPROD/data_BToKmumuNtuple/PR27_BToKmumu/A2",
    "/vols/cms/vc1116/BParking/ntuPROD/data_BToKmumuNtuple/PR27_BToKmumu/A3",
    "/vols/cms/vc1116/BParking/ntuPROD/data_BToKmumuNtuple/PR27_BToKmumu/A4",
    "/vols/cms/vc1116/BParking/ntuPROD/data_BToKmumuNtuple/PR27_BToKmumu/A5",
    "/vols/cms/vc1116/BParking/ntuPROD/data_BToKmumuNtuple/PR27_BToKmumu/A6",
    
    #"/vols/cms/vc1116/BParking/ntuPROD/data_BToKmumuNtuple/PR27_BToKmumu/B1",
    #"/vols/cms/vc1116/BParking/ntuPROD/data_BToKmumuNtuple/PR27_BToKmumu/B2",
    #"/vols/cms/vc1116/BParking/ntuPROD/data_BToKmumuNtuple/PR27_BToKmumu/B3",
    #"/vols/cms/vc1116/BParking/ntuPROD/data_BToKmumuNtuple/PR27_BToKmumu/B4",
    #"/vols/cms/vc1116/BParking/ntuPROD/data_BToKmumuNtuple/PR27_BToKmumu/B5",
]:
    for f in os.listdir(folder):
        if re.match("\w+.root",f):
            fullFilePath = os.path.join(folder,f)

            backgroundFiles.append(fullFilePath)
            
backgroundChain = Chain(backgroundFiles)
'''
signalFilesMu = []
for f in os.listdir("/vols/cms/vc1116/BParking/ntuPROD/mc_BToKmumuNtuple/PR27_BToKmumu"):
    if re.match("\w+.root",f):
        fullFilePath = os.path.join("/vols/cms/vc1116/BParking/ntuPROD/mc_BToKmumuNtuple/PR27_BToKmumu",f)
        signalFilesMu.append(fullFilePath)
        
signalChainMu = Chain(signalFilesMu[0:10])

nEvents,nPassBaseSelection,nPassSignalSelection = calculateEfficiency(
    signalChainMu,
    baseSelection=baseSelectionMu,
    signalSelection=signalSelectionMu,
    combinationSelection=combinationSelectionMu,
    nCombinations = lambda tree:tree.nBToKmumu,
    combinationCLVtx = lambda tree,i:-tree.BToKmumu_Chi2_vtx[i],
    genCombination = lambda tree:tree.BToKmumu_gen_index,
    isSignal=True
)
print "signal mu",nEvents,nPassBaseSelection,nPassSignalSelection,100.*nPassBaseSelection/nEvents,"%",100.*nPassSignalSelection/nPassBaseSelection,"%"
'''
signalFilesEle = []
for f in os.listdir("/vols/cms/tstreble/BPH/BToKee_ntuple/BToKee_18_09_07_elechargefix/"):
    if re.match("BToKee\w+.root",f):
        fullFilePath = os.path.join("/vols/cms/tstreble/BPH/BToKee_ntuple/BToKee_18_09_07_elechargefix",f)
        signalFilesEle.append(fullFilePath)
signalChainEle = Chain(signalFilesEle)


nEvents,nPassBaseSelection,nPassSignalSelection = calculateEfficiency(
    signalChainEle,
    baseSelection=baseSelectionEle,
    signalSelection=signalSelectionEle,
    combinationSelection=combinationSelectionEle,
    nCombinations = lambda tree:tree.nBToKee,
    combinationCLVtx = lambda tree,i:-tree.BToKmumu_Chi2_vtx[i],
    genCombination = lambda tree:tree.BToKee_gen_index,
    isSignal=True
)
print "signal ele",nEvents,nPassBaseSelection,nPassSignalSelection,100.*nPassBaseSelection/nEvents,"%",100.*nPassSignalSelection/nPassBaseSelection,"%"
'''

''''
nEvents,nPassBaseSelection,nPassSignalSelection = calculateEfficiency(
    backgroundChain,
    baseSelection=baseSelectionMu,
    signalSelection=signalSelectionMu,
    combinationSelection=combinationSelectionMu,
    nCombinations = lambda tree:tree.nBToKmumu,
    combinationCLVtx = lambda tree,i:-tree.BToKmumu_Chi2_vtx[i],
    genCombination = lambda tree:tree.BToKmumu_gen_index,
    isSignal=False
)
print "background mu",nEvents,nPassBaseSelection,nPassSignalSelection,100.*nPassBaseSelection/nEvents,"%",100.*nPassSignalSelection/nPassBaseSelection,"%"
'''



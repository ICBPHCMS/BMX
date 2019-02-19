import sys
import os
import time
import numpy
import math
import array
import random
import h5py
import re
import time
import uproot

def getValue(tree,field,n):
    #print "get",field
    if not hasattr(tree,field):
        print "Field '"+field+"' not found in tree '"+tree.GetCurrentFile()+"'"
        return 0
    arr = getattr(tree,field)
    #print " -> ",arr,len(arr)
    if len(arr)<=n:
        print "Field '"+field+"' has insufficient length! "+str(n)+" vs. "+str(len(arr))+" in tree '"+tree.GetCurrentFile()+"'"
        return 0
    return arr[n]

def deltaPhi(phi1,phi2):
    if (math.fabs(phi1-phi2)<2.*math.pi):
        return math.fabs(phi1-phi2)
    n = round((phi1-phi2)/(2.*math.pi))
    return math.fabs((phi1-phi2)-n*2.*math.pi)
    
def jetSelection(tree, ijet):
    if getValue(tree,"Jet_pt",ijet)<20.:
        return False
    if getValue(tree,"Jet_jetId",ijet)==0:
        return False 
    if getValue(tree,"Jet_cleanmask",ijet)==0:
        return False
    if math.fabs(getValue(tree,"Jet_eta",ijet))>5.0:
        return False
    return True
    
def closestJet(tree,eta,phi):
    minDR2 = 100.
    for ijet in range(tree.nJet):
        if not jetSelection(tree,ijet):
            continue
        dr2 = (eta-getValue(tree,"Jet_eta",ijet))**2+deltaPhi(phi,getValue(tree,"Jet_phi",ijet))**2
        minDR2 = min(minDR2,dr2)
    return math.sqrt(minDR2)
    
def nJets(tree):
    njets = 0
    for ijet in range(tree.nJet):
        if not jetSelection(tree,ijet):
            continue
        njets+=1
    return njets
    
def leadingJetPt(tree):
    pt = -1
    for ijet in range(tree.nJet):
        if not jetSelection(tree,ijet):
            continue
        if pt<getValue(tree,"Jet_pt",ijet):
            pt = getValue(tree,"Jet_pt",ijet)
    return pt
    
def nCombinations(tree):
    nComb = 0
    for icomb in range(tree.nBToKstll):
        if not combinationSelection(tree,icomb,False):
            continue
        nComb+=1
    return nComb

def muonIsolation(tree,index):
    if index<0:
        return -1    
    if tree.nMuon>=index:
        return -1
    return getValue(tree,"Muon_pfRelIso03_all",index)

globalFeatures = [
    ["ncombinations",lambda tree: nCombinations(tree)],
    ["njets",lambda tree: nJets(tree)],
    ["leadingjet_pt",lambda tree: leadingJetPt(tree)]
]

featuresPerCombination = [
    #pt
    ["lepton1_pt",lambda i,tree: getValue(tree,"BToKstll_lep1_pt",i)],
    ["lepton2_pt",lambda i,tree: getValue(tree,"BToKstll_lep2_pt",i)],
    ["k_pt",lambda i,tree: getValue(tree,"BToKstll_kaon_pt",i)],

    #pt rel
    ["lepton1_ptrel",lambda i,tree: getValue(tree,"BToKstll_lep1_pt",i)/(getValue(tree,"BToKstll_B_pt",i)+1e-10)],
    ["lepton2_ptrel",lambda i,tree: getValue(tree,"BToKstll_lep2_pt",i)/(getValue(tree,"BToKstll_B_pt",i)+1e-10)],
    ["k_ptrel",lambda i,tree: getValue(tree,"BToKstll_kaon_pt",i)/(getValue(tree,"BToKstll_B_pt",i)+1e-10)],

    #eta
    ["lepton1_eta",lambda i,tree: math.fabs(getValue(tree,"BToKstll_lep1_eta",i))],
    ["lepton2_eta",lambda i,tree: math.fabs(getValue(tree,"BToKstll_lep2_eta",i))],
    ["k_eta",lambda i,tree: math.fabs(getValue(tree,"BToKstll_kaon_eta",i))],

    #isolation if lepton (-1 for charged candidate)
    ["lepton1_iso",lambda i,tree: muonIsolation(tree,getValue(tree,"BToKstll_lep1_index",i))],
    ["lepton2_iso",lambda i,tree: muonIsolation(tree,getValue(tree,"BToKstll_lep2_index",i))],
    
    #closest jet
    ["lepton1_deltaRJet",lambda i,tree: 
        closestJet(tree,getValue(tree,"BToKstll_lep1_eta",i),getValue(tree,"BToKstll_lep1_phi",i))
    ],
    ["lepton2_deltaRJet",lambda i,tree: 
        closestJet(tree,getValue(tree,"BToKstll_lep2_eta",i),getValue(tree,"BToKstll_lep2_phi",i))
    ],
    ["k_deltaRJet",lambda i,tree: 
        closestJet(tree,getValue(tree,"BToKstll_kaon_eta",i),getValue(tree,"BToKstll_kaon_phi",i))
    ],
    
    #delta R
    ["dilepton_deltaR",lambda i,tree: math.sqrt(
        (getValue(tree,"BToKstll_lep1_eta",i)-getValue(tree,"BToKstll_lep2_eta",i))**2+\
        deltaPhi(getValue(tree,"BToKstll_lep1_phi",i),getValue(tree,"BToKstll_lep2_phi",i))**2
    )],
    ["klepton1_deltaR",lambda i,tree: math.sqrt(
        (getValue(tree,"BToKstll_lep1_eta",i)-getValue(tree,"BToKstll_kaon_eta",i))**2+\
        deltaPhi(getValue(tree,"BToKstll_lep1_phi",i),getValue(tree,"BToKstll_kaon_phi",i))**2
    )],
    ["klepton2_deltaR",lambda i,tree: math.sqrt(
        (getValue(tree,"BToKstll_lep2_eta",i)-getValue(tree,"BToKstll_kaon_eta",i))**2+\
        deltaPhi(getValue(tree,"BToKstll_lep2_phi",i),getValue(tree,"BToKstll_kaon_phi",i))**2
    )],
    
    #delta XY
    ["dilepton_deltaXY",lambda i,tree: math.fabs(
        getValue(tree,"BToKstll_lep1_dxy",i)-getValue(tree,"BToKstll_lep2_dxy",i)
    )],
    ["klepton1_deltaXY",lambda i,tree: math.fabs(
        getValue(tree,"BToKstll_lep1_dxy",i)-getValue(tree,"BToKstll_kaon_dxy",i)
    )],
    ["klepton2_deltaXY",lambda i,tree: math.fabs(
        getValue(tree,"BToKstll_lep2_dxy",i)-getValue(tree,"BToKstll_kaon_dxy",i)
    )],
    
    #delta Z
    ["dilepton_deltaZ",lambda i,tree: math.fabs(
        getValue(tree,"BToKstll_lep1_dz",i)-getValue(tree,"BToKstll_lep2_dz",i)
    )],
    ["klepton1_deltaZ",lambda i,tree: math.fabs(
        getValue(tree,"BToKstll_lep1_dz",i)-getValue(tree,"BToKstll_kaon_dz",i)
    )],
    ["klepton2_deltaZ",lambda i,tree: math.fabs(
        getValue(tree,"BToKstll_lep2_dz",i)-getValue(tree,"BToKstll_kaon_dz",i)
    )],
    
    #B features
    ["B_pt",lambda i,tree: getValue(tree,"BToKstll_B_pt",i)],
    ["B_eta",lambda i,tree: math.fabs(getValue(tree,"BToKstll_B_eta",i))],
    ["alpha",lambda i,tree: getValue(tree,"BToKstll_B_cosAlpha",i)], #angle between B and (SV-PV)
    ["Lxy",lambda i,tree: getValue(tree,"BToKstll_B_Lxy",i)], #significance of displacement
    ["ctxy",lambda i,tree: getValue(tree,"BToKstll_B_ctxy",i)],
    ["vtx_CL",lambda i,tree: getValue(tree,"BToKstll_B_CL_vtx",i)],
    ["vtx_Chi2",lambda i,tree: getValue(tree,"BToKstll_B_Chi2_vtx",i)]
]


def myHash(value):
    h = ((int(value) >> 16) ^ int(value)) * 0x45d9f3b
    h = ((h >> 16) ^ h) * 0x45d9f3b
    h = (h >> 16) ^ h
    return h
    
    
def combinationSelection(tree,icombination,isSignal):
    if (tree.BToKstll_lep1_pt[icombination]<1. or math.fabs(tree.BToKstll_lep1_eta[icombination])>=2.4):
        return False
    if (tree.BToKstll_lep2_pt[icombination]<1. or math.fabs(tree.BToKstll_lep2_eta[icombination])>=2.4):
        return False
    if (tree.BToKstll_kaon_pt[icombination]<1. or math.fabs(tree.BToKstll_kaon_eta[icombination])>=2.4):
        return False
        
    #note: these cuts should also be applied already in the ntuples
    if (tree.BToKstll_B_CL_vtx[icombination]<0.001):
        return False
    if (tree.BToKstll_B_mass[icombination]<4. or tree.BToKstll_B_mass[icombination]>8.):
        return False
    return True
    
def baseSelection(tree,isSignal):
    #at least one b hypothesis
    if (tree.nBToKstll==0):
        return False
    
    #an additional tag muon
    if (tree.Muon_sel_index<0):
        return False
        
    if (tree.BToKstll_sel_index<0):
        return False
    
    #signal defined to be fully matched to gen (no product missing after reconstruction)
    if (isSignal and tree.BToKstll_gen_index<0):
        return False
        
    if not combinationSelection(tree,tree.BToKstll_sel_index,isSignal):
        return False
        
    return True
    
   
def refSelectionMu(tree,icombination):
    if getValue(tree,"BToKstll_B_CL_vtx",icombination)<=0.1:
        return False
    if getValue(tree,"BToKstll_B_cosAlpha",icombination)<=0.999:
        return False
    if getValue(tree,"BToKstll_B_Lxy",icombination)<=6.:
        return False
    if getValue(tree,"BToKstll_B_pt",icombination)<=10.:
        return False
    if getValue(tree,"BToKstll_kaon_pt",icombination)<=1.5:
        return False
        
    return True
    
    
def buildArrays(tree,Ncomb,selectedCombinationsSortedByVtxCL,isSignal=False):
    gobalFeatureArray = numpy.zeros((Ncomb,len(globalFeatures)),dtype=numpy.float32)
    combinationFeatureArray = numpy.zeros((Ncomb,len(featuresPerCombination)),dtype=numpy.float32)
    #one hot encoding of correct triplet (last one if no triplet is correct or background)
    genIndexArray = numpy.zeros((Ncomb+1),dtype=numpy.float32)
    bmassArray = numpy.zeros((Ncomb),dtype=numpy.float32)
    refSelArray = numpy.zeros((Ncomb),dtype=numpy.float32)
    
    #set to last one by default == no triplet is correct
    genCombinationIndex = Ncomb
    if isSignal:
        genIndex = int(tree.BToKstll_gen_index)
        if genIndex>=0:
            #check if triplet is selected
            for iselectedCombination,combinationIndex in enumerate(selectedCombinationsSortedByVtxCL):
                if combinationIndex==genIndex:
                    genCombinationIndex = iselectedCombination
                    break
           
    if genCombinationIndex>=Ncomb:
        genCombinationIndex = Ncomb
    genIndexArray[genCombinationIndex] = 1.
    
    for iselectedCombination,combinationIndex in enumerate(selectedCombinationsSortedByVtxCL):
        if iselectedCombination>=Ncomb:
            break
        
        for ifeature in range(len(globalFeatures)):
            value = globalFeatures[ifeature][1](tree)
            gobalFeatureArray[ifeature]=value
        for icombfeature in range(len(featuresPerCombination)):
            value = featuresPerCombination[ifeature][1](combinationIndex,tree)
            combinationFeatureArray[iselectedCombination,ifeature]=value
       
        
        bmassArray[iselectedCombination] = getValue(tree,"BToKstll_B_mass",combinationIndex)
        refSelArray[iselectedCombination] = 1. if refSelectionMu(tree,combinationIndex) else 0.
 
    truthArray = numpy.array([1. if isSignal else 0.],dtype=numpy.float32)
    
    return {
        "feature":combinationFeatureArray,
        "global":gobalFeatureArray,
        "genIndex":genIndexArray,
        "refSel":refSelArray,
        "bmass":bmassArray,
        "truth":truthArray
    }
    
    
def writeEvent(tree,index,writer,isSignal=False,isTestData=False,fCombination=-1,fixedLen=-1):
    
    
    if (not baseSelection(tree,isSignal)):
        return False
    
    massVtxCLIndex=[]
    selectedCombinations=[]
    
    for icombination in range(tree.nBToKstll):
        if not combinationSelection(tree,icombination,isSignal):
            continue
        massVtxCLIndex.append([
            -getValue(tree,"BToKstll_B_CL_vtx",icombination),
            getValue(tree,"BToKstll_B_mass",icombination),
            icombination
        ])
        selectedCombinations.append(icombination)
        
    if len(selectedCombinations)==0:
        return False
        
    massVtxCLIndexSortedByVtxCL = sorted(massVtxCLIndex,key=lambda elem: elem[0])    
    selectedCombinationsSortedByVtxCL = map(lambda x:x[2],massVtxCLIndexSortedByVtxCL)

    record = {}
 
    Ncomb = len(selectedCombinationsSortedByVtxCL)
    if fixedLen>0:
        Ncomb = fixedLen
    selectedCombinationsSortedByVtxCL = selectedCombinationsSortedByVtxCL[:Ncomb]
    
    #require that the gen-matched signal combination is within selected combinations
    if isSignal:
        genCombinationIndex = -1
        for i,combinationIndex in enumerate(selectedCombinationsSortedByVtxCL):
            if tree.BToKstll_gen_index==combinationIndex:
                genCombinationIndex = i
                break
          
        if genCombinationIndex<0:   
            return False

    

    data = buildArrays(tree,Ncomb,selectedCombinationsSortedByVtxCL,isSignal=isSignal)
    
    if not numpy.all(numpy.isfinite(data["feature"])):
        return False
    

    writer["features"].append(data["feature"])
    writer["truth"].append(data["truth"])
    writer["refSel"].append(data["refSel"])
    writer["bmass"].append(data["bmass"])
    writer["genIndex"].append(data["genIndex"])
    
    
    return True
    
    

class Chain(object):
    def __init__(self,fileList):
        self._fileList = fileList
        self._nEvents = []
        
        self._currentFile = None
        self._currentTree = None
        self._currentFileName = ""
        self._currentEntry = 0
        self._currentOffset = 0
        self._buffer = {}
        
        self._fileEventPairs = []
        for i,f in enumerate(self._fileList):
            try:
                print i,'/',len(self._fileList),'...',f
                rootFile = uproot.open(f,localsource=uproot.FileSource.defaults)
                tree = rootFile["Events"]
                nevents = len(tree)
                self._fileEventPairs.append([f,nevents])
            except:
                print "Error - cannot open file: ",f
                
        self._sumEvents = sum(map(lambda x:x[1],self._fileEventPairs))
            
    def GetEntries(self):
        return self._sumEvents
        
    def GetCurrentFile(self):
        return self._currentFileName
            
    def GetEntry(self,i):
        if i<0:
            print "Error - event entry negative: ",i
            i=0
        if self._currentTree!=None and (i-self._currentOffset)<len(self._currentTree) and (i-self._currentOffset)>=0:
            self._currentEntry = i-self._currentOffset
        else:
            del self._currentTree
            self._currentTree = None
            del self._buffer
            self._buffer = {}
            s = 0
            i = i%self._sumEvents #loop
            for e in range(len(self._fileEventPairs)):
                if s<=i and (s+self._fileEventPairs[e][1])>i:
                    print "opening",self._fileEventPairs[e][0]
                    self._currentFile = uproot.open(self._fileEventPairs[e][0],localsource=uproot.FileSource.defaults)
                    self._currentFileName = self._fileEventPairs[e][0]
                    self._currentTree = self._currentFile["Events"]
                    self._currentOffset = s
                    self._currentEntry = i-self._currentOffset
                    break
                s+=self._fileEventPairs[e][1]
                    
    def __getattr__(self,k):
        if not self._buffer.has_key(k):
            self._buffer[k] = self._currentTree[k].array()
            #print "loading branch ",k," with entries ",len(self._buffer[k])," and shape ",self._buffer[k][self._currentEntry].shape,", (tree len=",len(self._currentTree),")"
        if self._currentEntry>=len(self._buffer[k]):
            print "Error - buffer for branch '",k,"' only ",len(self._buffer[k])," but requested entry ",self._currentEntry," (tree len=",len(self._currentTree),")"
            return 0
        return self._buffer[k][self._currentEntry]

def convert(outputFolder,signalChain,backgroundChain,repeatSignal=1,nBatch=1,batch=0,testFractionSignal=0.2,testFractionBackground=0.5,fixedLen=-1):

     
    writerTrain = {
        "features":[],
        "scales":[],
        "genIndex":[],
        "truth":[],
        "bmass":[],
        "refSel":[]
    }
    
    writerTest = {
        "features":[],
        "scales":[],
        "genIndex":[],
        "truth":[],
        "bmass":[],
        "refSel":[]
    }
    
    
    nSignal = signalChain.GetEntries()  
    nBackground = backgroundChain.GetEntries()
 
    print "Input signal (total): ",nSignal
    print "Input background (total): ",nBackground
   
    signalEntry = 0
    backgroundEntry = 0
    
    nSignalWrittenTrain = 0
    nBackgroundWrittenTrain = 0
    
    nSignalWrittenTest = 0
    nBackgroundWrittenTest = 0
    
    nSignalBatch = int(round(1.*nSignal*repeatSignal/nBatch))
    nBackgroundBatch = nBackground #no background batching since files are already exclusive!
    
    print "Input signal batch (total): ",nSignalBatch
    print "Input background batch (total): ",nBackgroundBatch
    
    uniqueSignalTrainEntries = set()
    uniqueSignalTestEntries = set()
    
    startTime = time.time()
    
    for globalEntry in range(nSignalBatch+nBackgroundBatch):
        if globalEntry%500==0:
            #print globalEntry
            print "processed: %.3f, written: train=%i/%i, test=%i/%i, preselection eff: %.1f%%/%.1f%% signal/background"%(
                100.*globalEntry/(nSignalBatch+nBackgroundBatch),
                nSignalWrittenTrain,nBackgroundWrittenTrain,
                nSignalWrittenTest,nBackgroundWrittenTest,
                100.*(nSignalWrittenTrain+nSignalWrittenTest)/signalEntry if signalEntry>0 else 0.,
                100.*(nBackgroundWrittenTrain+nBackgroundWrittenTest)/backgroundEntry if backgroundEntry>0 else 0.,
            )
        '''
        #stop conversion if approaching timeout of 3h
        if globalEntry%100==0 and (((time.time()-startTime)/60./60.)>2.5):
            print "processed: %.3f, written: train=%i/%i, test=%i/%i, preselection eff: %.1f%%/%.1f%% signal/background"%(
                100.*globalEntry/(nSignalBatch+nBackgroundBatch),
                nSignalWrittenTrain,nBackgroundWrittenTrain,
                nSignalWrittenTest,nBackgroundWrittenTest,
                100.*(nSignalWrittenTrain+nSignalWrittenTest)/signalEntry if signalEntry>0 else 0.,
                100.*(nBackgroundWrittenTrain+nBackgroundWrittenTest)/backgroundEntry if backgroundEntry>0 else 0.,
            )
            print "Forced stop after %5.3fh"%((time.time()-startTime)/60./60.)
            break
        '''
        #pseudo randomly choose signal or background
        h = myHash(globalEntry*23+batch*7)
        itree = (h+h/(nSignalBatch+nBackgroundBatch))%(nSignalBatch+nBackgroundBatch)
        
        if (itree<(nSignalBatch)):
            signalEntry+=1
            signalEvent = (nSignalBatch*batch+signalEntry)%nSignal
            #choose train/test depending on event number only! => works even if trees are repeated 
            hSignal = myHash(signalEvent)
            hSignal = (hSignal+hSignal/1000)%1000
            signalChain.GetEntry(signalEvent)
            if (hSignal>testFractionSignal*1000):
                if (writeEvent(signalChain,nSignalWrittenTrain+nBackgroundWrittenTrain,writerTrain,isSignal=True,fixedLen=fixedLen)):
                    nSignalWrittenTrain+=1
                    uniqueSignalTrainEntries.add(signalEvent)
            else:
                if (writeEvent(signalChain,nSignalWrittenTest+nBackgroundWrittenTest,writerTest,isSignal=True,fixedLen=fixedLen)):
                    nSignalWrittenTest+=1
                    uniqueSignalTestEntries.add(signalEvent)
        else:
            backgroundEntry+=1
            backgroundEvent = (nBackgroundBatch*batch+backgroundEntry)%nBackground
            #choose train/test depending on event number only! => works even if trees are repeated 
            hBackground = myHash(backgroundEvent)
            hBackground = (hBackground+hBackground/1000)%1000
            backgroundChain.GetEntry(backgroundEvent)
            
            if (hBackground>testFractionBackground*1000):
                if (writeEvent(backgroundChain,nSignalWrittenTrain+nBackgroundWrittenTrain,writerTrain,isSignal=False,fixedLen=fixedLen)):
                    nBackgroundWrittenTrain+=1 
            else:
                if (writeEvent(backgroundChain,nSignalWrittenTest+nBackgroundWrittenTest,writerTest,isSignal=False,fixedLen=fixedLen)):
                    nBackgroundWrittenTest+=1 
                    

    for k in writerTrain.keys():
        writerTrain[k] = numpy.stack(writerTrain[k],axis=0)
    for k in writerTest.keys():
        writerTest[k] = numpy.stack(writerTest[k],axis=0)
    
    numpy.savez_compressed(
        os.path.join(outputFolder,"train_%i_%i.npz"%(nBatch,batch)),
        **writerTrain
    )
    
    numpy.savez_compressed(
        os.path.join(outputFolder,"test_%i_%i.npz"%(nBatch,batch)),
        **writerTest
    )
    
    
    print "Total signal written: %i/%i/%i total/train/test, test frac: %.3f"%(signalEntry,nSignalWrittenTrain,nSignalWrittenTest,100.*nSignalWrittenTest/(nSignalWrittenTest+nSignalWrittenTrain))
    print "Total background written: %i/%i/%i total/train/test, test frac: %.3f"%(backgroundEntry,nBackgroundWrittenTrain,nBackgroundWrittenTest,100.*nBackgroundWrittenTest/(nBackgroundWrittenTest+nBackgroundWrittenTrain))
    print "Overlap signal train/test: %i"%(len(uniqueSignalTrainEntries.intersection(uniqueSignalTestEntries)))

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-n', type=int, dest='n', help="Number of batches")
parser.add_argument('-b', type=int, dest='b', help="Current batch")
parser.add_argument('-o','--output', type=str, dest='output', help="Ouput folder")

args = parser.parse_args()


signalFiles = [
    #'/vols/cms/vc1116/BParking/ntuPROD/BPH_MC_Ntuple/ntu_PR38_LTT_BPH_MC_BuToK_ToMuMu_1.root',
    #'/vols/cms/vc1116/BParking/ntuPROD/BPH_MC_Ntuple/ntu_PR38_LTT_BPH_MC_BuToK_ToMuMu_2.root',
    '/vols/cms/vc1116/BParking/ntuPROD/BPH_MC_Ntuple/ntu_PR38_LTT_BPH_MC_BuToK_ToMuMu.root'
]


backgroundFiles = []

for folder in [
    "/vols/cms/vc1116/BParking/ntuPROD/data_BToKmumuNtuple/PR38_ltt_BToKmumu/A1",
    "/vols/cms/vc1116/BParking/ntuPROD/data_BToKmumuNtuple/PR38_ltt_BToKmumu/A2",
    "/vols/cms/vc1116/BParking/ntuPROD/data_BToKmumuNtuple/PR38_ltt_BToKmumu/A3",
    "/vols/cms/vc1116/BParking/ntuPROD/data_BToKmumuNtuple/PR38_ltt_BToKmumu/A4",
    "/vols/cms/vc1116/BParking/ntuPROD/data_BToKmumuNtuple/PR38_ltt_BToKmumu/A5",
    "/vols/cms/vc1116/BParking/ntuPROD/data_BToKmumuNtuple/PR38_ltt_BToKmumu/A6",
    "/vols/cms/vc1116/BParking/ntuPROD/data_BToKmumuNtuple/PR38_ltt_BToKmumu/B4",
    "/vols/cms/vc1116/BParking/ntuPROD/data_BToKmumuNtuple/PR38_ltt_BToKmumu/B5",
    "/vols/cms/vc1116/BParking/ntuPROD/data_BToKmumuNtuple/PR38_ltt_BToKmumu/D1",
]:
    for f in os.listdir(folder):
        if re.match("\w+.root",f):
            fullFilePath = os.path.join(folder,f)
            backgroundFiles.append([fullFilePath,os.path.getsize(fullFilePath)])
            
           
random.seed(1234)
random.shuffle(signalFiles)

backgroundFiles = sorted(backgroundFiles,key=lambda x: x[1])
backgroundFilesBatched = []
for b in range(args.n):
    for i in range(int(1.*len(backgroundFiles)/args.n/2)):
        backgroundFilesBatched.append(backgroundFiles[(2*i*args.n+b)%len(backgroundFiles)][0])
        backgroundFilesBatched.append(backgroundFiles[((2*i+1)*args.n-b)%len(backgroundFiles)][0])


            
batchStartBackground = int(round(1.*len(backgroundFilesBatched)/args.n*args.b))
batchEndBackground = int(round(1.*len(backgroundFilesBatched)/args.n*(args.b+1)))

print "bkg slice: ",batchStartBackground,"-",batchEndBackground,"/",len(backgroundFilesBatched)
#sys.exit(1)
signalChain = Chain(signalFiles)
backgroundChain = Chain(backgroundFilesBatched[batchStartBackground:batchEndBackground])
#backgroundChain = Chain([backgroundFiles[0][0],backgroundFiles[1][0]])

convert(args.output,signalChain,backgroundChain,repeatSignal=50,nBatch=args.n,batch=args.b,fixedLen=10)




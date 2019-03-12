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
import glob
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
    
def closestPUVertex(tree,dz):
    minDZ = 100.
    #http://cmslxr.fnal.gov/source/DataFormats/PatCandidates/src/PackedCandidate.cc#0035
    #dZ = vertex_Z - candidate_Z
    for i in range(tree.nOtherPV):
        minDZ = min(minDZ,math.fabs(tree.OtherPV_z[i]-(tree.PV_z-dz)))
    return minDZ
    
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
    
def leadingSVPt(tree):
    pt = -1
    for isv in range(tree.nSV):
        if pt<getValue(tree,"SV_pt",isv):
            pt = getValue(tree,"SV_pt",isv)
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
    ["leadingjet_pt",lambda tree: leadingJetPt(tree)],
    ["nsv",lambda tree: tree.nSV],
    ["leadingSV_pt",lambda tree: leadingSVPt(tree)]
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
    ["lepton1_abseta",lambda i,tree: math.fabs(getValue(tree,"BToKstll_lep1_eta",i))],
    ["lepton2_abseta",lambda i,tree: math.fabs(getValue(tree,"BToKstll_lep2_eta",i))],
    ["k_eta",lambda i,tree: math.fabs(getValue(tree,"BToKstll_kaon_eta",i))],

    #isolation if lepton (-1 for charged candidate)
    ["lepton1_iso",lambda i,tree: muonIsolation(tree,getValue(tree,"BToKstll_lep1_index",i))],
    ["lepton2_iso",lambda i,tree: muonIsolation(tree,getValue(tree,"BToKstll_lep2_index",i))],
    
    #2nd lepton = muon?
    ["lepton2_isPFLepton",lambda i,tree: 1. if getValue(tree,"BToKstll_lep2_isPFLep",i) else 0.],
    
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
    ["lepton1lepton2_deltaXY",lambda i,tree: math.fabs(
        getValue(tree,"BToKstll_lep1_dxy",i)-getValue(tree,"BToKstll_lep2_dxy",i)
    )],
    ["klepton1_deltaXY",lambda i,tree: math.fabs(
        getValue(tree,"BToKstll_lep1_dxy",i)-getValue(tree,"BToKstll_kaon_dxy",i)
    )],
    ["klepton2_deltaXY",lambda i,tree: math.fabs(
        getValue(tree,"BToKstll_lep2_dxy",i)-getValue(tree,"BToKstll_kaon_dxy",i)
    )],
    
    #delta Z
    ["lepton1lepton2_deltaZ",lambda i,tree: math.fabs(
        getValue(tree,"BToKstll_lep1_dz",i)-getValue(tree,"BToKstll_lep2_dz",i)
    )],
    ["klepton1_deltaZ",lambda i,tree: math.fabs(
        getValue(tree,"BToKstll_lep1_dz",i)-getValue(tree,"BToKstll_kaon_dz",i)
    )],
    ["klepton2_deltaZ",lambda i,tree: math.fabs(
        getValue(tree,"BToKstll_lep2_dz",i)-getValue(tree,"BToKstll_kaon_dz",i)
    )],
    
    #delta VZ wrt PV?
    ["lepton1lepton2_deltaVZ",lambda i,tree: math.fabs(
        getValue(tree,"BToKstll_lep1_vz",i)-getValue(tree,"BToKstll_lep2_vz",i)
    )],
    ["klepton1_deltaVZ",lambda i,tree: math.fabs(
        getValue(tree,"BToKstll_lep1_vz",i)-getValue(tree,"BToKstll_kaon_vz",i)
    )],
    ["klepton2_deltaVZ",lambda i,tree: math.fabs(
        getValue(tree,"BToKstll_lep2_vz",i)-getValue(tree,"BToKstll_kaon_vz",i)
    )],
    
    #dxy wrt vertex
    ["k_deltaXY_wrtVtx",lambda i,tree: getValue(tree,"BToKstll_kaon_dxy_wrtllVtx",i)],
    
    #dz to PV
    ["k_deltaZ_PV",lambda i,tree: getValue(tree,"BToKstll_kaon_dz",i)],
    
    #dz to next PU vertex
    ["k_deltaZ_PU",lambda i,tree: closestPUVertex(tree,getValue(tree,"BToKstll_kaon_dz",i))],
    
    #B features
    ["B_pt",lambda i,tree: getValue(tree,"BToKstll_B_pt",i)],
    ["B_abseta",lambda i,tree: math.fabs(getValue(tree,"BToKstll_B_eta",i))],
    ["B_cos_alpha",lambda i,tree: getValue(tree,"BToKstll_B_cosAlpha",i)], #angle between B and (SV-PV)
    ["B_Lxy",lambda i,tree: getValue(tree,"BToKstll_B_Lxy",i)], #significance of displacement
    ["B_ctxy",lambda i,tree: getValue(tree,"BToKstll_B_ctxy",i)],
    ["B_vtx_CL",lambda i,tree: getValue(tree,"BToKstll_B_CL_vtx",i)],
    ["B_vtx_Chi2",lambda i,tree: getValue(tree,"BToKstll_B_Chi2_vtx",i)],
    
    #ll features
    ["dilepton_pt",lambda i,tree: getValue(tree,"BToKstll_ll_pt",i)],
    ["dilepton_ptrel",lambda i,tree: getValue(tree,"BToKstll_ll_pt",i)/(getValue(tree,"BToKstll_B_pt",i)+1e-10)],
    ["dilepton_abseta",lambda i,tree: math.fabs(getValue(tree,"BToKstll_ll_eta",i))],
    ["dilepton_Lxy",lambda i,tree: getValue(tree,"BToKstll_ll_Lxy",i)],
    ["dilepton_ctxy",lambda i,tree: getValue(tree,"BToKstll_ll_ctxy",i)],
    ["dilepton_CL_vtx",lambda i,tree: getValue(tree,"BToKstll_ll_CL_vtx",i)],
    ["dilepton_Chi2_vtx",lambda i,tree: getValue(tree,"BToKstll_ll_Chi2_vtx",i)],
]

'''
genFeaturesPerCombination = [
    ["k_deltaRGenMatch",lambda i,tree: getValue(tree,"BToKstll_genR_KfromB",i)],
    ["lepton1_deltaRGenMatch",lambda i,tree: getValue(tree,"BToKstll_genR_lep1fromB",i)],
    ["lepton2_deltaRGenMatch",lambda i,tree: getValue(tree,"BToKstll_genR_lep2fromB",i)],
    ["sum_deltaRGenMatch",lambda i,tree: 
        getValue(tree,"BToKstll_genR_KfromB",i)+\
        getValue(tree,"BToKstll_genR_lep1fromB",i)+\
        getValue(tree,"BToKstll_genR_lep2fromB",i)
    ],
    ["genllmass",lambda i,tree: getValue(tree,"BToKstll_gen_llMass",i)],
    ["genbmass",lambda i,tree: getValue(tree,"BToKstll_gen_mass",i)],
]
'''

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
        
    #require 2nd lepton to be from lepton collection 
    #if (tree.BToKstll_lep2_isPFLep[icombination]<1):
    #    return False
        
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
    gobalFeatureArray = numpy.zeros(len(globalFeatures),dtype=numpy.float32)
    combinationFeatureArray = numpy.zeros((Ncomb,len(featuresPerCombination)),dtype=numpy.float32)
    #one hot encoding of correct triplet (last one if no triplet is correct or background)
    genIndexArray = numpy.zeros((Ncomb+1),dtype=numpy.float32)
    bmassArray = numpy.zeros((Ncomb),dtype=numpy.float32)
    llmassArray = numpy.zeros((Ncomb),dtype=numpy.float32)
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
    
    
    for ifeature in range(len(globalFeatures)):
        value = globalFeatures[ifeature][1](tree)
        if not numpy.isfinite(value):
            print "Warning - skipped non-finite value for '"+globalFeatures[ifeature][0]+"' in %s"%(
                "MC" if isSignal else "data"
            )
            return None
        gobalFeatureArray[ifeature]=value
    
    for iselectedCombination,combinationIndex in enumerate(selectedCombinationsSortedByVtxCL):
        if iselectedCombination>=Ncomb:
            break
        for icombfeature in range(len(featuresPerCombination)):
            value = featuresPerCombination[icombfeature][1](combinationIndex,tree)
            if not numpy.isfinite(value):
                print "Warning - skipped non-finite value for '"+featuresPerCombination[icombfeature][0]+"' in %s"%(
                    "MC" if isSignal else "data"
                )
                return None
            combinationFeatureArray[iselectedCombination,icombfeature]=value
       
        bmassValue = getValue(tree,"BToKstll_B_mass",combinationIndex)
        if not numpy.isfinite(bmassValue):
            print "Warning - skipped non-finite value for 'BToKstll_B_mass' in %s"%(
                "MC" if isSignal else "data"
            )
            return None
            
        llmassValue = getValue(tree,"BToKstll_ll_mass",combinationIndex)
        if not numpy.isfinite(llmassValue):
            print "Warning - skipped non-finite value for 'BToKstll_ll_mass' in %s"%(
                "MC" if isSignal else "data"
            )
            return None
            
        bmassArray[iselectedCombination] = bmassValue
        llmassArray[iselectedCombination] = llmassValue
        refSelArray[iselectedCombination] = 1. if refSelectionMu(tree,combinationIndex) else 0.
 
    truthArray = numpy.array(1. if isSignal else 0.,dtype=numpy.float32)
    
    return {
        "combinationFeatures":combinationFeatureArray,
        "globalFeatures":gobalFeatureArray,
        "genIndex":genIndexArray,
        "refSel":refSelArray,
        "bmass":bmassArray,
        "llmass":llmassArray,
        "truth":truthArray
    }
    
    
def writeEvent(tree,index,writer,isSignal=False,isTestData=False,fCombination=-1,maxCombinations=-1):
    
    
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
    if maxCombinations>0:
        Ncomb = maxCombinations
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
    
    if data==None:
        return False
    
    
    for k in data.keys():
        numpy.nan_to_num(data[k], copy=False)
        writer[k].append(data[k])
    
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
        self._currentSize = 0
        self._prefetchSize = 10000
        self._buffer = {}
        
        self._fileEventPairs = []
        for i,f in enumerate(self._fileList):
            try:
                print i,'/',len(self._fileList),'...',f
                rootFile = self.OpenFile(f)
                tree = rootFile["Events"]
                nevents = len(tree)
                self._fileEventPairs.append([f,nevents])
            except Exception,e:
                print "Error - cannot open file: ",f
                print e
                
        self._sumEvents = sum(map(lambda x:x[1],self._fileEventPairs))
        
    def OpenFile(self, path):
        return uproot.open(path,localsource=lambda f: 
            uproot.FileSource(f,chunkbytes=8*1024,limitbytes=1024**2)
        )
            
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
                    self._currentFile = self.OpenFile(self._fileEventPairs[e][0])
                    self._currentFileName = self._fileEventPairs[e][0]
                    self._currentTree = self._currentFile["Events"]
                    self._currentSize = len(self._currentTree)
                    self._currentOffset = s
                    self._currentEntry = i-self._currentOffset
                    break
                s+=self._fileEventPairs[e][1]
                
    def prefetchBranch(self,k):
        maxPrefetch = min(self._currentEntry+self._prefetchSize,self._currentSize)
        self._buffer[k] = {
            "data": self._currentTree[k].array(entrystart=self._currentEntry, entrystop=maxPrefetch),
            "min":self._currentEntry,
            "max":maxPrefetch
        }
        '''
        print "loading branch ",k,
        print " with entries ",len(self._buffer[k]["data"]),
        print " (tree len=%i, fetched=[%i, %i])"%(self._currentSize,self._currentEntry,maxPrefetch)
        '''
                    
    def __getattr__(self,k):
        if self._currentEntry>self._currentSize:
            print "Error - at end of file: ",self._currentEntry,self._currentSize
            return
            
        if not self._buffer.has_key(k):
            self.prefetchBranch(k)
        elif self._currentEntry<self._buffer[k]["min"] or self._currentEntry>=self._buffer[k]["max"]:
            self.prefetchBranch(k)
        
        bufferOffset = self._currentEntry-self._buffer[k]["min"]
        #print "reading ",k,"entry=%i, offset=%i"%(self._currentEntry,bufferOffset)
        return self._buffer[k]["data"][bufferOffset]
        
def convert(
    outputFolder,
    signalChain,
    backgroundChain,
    repeatSignal=1,
    nBatch=1,
    batch=0,
    testFractionSignal=0.2,
    testFractionBackground=0.5,
    maxCombinations=-1
):

     
    writerTrain = {
        "combinationFeatures":[],
        "globalFeatures":[],
        "genIndex":[],
        "refSel":[],
        "bmass":[],
        "llmass":[],
        "truth":[]
    }
    
    writerTest = {
        "combinationFeatures":[],
        "globalFeatures":[],
        "genIndex":[],
        "refSel":[],
        "bmass":[],
        "llmass":[],
        "truth":[]
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
        if globalEntry%2000==0:
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
                if (writeEvent(signalChain,nSignalWrittenTrain+nBackgroundWrittenTrain,writerTrain,isSignal=True,maxCombinations=maxCombinations)):
                    nSignalWrittenTrain+=1
                    uniqueSignalTrainEntries.add(signalEvent)
            else:
                if (writeEvent(signalChain,nSignalWrittenTest+nBackgroundWrittenTest,writerTest,isSignal=True,maxCombinations=maxCombinations)):
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
                if (writeEvent(backgroundChain,nSignalWrittenTrain+nBackgroundWrittenTrain,writerTrain,isSignal=False,maxCombinations=maxCombinations)):
                    nBackgroundWrittenTrain+=1 
            else:
                if (writeEvent(backgroundChain,nSignalWrittenTest+nBackgroundWrittenTest,writerTest,isSignal=False,maxCombinations=maxCombinations)):
                    nBackgroundWrittenTest+=1 
                    
    
    for k in writerTrain.keys():
        if len(writerTrain[k])==0:
            print "Warning - training data ",k," empty -> skip"
            del writerTrain[k]
            continue
        writerTrain[k] = numpy.stack(writerTrain[k],axis=0)
        #print "saving train data ",k," with shape ",writerTrain[k].shape
        
    for k in writerTest.keys():
        if len(writerTest[k])==0:
            print "Warning - testing data ",k," empty -> skip"
            del writerTest[k]
            continue
        writerTest[k] = numpy.stack(writerTest[k],axis=0)
        #print "saving test data ",k," with shape ",writerTest[k].shape
    '''
    numpy.savez_compressed(
        os.path.join(outputFolder,"train_%i_%i.npz"%(nBatch,batch)),
        **writerTrain
    )
    
    numpy.savez_compressed(
        os.path.join(outputFolder,"test_%i_%i.npz"%(nBatch,batch)),
        **writerTest
    )
    
    
    #store = pandas.HDFStore(os.path.join(outputFolder,"train_%i_%i.hdf5"%(nBatch,batch)))
    
    dataframe = pandas.DataFrame()
    
    for row in range(writerTrain['truth'].shape[0]):
        dataRow = {}
        for k in writerTrain.keys():
            if k=="combinationFeatures":
                for i,feature in enumerate(featuresPerCombination):
                    dataRow[feature[0]] = writerTrain[k][row,:,i]
            elif k=="globalFeatures":
                for i,feature in enumerate(globalFeatures):
                    dataRow[feature[0]] = writerTrain[k][row,i]
            else:
                dataRow[k] = writerTrain[k][row]
        dataframe = dataframe.append([dataRow])
    dataframe.to_hdf(os.path.join(outputFolder,"train_%i_%i.hdf5"%(nBatch,batch)),"data",mode='w',table=True)
    
    #store.put("globalFeatures",dataframe)    
     
    #store.close()
    '''
    with h5py.File(os.path.join(outputFolder,"train_%i_%i.hdf5"%(nBatch,batch)),'w') as  h5Train:
        for k in writerTrain.keys():
            if k=="combinationFeatures":
                group = h5Train.create_group(k)
                for i,feature in enumerate(featuresPerCombination):
                    group.create_dataset(featuresPerCombination[i][0], data=writerTrain[k][:,:,i], compression="gzip", compression_opts=4, chunks=True)
                    print "train hdf5 writing ",k,"/",featuresPerCombination[i][0],writerTrain[k][:,:,i].shape
            elif k=="globalFeatures":
                group = h5Train.create_group(k)
                for i,feature in enumerate(globalFeatures):
                    group.create_dataset(globalFeatures[i][0], data=writerTrain[k][:,i], compression="gzip", compression_opts=4,chunks=True)
                    print "train hdf5 writing ",k,"/",globalFeatures[i][0],writerTrain[k][:,i].shape
            else:
                print "train hdf5 writing ",k,writerTrain[k].shape
                h5Train.create_dataset(k, data=writerTrain[k])
    
    with h5py.File(os.path.join(outputFolder,"test_%i_%i.hdf5"%(nBatch,batch)),'w') as h5Test:
        for k in writerTest.keys():
            if k=="combinationFeatures":
                group = h5Test.create_group(k)
                for i,feature in enumerate(featuresPerCombination):
                    group.create_dataset(featuresPerCombination[i][0], data=writerTest[k][:,:,i], compression="gzip", compression_opts=4, chunks=True)
                    print "test hdf5 writing ",k,"/",featuresPerCombination[i][0],writerTest[k][:,:,i].shape
            elif k=="globalFeatures":
                group = h5Test.create_group(k)
                for i,feature in enumerate(globalFeatures):
                    group.create_dataset(globalFeatures[i][0], data=writerTest[k][:,i], compression="gzip", compression_opts=4, chunks=True)
                    print "test hdf5 writing ",k,"/",globalFeatures[i][0],writerTest[k][:,i].shape
            else:
                print "test hdf5 writing ",k,writerTest[k].shape
                h5Test.create_dataset(k, data=writerTest[k])
    
    print "Total signal written: %i/%i/%i total/train/test, test frac: %.3f"%(signalEntry,nSignalWrittenTrain,nSignalWrittenTest,100.*nSignalWrittenTest/(nSignalWrittenTest+nSignalWrittenTrain))
    print "Total background written: %i/%i/%i total/train/test, test frac: %.3f"%(backgroundEntry,nBackgroundWrittenTrain,nBackgroundWrittenTest,100.*nBackgroundWrittenTest/(nBackgroundWrittenTest+nBackgroundWrittenTrain))
    print "Overlap signal train/test: %i"%(len(uniqueSignalTrainEntries.intersection(uniqueSignalTestEntries)))

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--mc', type=str, action='append', default=[], dest='mc', help="MC (=signal) folder")
parser.add_argument('--data', type=str, action='append', default=[], dest='data', help="Data (=background) folder")
parser.add_argument('--repeatSignal', type=int, default=30, dest='repeatSignal', help="Repeats signal n times")
parser.add_argument('-n', type=int, dest='numberOfBatches', help="Number of batches")
parser.add_argument('-b', type=int, dest='currentBatchNumber', help="Current batch")
parser.add_argument('-o','--output', type=str, dest='output', help="Ouput folder")

args = parser.parse_args()

signalFiles = []
#/vols/cms/vc1116/BParking/ntuPROD/BPH_MC_Ntuple/ntu_PR38_LTT_BPH_MC_BuToK_ToMuMu_*.root
for inputPattern in args.mc:
    files = map(
        lambda f: [os.path.abspath(f),os.path.getsize(f)],
        glob.glob(inputPattern)
    )
    print "Found "+str(len(files))+" MC (=signal) from "+inputPattern
    signalFiles.extend(files)
    
backgroundFiles = []
#/vols/cms/vc1116/BParking/ntuPROD/data_BToKmumuNtuple/PR38_ltt_BToKmumu/*/*.root
for inputPattern in args.data:
    files = map(
        lambda f: [os.path.abspath(f),os.path.getsize(f)],
        glob.glob(inputPattern)
    )
    print "Found "+str(len(files))+" data (=background) from "+inputPattern
    backgroundFiles.extend(files)
    
           
random.seed(1234)
random.shuffle(signalFiles)

backgroundFiles = sorted(backgroundFiles,key=lambda x: x[1])
backgroundFilesBatched = []
for b in range(args.numberOfBatches):
    for i in range(int(round(1.*len(backgroundFiles)/args.numberOfBatches/2))):
        backgroundFilesBatched.append(backgroundFiles[(2*i*args.numberOfBatches+b)%len(backgroundFiles)][0])
        backgroundFilesBatched.append(backgroundFiles[((2*i+1)*args.numberOfBatches-b)%len(backgroundFiles)][0])


            
batchStartBackground = int(round(1.*len(backgroundFilesBatched)/args.numberOfBatches*args.currentBatchNumber))
batchEndBackground = int(round(1.*len(backgroundFilesBatched)/args.numberOfBatches*(args.currentBatchNumber+1)))

print "bkg slice: ",batchStartBackground,"-",batchEndBackground,"/",len(backgroundFilesBatched)
#sys.exit(1)
signalChain = Chain(map(lambda f: f[0],signalFiles))
backgroundChain = Chain(backgroundFilesBatched[batchStartBackground:batchEndBackground])
#backgroundChain = Chain([backgroundFiles[0][0]])#,backgroundFiles[1][0]])

convert(
    args.output,
    signalChain,
    backgroundChain,
    repeatSignal=args.repeatSignal,
    nBatch=args.numberOfBatches,
    batch=args.currentBatchNumber,
    maxCombinations=10
)




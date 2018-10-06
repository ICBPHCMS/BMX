import ROOT
import sys
import os
import time
import numpy
import math
import array
import random
import h5py
import re
import uproot
import tensorflow as tf


def _int64_feature(array):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=array))
    
def _float_feature(array):
    return tf.train.Feature(float_list=tf.train.FloatList(value=array))
    
def _bytes_feature(array):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=array))

def getValue(tree,field,n):
    #print "get",field
    if not hasattr(tree,field):
        print "Field '"+field+"' not found"
        return 0
    arr = getattr(tree,field)
    #print " -> ",arr,len(arr)
    if len(arr)<=n:
        print "Field '"+field+"' has insufficient length! "+str(n)+" vs. "+str(len(arr))
        return 0
    return arr[n]

def deltaPhi(phi1,phi2):
    if (math.fabs(phi1-phi2)<2.*math.pi):
        return math.fabs(phi1-phi2)
    n = round((phi1-phi2)/(2.*math.pi))
    return math.fabs((phi1-phi2)-n*2.*math.pi)
    
def closestJet(tree,eta,phi):
    minDR2 = 100.
    for ijet in range(tree.nJet):
        if getValue(tree,"Jet_pt",ijet)<20.:
            continue
        if getValue(tree,"Jet_eta",ijet)>5.0:
            continue
        dr2 = (eta-getValue(tree,"Jet_eta",ijet))**2+deltaPhi(phi,getValue(tree,"Jet_phi",ijet))**2
        minDR2 = min(minDR2,dr2)
    return math.sqrt(minDR2)
    


features = [
    ["mu1_iso",lambda i,tree: math.log10(1e-5+getValue(tree,"Muon_pfRelIso03_all",getValue(tree,"BToKmumu_mu1_index",i)))],
    ["mu2_iso",lambda i,tree: math.log10(1e-5+getValue(tree,"Muon_pfRelIso03_all",getValue(tree,"BToKmumu_mu2_index",i)))],
    
    ["mumu_deltaR",lambda i,tree: math.sqrt((getValue(tree,"BToKmumu_mu1_eta",i)-getValue(tree,"BToKmumu_mu2_eta",i))**2+deltaPhi(getValue(tree,"BToKmumu_mu1_phi",i),getValue(tree,"BToKmumu_mu2_phi",i))**2)],
    ["mumu_deltaxy",lambda i,tree: math.log10(math.fabs(getValue(tree,"BToKmumu_mu1_dxy",i)-getValue(tree,"BToKmumu_mu2_dxy",i))+1e-10)],
    ["mumu_deltaz",lambda i,tree: math.log10(math.fabs(getValue(tree,"BToKmumu_mu1_dz",i)-getValue(tree,"BToKmumu_mu2_dz",i))+1e-10)],
    
    ["kmu1_deltaR",lambda i,tree: math.sqrt((getValue(tree,"BToKmumu_mu1_eta",i)-getValue(tree,"BToKmumu_kaon_eta",i))**2+deltaPhi(getValue(tree,"BToKmumu_mu1_phi",i),getValue(tree,"BToKmumu_kaon_phi",i))**2)],
    ["kmu2_deltaR",lambda i,tree: math.sqrt((getValue(tree,"BToKmumu_mu2_eta",i)-getValue(tree,"BToKmumu_kaon_eta",i))**2+deltaPhi(getValue(tree,"BToKmumu_mu2_phi",i),getValue(tree,"BToKmumu_kaon_phi",i))**2)],
    
    ["kmu1_relpt",lambda i,tree: math.log10(getValue(tree,"BToKmumu_kaon_pt",i)/(getValue(tree,"BToKmumu_mu1_pt",i)+1e-10))],
    ["kmu2_relpt",lambda i,tree: math.log10(getValue(tree,"BToKmumu_kaon_pt",i)/(getValue(tree,"BToKmumu_mu2_pt",i)+1e-10))],
    
    ["kmu1_deltaxy",lambda i,tree: math.log10(math.fabs(getValue(tree,"BToKmumu_mu1_dxy",i)-getValue(tree,"BToKmumu_kaon_dxy",i))+1e-10)],
    ["kmu2_deltaxy",lambda i,tree: math.log10(math.fabs(getValue(tree,"BToKmumu_mu2_dxy",i)-getValue(tree,"BToKmumu_kaon_dxy",i))+1e-10)],
    
    ["kmu1_deltaz",lambda i,tree: math.log10(math.fabs(getValue(tree,"BToKmumu_mu1_dz",i)-getValue(tree,"BToKmumu_kaon_dz",i))+1e-10)],
    ["kmu2_deltaz",lambda i,tree: math.log10(math.fabs(getValue(tree,"BToKmumu_mu1_dz",i)-getValue(tree,"BToKmumu_kaon_dz",i))+1e-10)],
    
    ["k_pt",lambda i,tree: math.log10(max(0.1,getValue(tree,"BToKmumu_kaon_pt",i)))],
    
    ["k_eta",lambda i,tree: math.fabs(getValue(tree,"BToKmumu_kaon_eta",i))],
    ["k_dxy",lambda i,tree: math.log10(math.fabs(getValue(tree,"BToKmumu_kaon_dxy",i))+1e-10)],
    ["k_dz",lambda i,tree: math.log10(math.fabs(getValue(tree,"BToKmumu_kaon_dz",i))+1e-10)],
    
    ["k_djet",lambda i,tree: math.log10(closestJet(tree,getValue(tree,"BToKmumu_kaon_eta",i),getValue(tree,"BToKmumu_kaon_phi",i))+1e-10)],
    
    ["B_pt",lambda i,tree: math.log10(max(0.1,getValue(tree,"BToKmumu_pt",i)))],
    ["B_eta",lambda i,tree: math.fabs(getValue(tree,"BToKmumu_eta",i))],
    
    ["alpha",lambda i,tree: math.acos(max(min(getValue(tree,"BToKmumu_cosAlpha",i),-1.),1.))], #angle between B and (SV-PV)
    ["Lxy",lambda i,tree: math.log10(getValue(tree,"BToKmumu_Lxy",i)+1e-10)], #significance of displacement
    ["ctxy",lambda i,tree: math.log10(getValue(tree,"BToKmumu_ctxy",i)+1e-10)],
    ["vtx_CL",lambda i,tree: math.log10(max(getValue(tree,"BToKmumu_CL_vtx",i),1e-5))]
]

scales = [
    ["mumu_mass",lambda i,tree: math.fabs(getValue(tree,"BToKmumu_mumu_mass",i))],
    ["B_mass",lambda i,tree: math.fabs(getValue(tree,"BToKmumu_mass",i))],
]

def countSelected(tree,selection):
    hist = ROOT.TH1F("selection"+str(random.random())+hash(selection),"",1,-1,1)
    tree.Project(hist.GetName(),selection)
    return hist.GetEntries()
    
def myHash(value):
    h = ((int(value) >> 16) ^ int(value)) * 0x45d9f3b
    h = ((h >> 16) ^ h) * 0x45d9f3b
    h = (h >> 16) ^ h
    return h
    
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
    if (isSignal and tree.BToKmumu_gen_index<0):
        return False
        
    if (getValue(tree,"BToKmumu_kaon_pt",tree.BToKmumu_sel_index)<=1. or math.fabs(getValue(tree,"BToKmumu_kaon_eta",tree.BToKmumu_sel_index))>=2.4):
        return False
        
    if (getValue(tree,"BToKmumu_mu1_pt",tree.BToKmumu_sel_index)<=1. or math.fabs(getValue(tree,"BToKmumu_mu1_eta",tree.BToKmumu_sel_index))>=2.4):
        return False
        
    if (getValue(tree,"BToKmumu_mu2_pt",tree.BToKmumu_sel_index)<=1. or math.fabs(getValue(tree,"BToKmumu_mu2_eta",tree.BToKmumu_sel_index))>=2.4):
        return False
        
    if (getValue(tree,"BToKmumu_mu1_charge",tree.BToKmumu_sel_index)*getValue(tree,"BToKmumu_mu2_charge",tree.BToKmumu_sel_index)>0):
        return False
        
    #veto B mass window for background to reject signal in data
    if (not isSignal and getValue(tree,"BToKmumu_mass",tree.BToKmumu_sel_index)<5.6):
        return False
        
    return True
    
    
def writeEvent(tree,index,writer,isSignal=False,fixedLen=-1):
    
    if (not baseSelection(tree,isSignal)):
        return False
    
    massVtxCLIndex=[]
    selectedCombinations=[]
    
    for icombination in range(tree.nBToKmumu):
        #select a significance of at least 0.1% (note: this is calculated from chi2 of fit; =0 occurs through numerical precision)
        #if (getValue(tree,"BToKmumu_CL_vtx",icombination)<0.001):
        #    continue
        if (getValue(tree,"BToKmumu_mu1_pt",icombination)<1. or math.fabs(getValue(tree,"BToKmumu_mu1_eta",icombination))>=2.4):
            continue
        if (getValue(tree,"BToKmumu_mu2_pt",icombination)<1. or math.fabs(getValue(tree,"BToKmumu_mu2_eta",icombination))>=2.4):
            continue
        if (getValue(tree,"BToKmumu_kaon_pt",icombination)<1. or math.fabs(getValue(tree,"BToKmumu_kaon_eta",icombination))>=2.4):
            continue
        massVtxCLIndex.append([getValue(tree,"BToKmumu_CL_vtx",icombination),getValue(tree,"BToKmumu_mass",icombination),icombination])
        selectedCombinations.append(icombination)
    if len(selectedCombinations)==0:
        return False
    
    
    
    massVtxCLIndexSortedByVtxCL = sorted(massVtxCLIndex,key=lambda elem: -elem[0])    
    selectedCombinationsSortedByVtxCL = map(lambda x:x[2],massVtxCLIndexSortedByVtxCL)
    '''
    if (not isSignal):
        minMassBestHalf = min(map(lambda elem:elem[1],massVtxCLIndexSortedByVtxCL[:(1+len(massVtxCLIndexSortedByVtxCL)/2)]))
        if (minMassBestHalf<5.6):
            return False
    '''
    record = {}
 
    Ncomb = len(selectedCombinationsSortedByVtxCL)
    if fixedLen>0:
        Ncomb = fixedLen
    
    featureArray = numpy.zeros((Ncomb*len(features)),dtype=numpy.float32)
    scaleArray = numpy.zeros((Ncomb*len(scales)),dtype=numpy.float32)
    #one hot encoding of correct triplet (last one if no triplet is correct or background)
    genIndexArray = numpy.zeros((Ncomb+1),dtype=numpy.float32)
    
    
    #set to last one by default == no triplet is correct
    genCombinationIndex = Ncomb
    if isSignal:
        genIndex = int(tree.BToKmumu_gen_index)
        if genIndex>=0:
            #check if triplet is selected
            for iselectedCombination,combinationIndex in enumerate(selectedCombinationsSortedByVtxCL):
                #if iselectedCombination>=Ncomb:
                #    break
                if combinationIndex==genIndex:
                    genCombinationIndex = iselectedCombination
                    break
    #if isSignal:
    #    print Ncomb,len(selectedCombinationsSortedByVtxCL),genCombinationIndex,
                    
    if genCombinationIndex>=Ncomb:
        genCombinationIndex = Ncomb
    genIndexArray[genCombinationIndex] = 1.
    
    #if isSignal:
    #    print genIndexArray
    
    
    
    for iselectedCombination,combinationIndex in enumerate(selectedCombinationsSortedByVtxCL):
        if iselectedCombination>=Ncomb:
            break
        for ifeature in range(len(features)):
            value = features[ifeature][1](combinationIndex,tree)
            featureArray[iselectedCombination*len(features)+ifeature]=value
            
        for iscale in range(len(scales)):
            value = scales[iscale][1](combinationIndex,tree)
            scaleArray[iselectedCombination*len(scales)+iscale]=value
    
        
    truthArray = numpy.array([1. if isSignal else 0.],dtype=numpy.float32)
    '''
    batchGroup = writer.create_group(str(index))
    
    batchGroup.create_dataset("features",data=featureArray)
    batchGroup.create_dataset("scales",data=scaleArray)
    batchGroup.create_dataset("genIndex",data=genIndexArray)
    batchGroup.create_dataset("truth",data=truthArray)
    '''
    
    record = {
        "features":_float_feature(featureArray),
        "scales":_float_feature(scaleArray),
        "genIndex":_float_feature(genIndexArray),
        "truth":_float_feature(truthArray)
    }
    example = tf.train.Example(features=tf.train.Features(feature=record))
    writer.write(example.SerializeToString())
    
    return True
    
class Chain(object):
    def __init__(self,fileList):
        self._fileList = fileList
        self._nEvents = []
        
        self._currentFile = None
        self._currentTree = None
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

def convert(outputFolder,signalChain,backgroundChain,repeatSignal=50,nBatch=1,batch=0,testFractionSignal=0.3,testFractionBackground=0.9,fixedLen=-1):
    '''
    if os.path.exists(rootFileName+".tfrecord.uncompressed"):
        logging.info("exists ... "+rootFileName+".tfrecord.uncompressed -> skip")
        return
    '''
    
    #writerTrain = h5py.File(os.path.join(outputFolder,"train_%i_%i.hdf5"%(nBatch,batch)),'w')
    #writerTest = h5py.File(os.path.join(outputFolder,"test_%i_%i.hdf5"%(nBatch,batch)),'w')
    
    writerTrain = tf.python_io.TFRecordWriter(os.path.join(outputFolder,"train_%i_%i.tfrecord"%(nBatch,batch)),options=tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP))
    writerTest = tf.python_io.TFRecordWriter(os.path.join(outputFolder,"test_%i_%i.tfrecord"%(nBatch,batch)),options=tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP))
    
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
    
    for globalEntry in range(nSignalBatch+nBackgroundBatch):
        
    
        if (nSignalWrittenTrain+nBackgroundWrittenTrain+1)%50==0:
            writerTrain.flush()
    
        if (nSignalWrittenTest+nBackgroundWrittenTest+1)%50==0:
            writerTest.flush()
    
        if globalEntry%500==0:
            #print globalEntry
            print "processed: %.3f, written: train=%i/%i, test=%i/%i, preselection eff: %.1f%%/%.1f%% signal/background"%(
                100.*globalEntry/(nSignalBatch+nBackgroundBatch),
                nSignalWrittenTrain,nBackgroundWrittenTrain,
                nSignalWrittenTest,nBackgroundWrittenTest,
                100.*(nSignalWrittenTrain+nSignalWrittenTest)/signalEntry if signalEntry>0 else 0.,
                100.*(nBackgroundWrittenTrain+nBackgroundWrittenTest)/backgroundEntry if backgroundEntry>0 else 0.,
            )
        
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
                    
    writerTrain.close()
    writerTest.close()
    
    print "Total signal written: %i/%i/%i total/train/test, test frac: %.3f"%(nSignal,nSignalWrittenTrain,nSignalWrittenTest,100.*nSignalWrittenTest/(nSignalWrittenTest+nSignalWrittenTrain))
    print "Total background written: %i/%i/%i total/train/test, test frac: %.3f"%(nBackground,nBackgroundWrittenTrain,nBackgroundWrittenTest,100.*nBackgroundWrittenTest/(nBackgroundWrittenTest+nBackgroundWrittenTrain))
    print "Overlap signal train/test: %i"%(len(uniqueSignalTrainEntries.intersection(uniqueSignalTestEntries)))

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-n', type=int, dest='n', help="Number of batches")
parser.add_argument('-b', type=int, dest='b', help="Current batch")
parser.add_argument('-o','--output', type=str, dest='output', help="Ouput folder")

args = parser.parse_args()


signalFiles = []

for f in os.listdir("/vols/cms/tstreble/BPH/BToKmumu_ntuple/BToKmumu_18_08_14_new/"):
    if re.match("BToKmumuNtuple\w+.root",f):
        fullFilePath = os.path.join("/vols/cms/tstreble/BPH/BToKmumu_ntuple/BToKmumu_18_08_14_new/",f)
        signalFiles.append(fullFilePath)

backgroundList = [
    "/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking1_2018A_18_08_14_new/BToKmumuNtuple*.root",
    "/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking2_2018A_18_08_14_new/BToKmumuNtuple*.root",
    "/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking3_2018A_18_08_14_new/BToKmumuNtuple*.root",
    "/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking4_2018A_18_08_14_new/BToKmumuNtuple*.root",
    "/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking5_2018A_18_08_14_new/BToKmumuNtuple*.root",
    "/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking6_2018A_18_08_14_new/BToKmumuNtuple*.root",
    
    "/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking1_2018B_18_08_14_new/BToKmumuNtuple*.root",
    "/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking2_2018B_18_08_14_new/BToKmumuNtuple*.root",
    "/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking3_2018B_18_08_14_new/BToKmumuNtuple*.root",
    "/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking4_2018B_18_08_14_new/BToKmumuNtuple*.root",
    "/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking5_2018B_18_08_14_new/BToKmumuNtuple*.root",
    "/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking6_2018B_18_08_14_new/BToKmumuNtuple*.root",
]

backgroundFiles = []

for folder in [
    '/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking1_2018A_18_08_14_new/',
    '/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking2_2018A_18_08_14_new/',
    '/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking3_2018A_18_08_14_new/',
    '/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking4_2018A_18_08_14_new/',
    '/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking5_2018A_18_08_14_new/',
    '/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking6_2018A_18_08_14_new/',
    
    '/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking1_2018B_18_08_14_new/',
    '/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking2_2018B_18_08_14_new/',
    '/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking3_2018B_18_08_14_new/',
    '/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking4_2018B_18_08_14_new/',
    '/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking5_2018B_18_08_14_new/',
]:
    for f in os.listdir(folder):
        if re.match("BToKmumuNtuple\w+.root",f):
            fullFilePath = os.path.join(folder,f)
            '''
            rootFile = ROOT.TFile(fullFilePath)
            if not rootFile:
                print "skip",fullFilePath
                rootFile.Close()
                continue
            tree = rootFile.Get("Events")
            if not tree:
                print "skip",fullFilePath
                rootFile.Close()
                continue
            rootFile.Close()
            '''
            backgroundFiles.append(fullFilePath)
           
random.seed(1234)
random.shuffle(backgroundFiles)
random.shuffle(signalFiles)
            
batchStartBackground = int(round(1.*len(backgroundFiles)/args.n*args.b))
batchEndBackground = int(round(1.*len(backgroundFiles)/args.n*(args.b+1)))

print "bkg slice: ",batchStartBackground,"-",batchEndBackground,"/",len(backgroundFiles)

signalChain = Chain(signalFiles)
backgroundChain = Chain(backgroundFiles[batchStartBackground:batchEndBackground])



#backgroundChain.Add("/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking1_2018A_18_08_14_new/BToKmumuNtuple*.root")
#backgroundChain.Add("/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking2_2018A_18_08_14_new/BToKmumuNtuple*.root")
#backgroundChain.Add("/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking3_2018A_18_08_14_new/BToKmumuNtuple*.root")
#backgroundChain.Add("/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking4_2018A_18_08_14_new/BToKmumuNtuple*.root")
#backgroundChain.Add("/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking5_2018A_18_08_14_new/BToKmumuNtuple*.root")
#backgroundChain.Add("/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking6_2018A_18_08_14_new/BToKmumuNtuple*.root")


convert(args.output,signalChain,backgroundChain,nBatch=args.n,batch=args.b,fixedLen=200)




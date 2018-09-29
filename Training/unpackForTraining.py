import ROOT
import sys
import os
import time
import numpy
import math
import array
import h5py

def deltaPhi(phi1,phi2):
    if (math.fabs(phi1-phi2)<2.*math.pi):
        return math.fabs(phi1-phi2)
    n = round((phi1-phi2)/(2.*math.pi))
    return math.fabs((phi1-phi2)-n*2.*math.pi)
    
def closestJet(tree,eta,phi):
    minDR2 = 100.
    for ijet in range(tree.nJet):
        if tree.Jet_pt[ijet]<20.:
            continue
        if tree.Jet_eta[ijet]>5.0:
            continue
        dr2 = (eta-tree.Jet_eta[ijet])**2+deltaPhi(phi,tree.Jet_phi[ijet])**2
        minDR2 = min(minDR2,dr2)
    return math.sqrt(minDR2)

features = [
    ["mu1_iso",lambda i,tree: math.log10(1e-5+tree.Muon_pfRelIso03_all[tree.BToKmumu_mu1_index[i]])],
    ["mu2_iso",lambda i,tree: math.log10(1e-5+tree.Muon_pfRelIso03_all[tree.BToKmumu_mu2_index[i]])],
    ["kmu1_relpt",lambda i,tree: math.log10(tree.BToKmumu_kaon_pt[i]/(tree.BToKmumu_mu1_pt[i]))],
    ["mumu_deltaR",lambda i,tree: math.sqrt((tree.BToKmumu_mu1_eta[i]-tree.BToKmumu_mu2_eta[i])**2+deltaPhi(tree.BToKmumu_mu1_phi[i],tree.BToKmumu_mu2_phi[i])**2)],
    ["mumu_deltaxy",lambda i,tree: math.log10(math.fabs(tree.BToKmumu_mu1_dxy[i]-tree.BToKmumu_mu2_dxy[i])+1e-10)],
    ["mumu_deltaz",lambda i,tree: math.log10(math.fabs(tree.BToKmumu_mu1_dz[i]-tree.BToKmumu_mu2_dz[i])+1e-10)],
    ["kmu1_deltaR",lambda i,tree: math.sqrt((tree.BToKmumu_mu1_eta[i]-tree.BToKmumu_kaon_eta[i])**2+deltaPhi(tree.BToKmumu_mu1_phi[i],tree.BToKmumu_kaon_phi[i])**2)],
    ["kmu2_deltaR",lambda i,tree: math.sqrt((tree.BToKmumu_mu2_eta[i]-tree.BToKmumu_kaon_eta[i])**2+deltaPhi(tree.BToKmumu_mu2_phi[i],tree.BToKmumu_kaon_phi[i])**2)],
    ["kmu1_deltaxy",lambda i,tree: math.log10(math.fabs(tree.BToKmumu_mu1_dxy[i]-tree.BToKmumu_kaon_dxy[i])+1e-10)],
    ["kmu1_deltaz",lambda i,tree: math.log10(math.fabs(tree.BToKmumu_mu1_dz[i]-tree.BToKmumu_kaon_dz[i])+1e-10)],
    ["kmu2_relpt",lambda i,tree: math.log10(tree.BToKmumu_kaon_pt[i]/(tree.BToKmumu_mu2_pt[i]))],
    ["kmu2_deltaxy",lambda i,tree: math.log10(math.fabs(tree.BToKmumu_mu2_dxy[i]-tree.BToKmumu_kaon_dxy[i])+1e-10)],
    ["kmu2_deltaz",lambda i,tree: math.log10(math.fabs(tree.BToKmumu_mu1_dz[i]-tree.BToKmumu_kaon_dz[i])+1e-10)],
    ["k_relpt",lambda i,tree: tree.BToKmumu_mass[i]/(tree.BToKmumu_kaon_pt[i])],
    ["k_eta",lambda i,tree: math.fabs(tree.BToKmumu_kaon_eta[i])],
    ["k_dxy",lambda i,tree: math.log10(math.fabs(tree.BToKmumu_kaon_dxy[i])+1e-10)],
    ["k_dz",lambda i,tree: math.log10(math.fabs(tree.BToKmumu_kaon_dz[i])+1e-10)],
    ["k_djet",lambda i,tree: math.log10(closestJet(tree,tree.BToKmumu_kaon_eta[i],tree.BToKmumu_kaon_phi[i]))],
    ["alpha",lambda i,tree: math.acos(max(min(tree.BToKmumu_cosAlpha[i],-1.),1.))], #angle between B and (SV-PV)
    ["Lxy",lambda i,tree: math.log10(tree.BToKmumu_Lxy[i]+1e-10)], #significance of displacement
    ["ctxy",lambda i,tree: math.log10(tree.BToKmumu_ctxy[i]+1e-10)], #significance of displacement
    ["vtx_CL",lambda i,tree: math.log10(tree.BToKmumu_CL_vtx[i])]
]

scales = [
    #["k_pt",lambda i,tree: math.fabs(tree.BToKmumu_kaon_pt[i])],
    #["mu1_pt",lambda i,tree: math.fabs(tree.BToKmumu_mu1_pt[i])],
    #["mu2_pt",lambda i,tree: math.fabs(tree.BToKmumu_mu2_pt[i])],
    ["mumu_mass",lambda i,tree: math.fabs(tree.BToKmumu_mumu_mass[i])],
    ["B_mass",lambda i,tree: math.fabs(tree.BToKmumu_mass[i])],
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
    
    #signal defined to be fully matched to gen (no product missing after reconstruction)
    if (isSignal and tree.BToKmumu_gen_index<0):
        return False
        
    #both 
    if (tree.BToKmumu_kaon_pt[tree.BToKmumu_sel_index]<=1. and math.fabs(tree.BToKmumu_kaon_eta[tree.BToKmumu_sel_index])>=2.4):
        return False
        
    if (tree.BToKmumu_mu1_pt[tree.BToKmumu_sel_index]<=1. and math.fabs(tree.BToKmumu_mu1_pt[tree.BToKmumu_sel_index])>=2.4):
        return False
        
    if (tree.BToKmumu_mu2_pt[tree.BToKmumu_sel_index]<=1. and math.fabs(tree.BToKmumu_mu2_pt[tree.BToKmumu_sel_index])>=2.4):
        return False
        
    #veto B mass window for background to reject signal in data
    if (not isSignal and tree.BToKmumu_mass[tree.BToKmumu_sel_index]<5.6):
        return False
    
    return True
    
    
def writeEvent(tree,index,writer,isSignal=False):
    
    if (not baseSelection(tree,isSignal)):
        return False
        
    massVtxCLIndex=[]
    selectedCombinations=[]
    
    for icombination in range(tree.nBToKmumu):
        #select a significance of at least 0.1% (note: this is calculated from chi2 of fit; =0 occurs through numerical precision)
        if (tree.BToKmumu_CL_vtx[icombination]<0.001):
            continue
        if (tree.BToKmumu_mu1_pt[icombination]<1.):
            continue
        if (tree.BToKmumu_mu2_pt[icombination]<1.):
            continue
        if (tree.BToKmumu_kaon_pt[icombination]<1.):
            continue
        massVtxCLIndex.append([tree.BToKmumu_CL_vtx[icombination],tree.BToKmumu_mass[icombination],icombination])
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
 
    
    featureArray = numpy.zeros((len(selectedCombinationsSortedByVtxCL),len(features)),dtype=numpy.float32)
    scaleArray = numpy.zeros((len(selectedCombinationsSortedByVtxCL),len(scales)),dtype=numpy.float32)
    #one hot encoding of correct triplet (last one if no triplet is correct or background)
    genIndexArray = numpy.zeros((len(selectedCombinationsSortedByVtxCL)+1),dtype=numpy.float32)
    
    #set to last one by default == no triplet is correct
    genCombinationIndex = len(selectedCombinationsSortedByVtxCL)
    if isSignal:
        genIndex = int(tree.BToKmumu_gen_index)
        if genIndex>=0:
            #check if triplet is selected
            for iselectedCombination,combinationIndex in enumerate(selectedCombinationsSortedByVtxCL):
                if combinationIndex==genIndex:
                    genCombinationIndex = iselectedCombination
                    break
    genIndexArray[genCombinationIndex] = 1.
    '''
    if isSignal:
        print genIndexArray
    '''
    for iselectedCombination,combinationIndex in enumerate(selectedCombinationsSortedByVtxCL):
        for ifeature in range(len(features)):
            value = features[ifeature][1](combinationIndex,tree)
            featureArray[iselectedCombination][ifeature]=value
            
        for iscale in range(len(scales)):
            value = scales[iscale][1](combinationIndex,tree)
            scaleArray[iselectedCombination][iscale]=value
            
        
    truthArray = numpy.array([1. if isSignal else 0.],dtype=numpy.float32)
    
    batchGroup = writer.create_group(str(index))
    
    batchGroup.create_dataset("features",data=featureArray)
    batchGroup.create_dataset("scales",data=scaleArray)
    batchGroup.create_dataset("genIndex",data=genIndexArray)
    batchGroup.create_dataset("truth",data=truthArray)
    
    return True

def convert(outputFolder,signalChain,backgroundChain,repeatSignal=10,nBatch=1,batch=0,testFractionSignal=0.3,testFractionBackground=0.7,skipBackground=500):
    '''
    if os.path.exists(rootFileName+".tfrecord.uncompressed"):
        logging.info("exists ... "+rootFileName+".tfrecord.uncompressed -> skip")
        return
    '''
    
    writerTrain = h5py.File(os.path.join(outputFolder,"train_%i_%i.hdf5"%(nBatch,batch)),'w')
    writerTest = h5py.File(os.path.join(outputFolder,"test_%i_%i.hdf5"%(nBatch,batch)),'w')
    
    nSignal = signalChain.GetEntries()
    nBackground = int(1.*backgroundChain.GetEntries()/skipBackground)
    
    print "Input signal: ",nSignal
    print "Input background: ",nBackground
   
    signalEntry = 0
    backgroundEntry = 0
    
    nSignalWrittenTrain = 0
    nBackgroundWrittenTrain = 0
    
    nSignalWrittenTest = 0
    nBackgroundWrittenTest = 0
    
    
    for globalEntry in range(nSignal*repeatSignal+nBackground):
        if globalEntry%(50*nBatch)==0:
            print "processed: %.3f, written: %i/%i (%i/%i) signal/background"%(100.*globalEntry/(nSignal*repeatSignal+nBackground),nSignalWrittenTrain,nBackgroundWrittenTrain,nSignalWrittenTest,nBackgroundWrittenTest)
        
        #pseudo randomly choose signal or background
        h = myHash(globalEntry)
        itree = (h+h/(nSignal*repeatSignal+nBackground))%(nSignal*repeatSignal+nBackground)
        
        if (itree<(nSignal*repeatSignal)):
            signalEntry+=1
            #choose train/test depending on event number only! => works even if trees are repeated 
            hSignal = myHash(signalEntry%nSignal)
            hSignal = (hSignal+hSignal/1000)%1000
            #skip if not for current batch
            if (globalEntry%nBatch!=batch):
                continue
            signalChain.GetEntry(signalEntry%nSignal)
            if (hSignal>testFractionSignal*1000):
                if (writeEvent(signalChain,nSignalWrittenTrain+nBackgroundWrittenTrain,writerTrain,isSignal=True)):
                    nSignalWrittenTrain+=1
            else:
                if (writeEvent(signalChain,nSignalWrittenTest+nBackgroundWrittenTest,writerTest,isSignal=True)):
                    nSignalWrittenTest+=1
                    
        else:
            
            backgroundEntry+=1
            #choose train/test depending on event number only! => works even if trees are repeated 
            hBackground = myHash(backgroundEntry%nBackground)
            hBackground = (hBackground+hBackground/1000)%1000
            #skip if not for current batch
            if (globalEntry%nBatch!=batch):
                continue
            backgroundChain.GetEntry(backgroundEntry%nBackground)
            
            if (hBackground>testFractionBackground*1000):
                if (writeEvent(backgroundChain,nSignalWrittenTrain+nBackgroundWrittenTrain,writerTrain,isSignal=False)):
                    nBackgroundWrittenTrain+=1 
            else:
                if (writeEvent(backgroundChain,nSignalWrittenTest+nBackgroundWrittenTest,writerTest,isSignal=False)):
                    nBackgroundWrittenTest+=1 
                    
    writerTrain.close()
    writerTest.close()
    
    print "Total signal written: %i/%i/%i total/train/test, test frac: %.3f"%(nSignal,nSignalWrittenTrain,nSignalWrittenTest,100.*nSignalWrittenTest/(nSignalWrittenTest+nSignalWrittenTrain))
    print "Total background written: %i/%i/%i total/train/test, test frac: %.3f"%(nBackground,nBackgroundWrittenTrain,nBackgroundWrittenTest,100.*nBackgroundWrittenTest/(nBackgroundWrittenTest+nBackgroundWrittenTrain))


#backgroundChain.Add("/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking1_2018B_18_08_14_new/BToKmumuNtuple*.root")
#backgroundChain.Add("/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking2_2018B_18_08_14_new/BToKmumuNtuple*.root")
#backgroundChain.Add("/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking3_2018B_18_08_14_new/BToKmumuNtuple*.root")
#backgroundChain.Add("/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking4_2018B_18_08_14_new/BToKmumuNtuple*.root")
#backgroundChain.Add("/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking5_2018B_18_08_14_new/BToKmumuNtuple*.root")
#backgroundChain.Add("/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking6_2018B_18_08_14_new/BToKmumuNtuple*.root")


import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-n', type=int, dest='n', help="Number of batches")
parser.add_argument('-b', type=int, dest='b', help="Current batch")
parser.add_argument('-o','--output', type=str, dest='output', help="Ouput folder")

args = parser.parse_args()


signalChain = ROOT.TChain("Events")
signalChain.Add("/vols/cms/tstreble/BPH/BToKmumu_ntuple/BToKmumu_18_08_14_new/*.root")

backgroundChain = ROOT.TChain("Events")
backgroundList = [
    "/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking1_2018A_18_08_14_new/BToKmumuNtuple*.root",
    "/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking2_2018A_18_08_14_new/BToKmumuNtuple*.root",
    "/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking3_2018A_18_08_14_new/BToKmumuNtuple*.root",
    "/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking4_2018A_18_08_14_new/BToKmumuNtuple*.root",
    "/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking5_2018A_18_08_14_new/BToKmumuNtuple*.root",
    "/vols/cms/tstreble/BPH/BToKmumu_ntuple/BPHParking6_2018A_18_08_14_new/BToKmumuNtuple*.root",
]
backgroundChain.Add(backgroundList[args.b%len(backgroundList)])

convert(args.output,signalChain,backgroundChain,nBatch=args.n,batch=args.b,skipBackground=1)




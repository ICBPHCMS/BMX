import ROOT
import sys
import os
import time
import numpy
import math
import random
import re
import uproot


ROOT.gRandom.SetSeed(123)
ROOT.gROOT.SetBatch(True)
ROOT.gROOT.SetStyle("Plain")
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptFit(0)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

# For the Pad:
ROOT.gStyle.SetPadBorderMode(0)
ROOT.gStyle.SetPadColor(ROOT.kWhite)
ROOT.gStyle.SetPadGridX(True)
ROOT.gStyle.SetPadGridY(True)
ROOT.gStyle.SetGridColor(ROOT.kBlack)
ROOT.gStyle.SetGridStyle(2)
ROOT.gStyle.SetGridWidth(1)

ROOT.gStyle.SetHatchesSpacing(1.3)
ROOT.gStyle.SetHatchesLineWidth(2)

# For the axis titles:
ROOT.gStyle.SetTitleColor(1, "XYZ")
ROOT.gStyle.SetTitleFont(43, "XYZ")
ROOT.gStyle.SetTitleSize(30, "XYZ")
ROOT.gStyle.SetTitleXOffset(1.2)
ROOT.gStyle.SetTitleOffset(1.4, "YZ") # Another way to set the Offset

# For the axis labels:
ROOT.gStyle.SetLabelColor(1, "XYZ")
ROOT.gStyle.SetLabelFont(43, "XYZ")
ROOT.gStyle.SetLabelOffset(0.0077, "XYZ")
ROOT.gStyle.SetLabelSize(28, "XYZ")

# For the axis:
ROOT.gStyle.SetTickLength(0.03, "Y")
ROOT.gStyle.SetTickLength(0.05, "X")
ROOT.gStyle.SetNdivisions(1005, "X")
ROOT.gStyle.SetNdivisions(506, "Y")

ROOT.gStyle.SetPadTickX(1)  # To get tick marks on the opposite side of the frame
ROOT.gStyle.SetPadTickY(1)

ROOT.gStyle.SetPaperSize(8.0*1.6,7.0*1.6)
ROOT.TGaxis.SetMaxDigits(3)
ROOT.gStyle.SetLineScalePS(2)

ROOT.gStyle.SetPaintTextFormat("3.0f")


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
        
    if (tree.BToKmumu_kaon_pt[icombination]<=1. or math.fabs(tree.BToKmumu_kaon_eta[icombination])>=2.4):
        return False
        
    if (tree.BToKmumu_mu1_pt[icombination]<=1. or math.fabs(tree.BToKmumu_mu1_eta[icombination])>=2.4):
        return False
        
    if (tree.BToKmumu_mu2_pt[icombination]<=1. or math.fabs(tree.BToKmumu_mu2_eta[icombination])>=2.4):
        return False
        
    if (tree.BToKmumu_mu1_charge[icombination]*tree.BToKmumu_mu2_charge[icombination]>0):
        return False
        
    if (tree.BToKmumu_CL_vtx[icombination]<0.001):
        return False
        
    #if (tree.BToKmumu_cosAlpha[icombination]<0.9):
    #    return False
        
    if tree.BToKmumu_mass[icombination]<4. or tree.BToKmumu_mass[icombination]>7.:
        return False
        
    return True
    
def combinationSelectionEle(tree,icombination):
    #return True
    if (tree.BToKee_kaon_pt[icombination]<=1. or math.fabs(tree.BToKee_kaon_eta[icombination])>=2.4):
        return False
        
    if (tree.BToKee_ele1_pt[icombination]<=1. or math.fabs(tree.BToKee_ele1_eta[icombination])>=2.4):
        return False
        
    if (tree.BToKee_ele2_pt[icombination]<=1. or math.fabs(tree.BToKee_ele2_eta[icombination])>=2.4):
        return False
        
    if (tree.BToKee_ele1_charge[icombination]*tree.BToKee_ele2_charge[icombination]>0):
        return False
        
    if (tree.BToKee_CL_vtx[icombination]<0.001):
        return False
        
    
        
    if tree.BToKee_mass[icombination]<4. or tree.BToKmumu_mass[icombination]>7.:
        return False
        
    return True

def baseSelectionMu(tree,isSignal):
    #return True
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
        
    #gen-matched combination needs to yield bmass
    if (isSignal and (tree.BToKmumu_mass[tree.BToKmumu_gen_index]<5.15 or tree.BToKmumu_mass[tree.BToKmumu_gen_index]>5.45)):
        return False
        
    #guanarntees at least one combination surviving
    if (not combinationSelectionMu(tree,tree.BToKmumu_sel_index)):
        return False
        
    #veto B mass window for background to reject signal in data
    #if (not isSignal and tree.BToKmumu_mass[tree.BToKmumu_sel_index]>4. and tree.BToKmumu_mass[tree.BToKmumu_sel_index]<7.):
    #    return False
        
    #if (not isSignal and tree.BToKmumu_mass[tree.BToKmumu_sel_index]<3):
    #    return False
        
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
        
    #guanarntees at least one combination surviving
    if (not combinationSelectionEle(tree,tree.BToKee_sel_index)):
        return False
        
    #veto B mass window for background to reject signal in data
    if (not isSignal and tree.BToKee_mass[tree.BToKee_sel_index]>5. and tree.BToKee_mass[tree.BToKee_sel_index]<5.6):
        return False
        
    #if (tree.BToKmumu_cosAlpha[tree.BToKmumu_sel_index]<=0.99):
    #    return False
        
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
    
    #hist = ROOT.TH1F("nT"+str(random.random()),";#triplets;Events",50,0.5,50.5)
    #hist.SetDirectory(0)
    
    hist = ROOT.TH1F("nB"+str(random.random()),";Index of gen-matched triplet;Normalized Events",16,-0.5,15.5)
    hist.SetDirectory(0)
    
    if isSignal:
        combIndexDict = {}
    
    for ientry in range(nEvents):
        #if ientry%20000==0:
        #    print "processing ... %i/%i"%(ientry,nEvents)
        chain.GetEntry(ientry)
        if not baseSelection(chain,isSignal):
            continue
        
        selectedCombIndicesCLvtxPairs = []
        
        for icomb in range(nCombinations(chain)):
            if (combinationSelection(chain,icomb)):
                selectedCombIndicesCLvtxPairs.append([icomb,combinationCLVtx(chain,icomb)])
        if len(selectedCombIndicesCLvtxPairs)==0:
            continue
            
        #hist.Fill(len(selectedCombIndicesCLvtxPairs))
        nPassBaseSelection+=1
            
        selectedCombIndicesCLvtxPairsSorted = sorted(selectedCombIndicesCLvtxPairs,key=lambda elem: -elem[1])
        selectedCombIndicesSorted = map(lambda x:x[0],selectedCombIndicesCLvtxPairsSorted)
            
        if isSignal:
            genIndex = -1
            for icombSorted,icombIndex in enumerate(selectedCombIndicesSorted):
                if icombIndex==int(genCombination(chain)):
                    genIndex = icombSorted
                    break
            if not combIndexDict.has_key(genIndex):
                combIndexDict[genIndex] = 0
            combIndexDict[genIndex]+=1
            
            
            hist.Fill(genIndex)
        
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
    return nEvents,nPassBaseSelection,nPassSignalSelection,hist
    
   

backgroundFiles = []

for folder in [
    "/vols/cms/vc1116/BParking/ntuPROD/data_BToKmumuNtuple/PR35_BToKmumu/A1",
    #"/vols/cms/vc1116/BParking/ntuPROD/data_BToKmumuNtuple/PR35_BToKmumu/A2",
    #"/vols/cms/vc1116/BParking/ntuPROD/data_BToKmumuNtuple/PR35_BToKmumu/A3",
    #"/vols/cms/vc1116/BParking/ntuPROD/data_BToKmumuNtuple/PR35_BToKmumu/A4",
    #"/vols/cms/vc1116/BParking/ntuPROD/data_BToKmumuNtuple/PR35_BToKmumu/A5",
    #"/vols/cms/vc1116/BParking/ntuPROD/data_BToKmumuNtuple/PR35_BToKmumu/A6",
    
    #"/vols/cms/vc1116/BParking/ntuPROD/data_BToKmumuNtuple/PR35_BToKmumu/B1",
    #"/vols/cms/vc1116/BParking/ntuPROD/data_BToKmumuNtuple/PR35_BToKmumu/B2",
    #"/vols/cms/vc1116/BParking/ntuPROD/data_BToKmumuNtuple/PR35_BToKmumu/B3",
    #"/vols/cms/vc1116/BParking/ntuPROD/data_BToKmumuNtuple/PR35_BToKmumu/B4",
    #"/vols/cms/vc1116/BParking/ntuPROD/data_BToKmumuNtuple/PR35_BToKmumu/B5",
]:
    for f in os.listdir(folder):
        if re.match("\w+.root",f):
            fullFilePath = os.path.join(folder,f)

            backgroundFiles.append(fullFilePath)
            
#backgroundChain = Chain(backgroundFiles[0:60])

signalFilesMu = [
    #'/vols/cms/vc1116/BParking/ntuPROD/mc_BToKmumuNtuple/ntu_mc_BToKmumu_PR27.root',
    '/vols/cms/vc1116/BParking/ntuPROD/BPH_MC_Ntuple/ntu_PR36_BPH_MC_BuToK_ToMuMu.root'
]

        
signalChainMu = Chain(signalFilesMu)

nEvents,nPassBaseSelection,nPassSignalSelection,histSignal = calculateEfficiency(
    signalChainMu,
    baseSelection=baseSelectionMu,
    signalSelection=signalSelectionMu,
    combinationSelection=combinationSelectionMu,
    nCombinations = lambda tree:tree.nBToKmumu,
    #combinationCLVtx = lambda tree,i:-tree.BToKmumu_Chi2_vtx[i],
    combinationCLVtx = lambda tree,i:tree.BToKmumu_CL_vtx[i],
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
'''
nEvents,nPassBaseSelection,nPassSignalSelection,histBackground = calculateEfficiency(
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
histSignal.Scale(1./histSignal.Integral())
histSignal.SetLineColor(ROOT.kOrange+7)
histSignal.SetLineWidth(2)
'''
histBackground.Scale(1./histBackground.Integral())
histBackground.SetLineColor(ROOT.kViolet+1)
histBackground.SetLineWidth(2)
'''
cv = ROOT.TCanvas("cv","",800,650)
cv.SetLeftMargin(0.125)
cv.SetRightMargin(0.03)
cv.SetTopMargin(0.08)
cv.SetBottomMargin(0.115)
cv.SetLogy(1)
axis = ROOT.TH2F("axis",";#triplets;Normalized events",
    50,-0.5,15.5,40,0.01,1#1.4*max(map(lambda x:x.GetMaximum(),[histSignal,histBackground]))
)
#axis.Draw("AXIS")
histSignal.Draw("HISTSame")
#histBackground.Draw("HISTSame")
cv.Print("nIndex.pdf")


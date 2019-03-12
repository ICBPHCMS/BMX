import os
import h5py
import re
import math
from input_reader import *
import matplotlib.pyplot as plt
import numpy as np
from reference_model2 import BMXNetwork
import keras
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPadLeftMargin(0.15)
ROOT.gStyle.SetPadBottomMargin(0.12)
ROOT.gStyle.SetPadRightMargin(0.04)
ROOT.gStyle.SetPadTopMargin(0.04)
ROOT.gStyle.SetTitleOffset(1.1,'X')
ROOT.gStyle.SetTitleOffset(1.5,'Y')
ROOT.gStyle.SetTitleFont(43,'XYZ')
ROOT.gStyle.SetTitleSize(35,'XYZ')
ROOT.gStyle.SetLabelFont(43,'XYZ')
ROOT.gStyle.SetLabelSize(30,'XYZ')
ROOT.gStyle.SetLabelOffset(0.008,'X')
ROOT.gStyle.SetPadTickX(1)
ROOT.gStyle.SetTickLength(0.025,'X')
ROOT.gStyle.SetTickLength(0.015,'Y')
ROOT.gStyle.SetPadTickY(1)


import sklearn.metrics

inputCombinationFeatures = [
    'B_Lxy', 
    'B_abseta', 
    'B_cos_alpha', 
    'B_ctxy', 
    'B_pt', 
    'B_vtx_CL', 
    'B_vtx_Chi2', 
    
    'dilepton_CL_vtx', 
    'dilepton_Chi2_vtx', 
    'dilepton_Lxy', 
    'dilepton_abseta', 
    'dilepton_ctxy', 
    #'dilepton_pt', 
    #'dilepton_ptrel', 
    
    'k_deltaRJet', 
    'k_dxy_wrtVtx', 
    'k_dz', 
    'k_eta', 
    'k_mindz_wrtPU', 
    #'k_pt', 
    'k_ptrel', 
    
    #'klepton1_deltaR', 
    #'klepton1_deltaVZ', 
    'klepton1_deltaXY', 
    #'klepton1_deltaZ', 
    
    #'klepton2_deltaR', 
    #'klepton2_deltaVZ', 
    'klepton2_deltaXY', 
    #'klepton2_deltaZ', 
    
    #'dilepton_deltaR', 
    #'lepton1lepton2_deltaVZ', 
    'lepton1lepton2_deltaXY', 
    #'lepton1lepton2_deltaZ', 
    
    'lepton1_abseta', 
    'lepton1_deltaRJet', 
    'lepton1_dz', 
    'lepton1_iso', 
    #'lepton1_pt', 
    'lepton1_ptrel', 

    'lepton2_abseta', 
    'lepton2_deltaRJet', 
    'lepton2_dz', 
    'lepton2_iso',
    #'lepton2_pt', 
    'lepton2_ptrel'
    #'lepton2_isPFLepton',
]

inputGlobalFeatures = [
    'leadingSV_pt', 
    'ncombinations',
    'njets',
    'nsv',
    'leadingjet_pt'
]


dataTrain = HDF5Reader.fromFolder('../../mumut2/',"train\w+.hdf5",maxFiles=-1,nVertices=2)()
dataTest = HDF5Reader.fromFolder('../../mumut2/',"test\w+.hdf5",maxFiles=-1,nVertices=2)()


combinationFeaturesTrain = HDF5Reader.merge_combination_features(
    dataTrain,
    featureNames=inputCombinationFeatures
)
globalFeaturesTrain = HDF5Reader.merge_global_features(
    dataTrain,
    featureNames=inputGlobalFeatures
)

regressionTargetsTrain = np.stack(
    [
        dataTrain['bmass'],
        dataTrain['llmass']
    ],
    axis = 1
)
regressionTargetsTrain = regressionTargetsTrain[:,:,0]

combinationFeaturesTest = HDF5Reader.merge_combination_features(
    dataTest,
    featureNames=inputCombinationFeatures
)

globalFeaturesTest = HDF5Reader.merge_global_features(
    dataTest,
    featureNames=inputGlobalFeatures
)

#globalFeatures = HDF5Reader.merge_global_features(columns)

network = BMXNetwork(dropout_rate=0.1)
classModel = network.getClassModel(combinationFeaturesTrain,globalFeaturesTrain)
classModel.summary()

classOpt = keras.optimizers.Adam(
    lr=0.0001, 
    beta_1=0.9, 
    beta_2=0.999,
    epsilon=None, 
    decay=0.0, 
    amsgrad=False
)

classModel.compile(
    loss=['categorical_crossentropy'],
    optimizer=classOpt,
    metrics=['accuracy'],
)

fullModel = network.getFullModel(combinationFeaturesTrain,globalFeaturesTrain)
#fullModel.summary()

fullOpt = keras.optimizers.Adam(
    lr=0.001, 
    beta_1=0.9, 
    beta_2=0.999,
    epsilon=None, 
    decay=0.0, 
    amsgrad=False
)

fullModel.compile(
    loss=['categorical_crossentropy','mse'],
    optimizer=fullOpt,
    metrics=['accuracy'],
)

trainLoss = []
testLoss = []

trainAcc = []
testAcc = []

np.random.seed(12345)
batchSizeTrain = 40000
nBatchesTrain = int(math.floor(1.*combinationFeaturesTrain.shape[0]/batchSizeTrain))
batchSizeTest = 40000
nBatchesTest = int(math.floor(1.*combinationFeaturesTest.shape[0]/batchSizeTest))

nSignalTrain = np.sum(dataTrain['genIndex'][:,0:-1]) #sum over vertices
nBackgroundTrain = np.sum(dataTrain['genIndex'][:,-1])

weightsTrain = np.sum(dataTrain['genIndex'][:,0:-1],axis=1)
weightsTrain = weightsTrain*(nSignalTrain+nBackgroundTrain)/nSignalTrain + \
          (1.-weightsTrain)*(nSignalTrain+nBackgroundTrain)/nBackgroundTrain

nSignalTest = np.sum(dataTest['genIndex'][:,0:-1]) #sum over vertices
nBackgroundTest = np.sum(dataTest['genIndex'][:,-1])
weightsTest = np.sum(dataTest['genIndex'][:,0:-1],axis=1)
weightsTest = weightsTest*(nSignalTest+nBackgroundTest)/nSignalTest + \
          (1.-weightsTest)*(nSignalTest+nBackgroundTest)/nBackgroundTest

print "%i/%i train/test batches"%(nBatchesTrain,nBatchesTest)
print "%f/%f train/test signal weight"%(
    (nSignalTrain+nBackgroundTrain)/nSignalTrain,
    (nSignalTest+nBackgroundTest)/nSignalTest
)
print "%f/%f train/test background weight"%(
    (nSignalTrain+nBackgroundTrain)/nBackgroundTrain,
    (nSignalTest+nBackgroundTest)/nBackgroundTest
)


batchIndicesTrain = np.arange(combinationFeaturesTrain.shape[0])
'''
history = classModel.fit(
    [combinationFeaturesTrain],
    [dataTrain['genIndex']],
    batch_size=batchSize,
    epochs=10
)
print history.history
'''


if os.path.exists('model_weights.hdf5'):
    classModel.load_weights('model_weights.hdf5')
else:

    for epoch in range(150,250):
        #randomize input order
        np.random.shuffle(batchIndicesTrain)
        lossTrainSum = 0.
        accTrainSum = 0.
        
        for batch in range(nBatchesTrain):
            #if batch%10==0:
            #    print "Epoch %2i, batch %i/%i"%(epoch+1,batch+1,nBatchesTrain)
            result = classModel.train_on_batch(
                [
                    combinationFeaturesTrain[batchIndicesTrain[batch*batchSizeTrain:(batch+1)*batchSizeTrain]],
                    globalFeaturesTrain[batchIndicesTrain[batch*batchSizeTrain:(batch+1)*batchSizeTrain]]
                ],
                [
                    dataTrain['genIndex'][batchIndicesTrain[batch*batchSizeTrain:(batch+1)*batchSizeTrain]],
                    #regressionTargetsTrain[batchIndicesTrain[batch*batchSizeTrain:(batch+1)*batchSizeTrain]],
                ],
                sample_weight=[
                    weightsTrain[batchIndicesTrain[batch*batchSizeTrain:(batch+1)*batchSizeTrain]],
                    #weightsTrain[batchIndicesTrain[batch*batchSizeTrain:(batch+1)*batchSizeTrain]],
                ]
            )
            
        for batch in range(nBatchesTrain):
            #if batch%10==0:
            #    print "Epoch %2i, batch %i/%i"%(epoch+1,batch+1,nBatchesTrain)
            result = classModel.test_on_batch(
                [
                    combinationFeaturesTrain[batchIndicesTrain[batch*batchSizeTrain:(batch+1)*batchSizeTrain]],
                    globalFeaturesTrain[batchIndicesTrain[batch*batchSizeTrain:(batch+1)*batchSizeTrain]]
                ],
                [
                    dataTrain['genIndex'][batchIndicesTrain[batch*batchSizeTrain:(batch+1)*batchSizeTrain]]
                ],
                sample_weight=[
                    weightsTrain[batchIndicesTrain[batch*batchSizeTrain:(batch+1)*batchSizeTrain]]
                ]
            )
            lossTrainSum+=result[0]
            accTrainSum+=result[1]
            
        print epoch,lossTrainSum/nBatchesTrain,
        
        lossTestSum = 0.
        accTestSum = 0.
        for batch in range(nBatchesTest):
            #if batch%10==0:
            #    print "Epoch %2i, batch %i/%i"%(epoch+1,batch+1,nBatchesTest)
            result = classModel.test_on_batch(
                [
                    combinationFeaturesTest[batch*batchSizeTest:(batch+1)*batchSizeTest],
                    globalFeaturesTest[batch*batchSizeTest:(batch+1)*batchSizeTest]
                ],
                [
                    dataTest['genIndex'][batch*batchSizeTest:(batch+1)*batchSizeTest]
                ],
                sample_weight=[
                    weightsTest[batch*batchSizeTest:(batch+1)*batchSizeTest]
                ]
            )
            lossTestSum+=result[0]
            accTestSum+=result[1]
            
        print lossTestSum/nBatchesTest
        
        trainLoss.append(lossTrainSum/nBatchesTrain)
        trainAcc.append(1.-accTrainSum/nBatchesTrain)
        
        testLoss.append(lossTestSum/nBatchesTest)
        testAcc.append(1.-accTestSum/nBatchesTest)
        
        classModel.save_weights('model_weights_%i.hdf5'%epoch)
        
    fig = plt.figure()
    plt.plot(np.arange(len(trainLoss)),trainLoss,label='train')
    plt.plot(np.arange(len(testLoss)),testLoss,label='test')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend(loc='upper right')
    plt.savefig('loss.pdf')

    fig = plt.figure()
    plt.plot(np.arange(len(trainAcc)),trainAcc,label='train')
    plt.plot(np.arange(len(testAcc)),testAcc,label='test')
    plt.xlabel("Epoch")
    plt.ylabel("1-accuracy")
    plt.yscale("log")
    plt.legend(loc='upper right')
    plt.savefig('accuracy.pdf')

    classModel.save_weights('model_weights.hdf5')

predictionTrain = []
predictionTest = []

predictClassTrain = classModel.predict_on_batch([
    combinationFeaturesTrain,
    globalFeaturesTrain
])
predictClassTest = classModel.predict_on_batch([
    combinationFeaturesTest,
    globalFeaturesTest
])


bgEffTest1st,sigEffTest1st,thresTest1st = sklearn.metrics.roc_curve(
    dataTest['genIndex'][:,0]>0,
    predictClassTest[:,0]
)

bgEffTest2nd,sigEffTest2nd,thresTest2nd = sklearn.metrics.roc_curve(
    dataTest['genIndex'][:,1]>0,
    predictClassTest[:,1]
)


bgEffTestAll,sigEffTestAll,thresTestAll = sklearn.metrics.roc_curve(
    np.sum(dataTest['genIndex'][:,:-1],axis=1)>0,
    np.sum(predictClassTest[:,:-1],axis=1)
)

#signalEffTrainRef = np.sum(dataTrain['truth'][dataTrain['refSel'][:,0]>0])/np.sum(dataTrain['truth'])
#print signalEffTrainRef

bgEffTest1stRef = np.sum(dataTest['genIndex'][dataTest['refSel'][:,0]>0,:][:,-1])/np.sum(dataTest['genIndex'][:,-1])
sigEffTest1stRef = np.sum(dataTest['genIndex'][dataTest['refSel'][:,0]>0,:][:,0])/np.sum(dataTest['genIndex'][:,0])
print bgEffTest1stRef,sigEffTest1stRef

bgEffTest2ndRef = np.sum(dataTest['genIndex'][dataTest['refSel'][:,1]>0,:][:,-1])/np.sum(dataTest['genIndex'][:,-1])
sigEffTest2ndRef = np.sum(dataTest['genIndex'][dataTest['refSel'][:,1]>0,:][:,1])/np.sum(dataTest['genIndex'][:,1])
print bgEffTest2ndRef,sigEffTest2ndRef


fig = plt.figure()
plt.plot(sigEffTest1st,bgEffTest1st,label='1st vertex',color='#2e4fd1')
plt.plot(sigEffTest2nd,bgEffTest2nd,label='2nd vertex',color='#dd992c')
plt.plot(sigEffTest1stRef,bgEffTest1stRef,marker='.',color='#2e4fd1',markersize=10)
plt.plot(sigEffTest2ndRef,bgEffTest2ndRef,marker='.',color='#dd992c',markersize=10)
#plt.plot(sigEffTrainAll,bgEffTrainAll,label='all vertex')
plt.xlabel("Signal efficiency")
plt.ylabel("Background efficiency")
plt.yscale("log")
plt.legend(loc='upper left')
plt.savefig('roc.pdf')

isSignalTrain = np.sum(dataTrain['genIndex'][:,:-1],axis=1)>0

bgEffTrain1vs2,sigEffTrain1vs2,thresTrain1vs2 = sklearn.metrics.roc_curve(
    dataTrain['genIndex'][isSignalTrain][:,0]>0,
    predictClassTrain[isSignalTrain][:,0]/(1-predictClassTrain[isSignalTrain][:,2])
)

fig = plt.figure()
plt.plot(sigEffTrain1vs2,bgEffTrain1vs2,label='1st vertex',color='#2e4fd1')
#plt.plot(sigEffTrainAll,bgEffTrainAll,label='all vertex')
plt.xlabel("1 vertex efficiency")
plt.ylabel("2 vertex efficiency")
plt.yscale("log")
plt.legend(loc='upper left')
plt.savefig('roc_1vs2.pdf')


isDataTest = dataTest['genIndex'][:,-1]>0
bmassTestRef = dataTest['bmass'][(isDataTest)*(dataTest['refSel'][:,0]>0),0]
llmassTestRef = dataTest['llmass'][(isDataTest)*(dataTest['refSel'][:,0]>0),0]

sumBkgRef = np.sum(bmassTestRef[bmassTestRef>5.5])

bMassData = dataTest['bmass'][isDataTest,0]
predictionData = predictClassTest[isDataTest,0]

sumBkgPred=-1
thresSel = -1
for thres in 1-np.logspace(-4,0,1000):
    sumBkgPred = np.sum(bMassData[(bMassData>5.5)*(predictionData>thres)])
    if sumBkgPred>sumBkgRef:
        thresSel = thres
        break

print "matched thres: ",thresSel,sumBkgRef,sumBkgPred


def fillHist(hist,array):
    for i in range(array.shape[0]):
        hist.Fill(array[i])
        
bmassTestRef = dataTest['bmass'][(isDataTest)*(dataTest['llmass'][:,0]<2.5)*(dataTest['refSel'][:,0]>0),0]
bmassTestDisc = dataTest['bmass'][(isDataTest)*(dataTest['llmass'][:,0]<2.5)*(predictClassTest[:,0]>thresSel),0]

llmassTestDisc = dataTest['llmass'][(isDataTest)*(predictClassTest[:,0]>thresSel),0]


histBmassRef = ROOT.TH1F("histRef",";B mass (GeV); Events",60,4.,7)
histBmassRef.Sumw2()
fillHist(histBmassRef,bmassTestRef)
cv = ROOT.TCanvas("cv","",800,700)
histBmassRef.SetMarkerStyle(20)
histBmassRef.SetMarkerSize(1.25)
histBmassRef.SetMarkerStyle(20)
histBmassRef.SetLineColor(ROOT.kBlack)
histBmassRef.SetMarkerColor(ROOT.kBlack)


histBmassDisc = ROOT.TH1F("histDisc",";B mass (GeV); Events",60,4.,7)
histBmassDisc.Sumw2()
fillHist(histBmassDisc,bmassTestDisc)
histBmassDisc.SetMarkerStyle(20)
histBmassDisc.SetMarkerSize(1.25)
histBmassDisc.SetMarkerStyle(20)
histBmassDisc.SetLineColor(ROOT.kOrange+7)
histBmassDisc.SetMarkerColor(ROOT.kOrange+7)

cvB = ROOT.TCanvas("cv","",800,700)

axisB = ROOT.TH2F("axisB",";B mass (GeV); Events",
    50,4.,7,
    50,0,1.2*max(map(lambda h:h.GetMaximum(),[histBmassRef,histBmassDisc]))
)
axisB.Draw("AXIS")

histBmassRef.Draw("SamePE")
histBmassDisc.Draw("SamePE")

ROOT.gPad.RedrawAxis()

cvB.Print("bmass.pdf")




histllmassRef = ROOT.TH1F("histRef",";#mu#mu mass (GeV); Events",60,1.,4)
histllmassRef.Sumw2()
fillHist(histllmassRef,llmassTestRef)
cv = ROOT.TCanvas("cvb","",800,700)
histllmassRef.SetMarkerStyle(20)
histllmassRef.SetMarkerSize(1.25)
histllmassRef.SetMarkerStyle(20)
histllmassRef.SetLineColor(ROOT.kBlack)
histllmassRef.SetMarkerColor(ROOT.kBlack)


histllmassDisc = ROOT.TH1F("histDisc",";#mu#mu mass (GeV); Events",60,1.,4)
histllmassDisc.Sumw2()
fillHist(histllmassDisc,llmassTestDisc)
histllmassDisc.SetMarkerStyle(20)
histllmassDisc.SetMarkerSize(1.25)
histllmassDisc.SetMarkerStyle(20)
histllmassDisc.SetLineColor(ROOT.kOrange+7)
histllmassDisc.SetMarkerColor(ROOT.kOrange+7)

cvll = ROOT.TCanvas("cvll","",800,700)

axisll = ROOT.TH2F("axisll",";#mu#mu mass (GeV); Events",
    50,1.,4,
    50,0,1.2*max(map(lambda h:h.GetMaximum(),[histllmassRef,histllmassDisc]))
)
axisll.Draw("AXIS")

histllmassRef.Draw("SamePE")
histllmassDisc.Draw("SamePE")

ROOT.gPad.RedrawAxis()

cvll.Print("llmass.pdf")

'''
for batch in range(nBatchesTrain):
    result = classModel.predict_on_batch(
        [
            combinationFeaturesTrain[batch*batchSizeTrain:(batch+1)*batchSizeTrain]
        ],
    )

for batch in range(nBatchesTest):
    result = classModel.predict_on_batch(
        [
            combinationFeaturesTest[batch*batchSizeTest:(batch+1)*batchSizeTest]
        ],
    )
'''







import tensorflow as tf
import keras
from keras import backend as K
import numpy
import math
import h5py
import random
import ROOT
import re
import os

from sklearn import metrics

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

class Generator:
    def __init__(self, fileList):
        self.fileList = fileList
        
        self.data = []
     
        self.nSignal = 0
        self.nBackground = 0
     
        #since dataset is small load everything into memory 
        for f in self.fileList:
            with h5py.File(f, 'r') as hf:
                keys = hf.keys()
                for index in keys:
                    truthArray = numpy.array(hf[index]["truth"])
                    featureArray = numpy.array(hf[index]["features"])
                    scaleArray = numpy.array(hf[index]["scales"])
                    genIndexArray = numpy.array(hf[index]["genIndex"])
                    if (not numpy.all(numpy.isfinite(truthArray))) or \
                       (not numpy.all(numpy.isfinite(featureArray))) or \
                       (not numpy.all(numpy.isfinite(scaleArray))) or \
                       (not numpy.all(numpy.isfinite(genIndexArray))):
                        print "Warning - encountered inf/nan in training data"
                        continue
                        
                    if truthArray[0]>0.5:
                        self.nSignal+=1
                    else:
                        self.nBackground+=1
                        
                    self.data.append({
                        "truth":numpy.expand_dims((truthArray),axis=0),
                        "features":numpy.expand_dims((featureArray),axis=0),
                        "scales":numpy.expand_dims((scaleArray),axis=0),
                        "genIndex":numpy.expand_dims((genIndexArray),axis=0),
                    })
        self.SBratio = 1.*self.nSignal/self.nBackground
        
                    

    def __call__(self,batchSize=1):
        random.shuffle(self.data)
        for i in range(len(self.data)/batchSize):
            truthList = []
            weightList = []
            featureList = []
            scalesList = []
            genIndexList = []
            
            for b in range(batchSize):
                dataBatch = self.data[i*batchSize+b]
                truthList.append(dataBatch["truth"])
                weight = dataBatch["truth"][0,:]
                weight = (1.-weight)*self.SBratio+weight
                weightList.append(weight)
                
                featureList.append(dataBatch["features"])
                scalesList.append(dataBatch["scales"])
                genIndexList.append(dataBatch["genIndex"])
            
            batchDict = {
                "truth":truthList,
                "features":featureList,
                "weight":weightList,
                "scales":scalesList,
                "genIndex":genIndexList
            }
            yield batchDict 
        
        
                    
class Network:
    def __init__(self,regLoss=1e-6):
        self.convLayers = [
            keras.layers.Conv1D(32, 1, strides=1, activation=keras.layers.LeakyReLU(alpha=0.1), 
                kernel_regularizer=keras.regularizers.l2(regLoss)
            ),
            #drop same features (=filters) for all combinations
            keras.layers.Dropout(0.2,noise_shape=[1,1,32]),
            keras.layers.Conv1D(12, 1, strides=1, activation='tanh', 
                kernel_regularizer=keras.regularizers.l2(regLoss)
            ),
        ]
        
        self.lstmLayers = [
            keras.layers.LSTM(50, activation='tanh', recurrent_activation='hard_sigmoid',
                kernel_regularizer=keras.regularizers.l2(regLoss),
                implementation=2,
                go_backwards=True
            ),
            keras.layers.Dropout(0.1),
        ]
        '''
        self.lstmCombs = [
            keras.layers.LSTM(
                20, activation='tanh', recurrent_activation='hard_sigmoid',
                kernel_regularizer=keras.regularizers.l2(regLoss),
                implementation=2,
                return_sequences = True,
                go_backwards=True
            ),
            keras.layers.Dropout(0.1,noise_shape=[1,1,20]),
            keras.layers.LSTM(
                20, activation=None, recurrent_activation='hard_sigmoid',
                kernel_regularizer=keras.regularizers.l2(regLoss),
                implementation=2,
                return_sequences = True,
                go_backwards=False
            ),
            keras.layers.Dropout(0.1,noise_shape=[1,1,20]),
            keras.layers.Conv1D(1, 1, strides=1, activation=None, 
                kernel_regularizer=keras.regularizers.l2(regLoss)
            ),
        ]
        '''
        self.bypassLayers = [
            keras.layers.Lambda(lambda l:l[:,0,:]),
            #drop complete bypass
            keras.layers.Dropout(0.8,noise_shape=[1,1]),
        ]
        
        self.predictLayers = [
            keras.layers.Dense(50, activation=keras.layers.LeakyReLU(alpha=0.1),
                kernel_regularizer=keras.regularizers.l2(regLoss)
            ),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(20, activation=keras.layers.LeakyReLU(alpha=0.1),
                kernel_regularizer=keras.regularizers.l2(regLoss)
            ),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(1, activation='sigmoid',
                kernel_regularizer=keras.regularizers.l2(regLoss)
            )
        ]
        '''
        self.scaleLayers = [
            keras.layers.Conv1D(64, 1, strides=1, activation=keras.layers.LeakyReLU(alpha=0.1), 
                kernel_regularizer=keras.regularizers.l2(regLoss)
            ),
            keras.layers.Conv1D(2, 1, strides=1, activation=None, 
                kernel_regularizer=keras.regularizers.l2(regLoss)
            ),
        ]
        
        self.combConvLayers = [
            keras.layers.Conv1D(32, 1, strides=1, activation=keras.layers.LeakyReLU(alpha=0.1), 
                kernel_regularizer=keras.regularizers.l2(regLoss)
            ),
            #drop same features (=filters) for all combinations
            keras.layers.Dropout(0.1,noise_shape=[1,1,32]),
            keras.layers.Conv1D(12, 1, strides=1, activation=None, 
                kernel_regularizer=keras.regularizers.l2(regLoss)
            ),
        ]
        '''
    def getFeatures(self,x):
        conv = self.convLayers[0](x)
        for layer in self.convLayers[1:]:
            conv = layer(conv)
        return conv

    def getDiscriminant(self,x):
        conv = self.getFeatures(x)
        lstm = self.lstmLayers[0](conv)
        for layer in self.lstmLayers[1:]:
            lstm = layer(lstm)
        bypass = self.bypassLayers[0](conv)
        for layer in self.bypassLayers[1:]:
            bypass = layer(bypass)
        features = keras.layers.Concatenate()([lstm,bypass])
        predict = self.predictLayers[0](features)
        for layer in self.predictLayers[1:]:
            predict = layer(predict)
        return predict
        
    '''
    def predictBestCombination(self,x):
        comb = self.combConvLayers[0](x)
        for layer in self.combConvLayers[1:]:
            comb = layer(comb)
            
        #TODO: use result of discrimant network as extra row
        comb = keras.layers.Lambda(lambda l: tf.concat([l[:,:,:],l[:,0:1,:]],axis=1))(comb)
        comb = self.lstmCombs[0](comb)
        for layer in self.lstmCombs[1:]:
            comb = layer(comb)
        comb = keras.layers.Lambda(lambda l: l[:,:,0])(comb)
        comb = keras.layers.Softmax(axis=-1)(comb)
        print comb
        return comb
        
    
        
    def getScalePrediction(self,x):
        scales = self.getFeatures(x)
        #reverse gradient
        scales = keras.layers.Lambda(lambda l:tf.stop_gradient(l+l)-l)(scales)
        for layer in self.scaleLayers:
            scales = layer(scales)
        
        return scales
    '''
    
class NetworkDense:
    def __init__(self,regLoss=1e-6):
        
        self.bypassLayers = [
            keras.layers.Lambda(lambda l:l[:,0,:]),
        ]
        
        self.predictLayers = [
            keras.layers.Dense(50, activation=keras.layers.LeakyReLU(alpha=0.1),
                kernel_regularizer=keras.regularizers.l2(regLoss)
            ),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(50, activation=keras.layers.LeakyReLU(alpha=0.1),
                kernel_regularizer=keras.regularizers.l2(regLoss)
            ),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(50, activation=keras.layers.LeakyReLU(alpha=0.1),
                kernel_regularizer=keras.regularizers.l2(regLoss)
            ),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(50, activation=keras.layers.LeakyReLU(alpha=0.1),
                kernel_regularizer=keras.regularizers.l2(regLoss)
            ),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(1, activation='sigmoid',
                kernel_regularizer=keras.regularizers.l2(regLoss)
            )
        ]
        
    
    def getDiscriminant(self,x):
        bypass = self.bypassLayers[0](x)
        for layer in self.bypassLayers[1:]:
            bypass = layer(bypass)
        predict = self.predictLayers[0](bypass)
        for layer in self.predictLayers[1:]:
            predict = layer(predict)
        return predict


def drawROC(name,sigEff,bgEff,signalName="Signal",backgroundName="Background",auc=None,style=1):
    cv = ROOT.TCanvas("cv_roc"+str(random.random()),"",800,700)
    cv.SetPad(0.0, 0.0, 1.0, 1.0)
    cv.SetFillStyle(4000)

    cv.SetBorderMode(0)
    #cv.SetGridx(True)
    #cv.SetGridy(True)

    #For the frame:
    cv.SetFrameBorderMode(0)
    cv.SetFrameBorderSize(1)
    cv.SetFrameFillColor(0)
    cv.SetFrameFillStyle(0)
    cv.SetFrameLineColor(1)
    cv.SetFrameLineStyle(1)
    cv.SetFrameLineWidth(1)

    # Margins:
    cv.SetLeftMargin(0.125)
    cv.SetRightMargin(0.03)
    cv.SetTopMargin(0.08)
    cv.SetBottomMargin(0.115)

    # For the Global title:
    cv.SetTitle("")

    # For the axis:
    cv.SetTickx(1)  # To get tick marks on the opposite side of the frame
    cv.SetTicky(1)

    cv.SetLogy(1)

    axis=ROOT.TH2F("axis" + str(random.random()),";" + signalName + " efficiency;" + backgroundName + " efficiency", 
        50, 0, 1.0, 50, 10**-7, 1.0
    )
    axis.GetYaxis().SetNdivisions(508)
    axis.GetXaxis().SetNdivisions(508)
    axis.GetXaxis().SetTickLength(0.015/(1-cv.GetLeftMargin()-cv.GetRightMargin()))
    axis.GetYaxis().SetTickLength(0.015/(1-cv.GetTopMargin()-cv.GetBottomMargin()))
    #axis.GetYaxis().SetNoExponent(True)
    axis.Draw("AXIS")

    #### draw here
    graphF = ROOT.TGraph(len(sigEff),numpy.array(sigEff),numpy.array(bgEff))
    graphF.SetLineWidth(0)
    graphF.SetFillColor(ROOT.kOrange+10)
    #graphF.Draw("SameF")

    graphL = ROOT.TGraph(len(sigEff),numpy.array(sigEff),numpy.array(bgEff))
    graphL.SetLineColor(ROOT.kOrange+7)
    graphL.SetLineWidth(3)
    graphL.SetLineStyle(style)
    graphL.Draw("SameL")

    ROOT.gPad.RedrawAxis()
    
    markerCutBased = ROOT.TMarker(0.336,0.00044,20)
    markerCutBased.SetMarkerSize(1.4)
    markerCutBased.SetMarkerColor(ROOT.kViolet+2)
    markerCutBased.Draw("Same")
    
    pMarker=ROOT.TPaveText(0.3,0.00044,0.3,0.00044)
    pMarker.SetFillStyle(0)
    pMarker.SetBorderSize(0)
    pMarker.SetTextFont(63)
    pMarker.SetTextSize(25)
    pMarker.SetTextColor(ROOT.kViolet+2)
    pMarker.SetTextAlign(32)
    pMarker.AddText("cut-based")
    pMarker.Draw("Same")

    pCMS=ROOT.TPaveText(cv.GetLeftMargin(),0.94,cv.GetLeftMargin(),0.94,"NDC")
    pCMS.SetFillColor(ROOT.kWhite)
    pCMS.SetBorderSize(0)
    pCMS.SetTextFont(63)
    pCMS.SetTextSize(30)
    pCMS.SetTextAlign(11)
    pCMS.AddText("CMS")
    pCMS.Draw("Same")

    pPreliminary=ROOT.TPaveText(cv.GetLeftMargin()+0.095,0.94,cv.GetLeftMargin()+0.095,0.94,"NDC")
    pPreliminary.SetFillColor(ROOT.kWhite)
    pPreliminary.SetBorderSize(0)
    pPreliminary.SetTextFont(53)
    pPreliminary.SetTextSize(30)
    pPreliminary.SetTextAlign(11)
    pPreliminary.AddText("Simulation")
    pPreliminary.Draw("Same")

    if auc:
        pAUC=ROOT.TPaveText(1-cv.GetRightMargin(),0.94,1-cv.GetRightMargin(),0.94,"NDC")
        pAUC.SetFillColor(ROOT.kWhite)
        pAUC.SetBorderSize(0)
        pAUC.SetTextFont(43)
        pAUC.SetTextSize(30)
        pAUC.SetTextAlign(31)
        pAUC.AddText("AUC: % 4.1f %%" % (auc*100.0))
        pAUC.Draw("Same")

    cv.Update()
    cv.Print(name+".pdf")
    cv.Print(name+".root")
    
def drawDist(name,signalHists,backgroundHists):
    cv = ROOT.TCanvas("cv_dist"+str(random.random()),"",800,700)
    cv.SetPad(0.0, 0.0, 1.0, 1.0)
    cv.SetFillStyle(4000)

    cv.SetBorderMode(0)
    #cv.SetGridx(True)
    #cv.SetGridy(True)

    #For the frame:
    cv.SetFrameBorderMode(0)
    cv.SetFrameBorderSize(1)
    cv.SetFrameFillColor(0)
    cv.SetFrameFillStyle(0)
    cv.SetFrameLineColor(1)
    cv.SetFrameLineStyle(1)
    cv.SetFrameLineWidth(1)

    # Margins:
    cv.SetLeftMargin(0.125)
    cv.SetRightMargin(0.03)
    cv.SetTopMargin(0.08)
    cv.SetBottomMargin(0.115)

    # For the Global title:
    cv.SetTitle("")

    # For the axis:
    cv.SetTickx(1)  # To get tick marks on the opposite side of the frame
    cv.SetTicky(1)

    cv.SetLogy(1)
    
    for h in signalHists+backgroundHists:
        h.Scale(1.*h.GetNbinsX()/h.Integral())

    ymax = max(map(lambda h: h.GetMaximum(),signalHists+backgroundHists))

    axis=ROOT.TH2F("axis" + str(random.random()),";NN discriminant;Normalized events", 
        50, 0, 1.0, 50, 0.06, math.exp(1.1*math.log(ymax))
    )
    axis.GetYaxis().SetNdivisions(508)
    axis.GetXaxis().SetNdivisions(508)
    axis.GetXaxis().SetTickLength(0.015/(1-cv.GetLeftMargin()-cv.GetRightMargin()))
    axis.GetYaxis().SetTickLength(0.015/(1-cv.GetTopMargin()-cv.GetBottomMargin()))
    #axis.GetYaxis().SetNoExponent(True)
    axis.Draw("AXIS")

    signalHists[0].SetLineWidth(3)
    signalHists[0].SetLineColor(ROOT.kOrange+7)
    signalHists[0].Draw("HISTSAME")
    
    signalHists[1].SetLineWidth(2)
    signalHists[1].SetMarkerColor(ROOT.kOrange-3)
    signalHists[1].SetLineColor(ROOT.kOrange-3)
    signalHists[1].SetMarkerStyle(20)
    signalHists[1].SetMarkerSize(1.2)
    signalHists[1].Draw("SAMEPE")
    
    backgroundHists[0].SetLineWidth(3)
    backgroundHists[0].SetLineColor(ROOT.kAzure-5)
    backgroundHists[0].Draw("HISTSAME")
    
    backgroundHists[1].SetLineWidth(2)
    backgroundHists[1].SetMarkerColor(ROOT.kAzure-4)
    backgroundHists[1].SetLineColor(ROOT.kAzure-4)
    backgroundHists[1].SetMarkerStyle(20)
    backgroundHists[1].SetMarkerSize(1.2)
    backgroundHists[1].Draw("SAMEPE")

    ROOT.gPad.RedrawAxis()

    pCMS=ROOT.TPaveText(cv.GetLeftMargin(),0.94,cv.GetLeftMargin(),0.94,"NDC")
    pCMS.SetFillColor(ROOT.kWhite)
    pCMS.SetBorderSize(0)
    pCMS.SetTextFont(63)
    pCMS.SetTextSize(30)
    pCMS.SetTextAlign(11)
    pCMS.AddText("CMS")
    pCMS.Draw("Same")

    pPreliminary=ROOT.TPaveText(cv.GetLeftMargin()+0.095,0.94,cv.GetLeftMargin()+0.095,0.94,"NDC")
    pPreliminary.SetFillColor(ROOT.kWhite)
    pPreliminary.SetBorderSize(0)
    pPreliminary.SetTextFont(53)
    pPreliminary.SetTextSize(30)
    pPreliminary.SetTextAlign(11)
    pPreliminary.AddText("Simulation")
    pPreliminary.Draw("Same")

    cv.Update()
    cv.Print(name+".pdf")
    cv.Print(name+".root")
        
        
basepath = "/vols/cms/mkomm/BPH/training/"

trainFiles = []
testFiles = []

for l in os.listdir(basepath):
    filePath = os.path.join(basepath,l)
    if re.match("train_\w+.hdf5",l):
        try:
            f = h5py.File(filePath, 'r')
            trainFiles.append(filePath)
            f.close()
        except Exception,e:
            print "cannot open file",filePath," -> skip"
            
    if re.match("test_\w+.hdf5",l):
        try:
            f = h5py.File(filePath, 'r')
            testFiles.append(filePath)
            f.close()
        except Exception,e:
            print "cannot open file",filePath," -> skip"
        
#trainFiles = trainFiles[0:10]
#testFiles = testFiles[0:10]
        
print "found ",len(trainFiles),"/",len(testFiles)," train/test files"
#sys.exit(1)
#net = Network()
net = NetworkDense()
                   
inputList = []
outputDiscriminantList = []
#outputBestCombinationList = []
#outputScalesList = []

batchSize = 150

for i in range(batchSize):
    if (i+1)%10==0:
        print "allocating network %i/%i"%(i+1,batchSize)
    inputs = keras.layers.Input(batch_shape=(1,None,22))
    prediction = net.getDiscriminant(inputs)
    #bestComb = net.predictBestCombination(inputs)
    #outputBestCombinationList.append(bestComb)
    #scales = net.getScalePrediction(inputs)
    inputList.append(inputs)
    outputDiscriminantList.append(prediction)
    
    #outputScalesList.append(scales)
    


model = keras.models.Model(inputs=inputList,outputs=outputDiscriminantList)
opt = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(
    opt, 
    loss=[keras.losses.binary_crossentropy]*batchSize,
    metrics=[],
    loss_weights=[1.]*batchSize
)
    
modeltest = keras.models.Model(inputs=inputList[0],outputs=[outputDiscriminantList[0]])
modeltest.summary()


outputFolder = "result_dense"

fstat = open(os.path.join(outputFolder,"model_stat.txt"),"w")
fstat.close()

print "-"*70
trainBatch = Generator(trainFiles)
testBatch = Generator(testFiles)
print "Signal: %i/%i (train/test)"%(trainBatch.nSignal,testBatch.nSignal)
print "Background: %i/%i (train/test)"%(trainBatch.nBackground,testBatch.nBackground)
print "S/B: %.2f%%/%.2f%% (train/test)"%(100.*trainBatch.nSignal/trainBatch.nBackground,100.*testBatch.nSignal/testBatch.nBackground)
print "-"*70

    
previousLoss = 1e10
for epoch in range(1,51):
    if epoch>1:
        model.load_weights(os.path.join(outputFolder,"weights_%i.hdf5"%(epoch-1)))

    

    stepTrain = 0 
    totalLossTrain = 0.
    #totalLossCombTrain = 0.
    nAccTrain = 0
    nAccSignalTrain = 0
    #nAccCombTrain = 0
    nSignalTrain = 0
    
    histSignalTrain = ROOT.TH1F("signalTrain"+str(epoch)+str(random.random()),"",10000,0,1)
    histSignalTrain.Sumw2()
    histBackgroundTrain = ROOT.TH1F("backgroundTrain"+str(epoch)+str(random.random()),"",10000,0,1)
    histBackgroundTrain.Sumw2()
    
    classLossFct = model.total_loss #includes also regularization loss
    inputGradients = tf.gradients(classLossFct,model.inputs)
    
    sess = K.get_session()
    
    for batch in trainBatch(batchSize):
        if (epoch==0 and stepTrain>10) or (epoch>0):
        
            feedDict = {
                K.learning_phase(): 0
            }
            for i in range(len(model.outputs)):
                feedDict[model.targets[i]] = batch["truth"][i]
                feedDict[model.sample_weights[i]] = numpy.ones(batch["truth"][i].shape[0])
            
            for i in range(len(model.inputs)):
                feedDict[model.inputs[i]] = batch["features"][i]

            classLossVal,inputGradientsVal = sess.run([classLossFct,inputGradients],feed_dict=feedDict)         
            
            direction = numpy.abs(numpy.random.normal(0,0.9,len(model.inputs)))+0.1
            for i in range(len(model.inputs)):
                batch["features"][i]+=direction[i]*inputGradientsVal[i]
            
    
    
        stepTrain += 1
        result = model.train_on_batch(batch["features"],batch["truth"],sample_weight=batch["weight"])
        predict = model.predict_on_batch(batch["features"])
        loss = sum(result[1:batchSize+1])/batchSize
        #lossComb = sum(result[batchSize+1:2*batchSize+1])/batchSize
        totalLossTrain+=loss
        #totalLossCombTrain+=lossComb 
        
        Nbatch = len(batch["truth"])

        for i in range(Nbatch):
            if batch["truth"][i][0][0]>0.5:
                nSignalTrain+=1
                histSignalTrain.Fill(predict[i])
                if predict[i]>0.5:
                    nAccSignalTrain+=1
                    nAccTrain+=1

            else:
                histBackgroundTrain.Fill(predict[i])
                if predict[i]<0.5:
                    nAccTrain+=1

                    
        if stepTrain%10==0:
            print "Training step %i-%i: loss=%.4f, acc=%.3f%%, accSignal=%.3f%%"%(
                epoch,stepTrain,loss,
                100.*nAccTrain/stepTrain/batchSize,
                100.*nAccSignalTrain/nSignalTrain,
            )


    model.save_weights(os.path.join(outputFolder,"weights_%i.hdf5"%epoch))

    stepTest = 0 
    totalLossTest = 0.
    #totalLossCombTest = 0.
    nAccTest = 0
    nAccSignalTest = 0
    #nAccCombTest = 0
    nSignalTest = 0
    
    histSignalTest = ROOT.TH1F("signalTest"+str(epoch)+str(random.random()),"",10000,0,1)
    histSignalTest.Sumw2()
    histBackgroundTest = ROOT.TH1F("backgroundTest"+str(epoch)+str(random.random()),"",10000,0,1)
    histBackgroundTest.Sumw2()
    
    scoreTest = []
    truthTest = []
    
    for batch in testBatch(batchSize):
        stepTest += 1
        result = model.test_on_batch(batch["features"],batch["truth"],sample_weight=batch["weight"])
        predict = model.predict_on_batch(batch["features"])
        loss = sum(result[1:batchSize+1])/batchSize
        #lossComb = sum(result[batchSize+1:2*batchSize+1])/batchSize
        totalLossTest+=loss
        #totalLossCombTest+=lossComb
        
        Nbatch = len(batch["truth"])
        
        for i in range(Nbatch):
            scoreTest.append(predict[i][0][0])
            if batch["truth"][i][0][0]>0.5:
                nSignalTest+=1
                histSignalTest.Fill(predict[i][0][0])
                
                
                truthTest.append(1)
                if predict[i][0][0]>0.5:
                    nAccSignalTest+=1
                    nAccTest+=1
                
                
                
            else:
                truthTest.append(0)
            
                histBackgroundTest.Fill(predict[i][0][0])
                if predict[i][0][0]<0.5:
                    nAccTest+=1
                    
            
        
        if stepTest%10==0:
            print "Testing step %i-%i: loss=%.4f, acc=%.3f%%, accSignal=%.3f%%"%(
                epoch,stepTest,loss,
                100.*nAccTest/stepTest/batchSize,
                100.*nAccSignalTest/nSignalTest,
            )
           
            
    bgEffTest,sigEffTest,thres = metrics.roc_curve(truthTest, scoreTest)
    aucTest = metrics.auc(bgEffTest,sigEffTest)
    #sigEffTest, bgRejTest, bgEffTest = getROC(histSignalTest,histBackgroundTest)
    #aucTest = getAUC(sigEffTest, bgRejTest)
    drawROC(os.path.join(outputFolder,"roc_epoch%i"%epoch),sigEffTest,bgEffTest,auc=aucTest)
    
    histSignalTrain.Rebin(500)
    histBackgroundTrain.Rebin(500)
    
    histSignalTest.Rebin(500)
    histBackgroundTest.Rebin(500)
    
    drawDist(
        os.path.join(outputFolder,"dist_epoch%i"%epoch),
        [histSignalTrain,histSignalTest],
        [histBackgroundTrain,histBackgroundTest]
    )
        
    avgLossTrain = totalLossTrain/stepTrain
    avgLossTest = totalLossTest/stepTest
        
    print "-"*70
    lr = float(K.get_value(model.optimizer.lr))
    print "Epoch %i summary: lr=%.3e, loss=%.3f/%.3f, acc=%.2f%%/%.2f%%, accSignal=%.2f%%/%.2f%% (train/test), AUC=%.2f%%"%(
        epoch,
        lr,
        avgLossTrain,avgLossTest,
        100.*nAccTrain/stepTrain/batchSize,100.*nAccTest/stepTest/batchSize,
        100.*nAccSignalTrain/nSignalTrain,100.*nAccSignalTest/nSignalTest,
        100.*aucTest
    ) 
    
    fstat = open(os.path.join(outputFolder,"model_stat.txt"),"a")
    fstat.write("%11.4e,%8.4f,%8.4f,%8.4f,%8.4f,%8.4f,%8.4f,%8.4f\n"%(
        lr,avgLossTrain,avgLossTest,
        100.*nAccTrain/stepTrain/batchSize,100.*nAccTest/stepTest/batchSize,
        100.*nAccSignalTrain/nSignalTrain,100.*nAccSignalTest/nSignalTest,
        100.*aucTest
    ))
    fstat.close()
    
    
    if (avgLossTrain>previousLoss):
        lr *= 0.8
        K.set_value(model.optimizer.lr, lr)
        print "Reducing learning rate to: %.4e"%lr
        previousLoss = 0.5*(previousLoss+avgLossTrain)
    else:
        previousLoss = avgLossTrain
    print "-"*70
    
    
    
    

    

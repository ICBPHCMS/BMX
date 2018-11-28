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
import sys
from scipy.interpolate import interp1d

from sklearn import metrics

from network_dense import Network

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

                               
def readFileMultiEntryAhead(filenameQueue, nBatches=200):
    reader = tf.TFRecordReader(options=tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP))
    key, rawDataBatch = reader.read_up_to(filenameQueue,nBatches) 
    #key is just a dataset identifier set by the reader
    #rawData is of type string
    
    ncombinations = 50
    nfeatures = 25
    nscales = 2
    nveto = 1
    featureList = {
        'truth': tf.FixedLenFeature([1], tf.float32),
        'features': tf.FixedLenFeature([ncombinations*nfeatures], tf.float32),
        'genIndex': tf.FixedLenFeature([ncombinations+1], tf.float32),
        'scales': tf.FixedLenFeature([ncombinations*nscales], tf.float32),
        'veto': tf.FixedLenFeature([1], tf.float32),
        'refSel': tf.FixedLenFeature([1], tf.float32),
    }
    
    parsedDataBatch = tf.parse_example(rawDataBatch, features=featureList)
    
    parsedDataBatch['features']=tf.reshape(parsedDataBatch['features'],[-1,ncombinations,nfeatures])
    parsedDataBatch['scales']=tf.reshape(parsedDataBatch['scales'],[-1,ncombinations,nscales])
    return parsedDataBatch
                        
                        
def input_pipeline(files, batchSize,repeat=1):
    with tf.device('/cpu:0'):
        fileListQueue = tf.train.string_input_producer(
                files, num_epochs=repeat, shuffle=True)

        readers = []
        maxThreads = 3
        #if os.env.has_key('OMP_NUM_THREADS') and int(os.env['OMP_NUM_THREADS'])>0 and int(os.env['OMP_NUM_THREADS'])<maxThreads:
        #    maxThreads = int(os.env['OMP_NUM_THREADS'])
        for _ in range(min(1+int(len(files)/2.), maxThreads)):
            reader_batch = max(1,int(batchSize/20.))
            readers.append(readFileMultiEntryAhead(fileListQueue,reader_batch))
            
        minAfterDequeue = batchSize * 2
        capacity = minAfterDequeue + 3*batchSize
        batch = tf.train.shuffle_batch_join(
            readers,
            batch_size=batchSize,
            capacity=capacity,
            min_after_dequeue=minAfterDequeue,
            enqueue_many=True  # requires to read examples in batches!
        )
        
        return batch
        
def fitMass(name,massHist,title=""):
    w = ROOT.RooWorkspace("w"+str(name)+str(random.random()))
    w.factory("x[4.5, 6.5]")
    w.factory("nbackground[10000, 0, 100000]")
    w.factory("nsignal[100, 0.0, 10000]")
    
    w.factory("Gaussian::smodel(x,mu[5.3,5.1,5.5],sigma[0.05,0.015,0.15])")
    
    smodel = w.pdf("smodel")
    
    w.factory("Exponential::bmodel(x,tau[-2,-3,0])")
    bmodel = w.pdf("bmodel")
    
    w.factory("SUM::model(nbackground * bmodel, nsignal * smodel)")
    model = w.pdf("model")
    
    hBMass = ROOT.RooDataHist("hBMass", "hBMass", ROOT.RooArgList(w.var("x")), massHist)
    #w.Print()

    r = model.fitTo(hBMass, ROOT.RooFit.Minimizer("Minuit2"),ROOT.RooFit.Save(True))
    #r.Print()
    
    
    plot = w.var("x").frame()
    plot.SetXTitle("K#mu#mu mass (GeV)")
    plot.SetTitle("")
    hBMass.plotOn(plot)
    model.plotOn(plot)
    model.plotOn(plot, ROOT.RooFit.Components("bmodel"),ROOT.RooFit.LineStyle(ROOT.kDashed))
    model.plotOn(plot, ROOT.RooFit.Components("smodel"),ROOT.RooFit.LineColor(ROOT.kRed))
    
    nsignal = r.floatParsFinal().find("nsignal").getValV()
    nsignalErr = r.floatParsFinal().find("nsignal").getError()
    
    nbkg = r.floatParsFinal().find("nbackground").getValV()
    nbkgErr = r.floatParsFinal().find("nbackground").getError()
    
    meanVal = r.floatParsFinal().find("mu").getValV()
    sigmaVal = r.floatParsFinal().find("sigma").getValV()

    w.var("x").setRange("signalRange", meanVal - 3.*sigmaVal, meanVal + 3.*sigmaVal)

    xargs = ROOT.RooArgSet(w.var("x"))
    bkgIntegralRange = w.pdf("bmodel").createIntegral(xargs, ROOT.RooFit.NormSet(xargs), ROOT.RooFit.Range("signalRange")).getVal()
    
    cv = ROOT.TCanvas("cvw"+str(name)+str(random.random()),"",800,700)
    cv.SetMargin(0.16,0.05,0.15,0.1)
    plot.Draw()
    pText = ROOT.TPaveText(1-cv.GetRightMargin(),0.94,1-cv.GetRightMargin(),0.94,"NDC")
    pText.SetBorderSize(0)
    pText.SetFillColor(ROOT.kWhite)
    pText.SetTextSize(28)
    pText.SetTextFont(43)
    pText.SetTextAlign(32)
    pText.AddText(0.65,0.9, "%s S: %.2f#pm%.2f, B: %.2f#pm%.2f"%(title,nsignal, nsignalErr,nbkg*bkgIntegralRange, nbkgErr*bkgIntegralRange))
    pText.Draw("Same")
    cv.Update()
    cv.Print(name+".pdf")
    
    return {
        "s":nsignal,
        "serr":nsignalErr,
        "b":nbkg*bkgIntegralRange,
        "berr":nbkgErr*bkgIntegralRange
    }
          
def fitBMass(name,hist):
    ROOT.RooFit.PrintLevel(4)
    perfInfo = []
    for wpBin in range(hist.GetYaxis().GetNbins()):
        #print " ---------------- ",wpBin," ---------------- "
        massHist = hist.ProjectionX(hist.GetName()+str(wpBin)+str(random.random()),wpBin+1,wpBin+1)
        if massHist.Integral()>=1500:
            massHist.Rebin(2)
        elif massHist.Integral()>=500 and massHist.Integral()<1500:
            massHist.Rebin(3)
        elif massHist.Integral()>=20 and massHist.Integral()<500:
            massHist.Rebin(4)
        elif massHist.Integral()<20:
            continue
        info = fitMass(name+"_fit"+str(wpBin),massHist,"WP: %.2f%%,"%(hist.GetYaxis().GetBinCenter(wpBin+1)*100.))
        info["w"] = hist.GetYaxis().GetBinCenter(wpBin+1)
        info["wpBin"] = wpBin
        perfInfo.append(info)
    massHistRef = hist.ProjectionX(hist.GetName()+str(random.random()),0,0)
    massHistRef.Rebin(3)
    perfInfoRef = fitMass(name+"_fitref",massHistRef,"reference,")
    return perfInfo,perfInfoRef
        
def drawROC(name,effAucList,signalName="Signal",backgroundName="Background",style=1,ref=None):
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
   
    rootObj = []
    for i,effAucDict in enumerate(effAucList):

        graphL = ROOT.TGraph(len(effAucDict["sigEff"]),numpy.array(effAucDict["sigEff"]),numpy.array(effAucDict["bgEff"]))
        rootObj.append(graphL)
        graphL.SetLineColor(ROOT.kOrange+7)
        graphL.SetLineWidth(2+i)
        graphL.SetLineStyle(1+i)
        graphL.Draw("SameLE")

    ROOT.gPad.RedrawAxis()
    if ref!=None:
        markerRef = ROOT.TMarker(ref[0],ref[1],20)
        markerRef.SetMarkerSize(1.3)
        markerRef.SetMarkerColor(ROOT.kBlue+1)
        markerRef.Draw("Same")
    
    '''
    markerCutBased = ROOT.TMarker(0.338,0.00059,20)
    markerCutBased.SetMarkerSize(1.5)
    markerCutBased.SetMarkerColor(ROOT.kViolet+2)
    markerCutBased.Draw("Same")
    
    pMarker=ROOT.TPaveText(0.3,0.00059,0.3,0.00059)
    pMarker.SetFillStyle(0)
    pMarker.SetBorderSize(0)
    pMarker.SetTextFont(63)
    pMarker.SetTextSize(25)
    pMarker.SetTextColor(ROOT.kViolet+2)
    pMarker.SetTextAlign(32)
    pMarker.AddText("cut-based")
    pMarker.Draw("Same")
    '''
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

    pAUC=ROOT.TPaveText(1-cv.GetRightMargin(),0.94,1-cv.GetRightMargin(),0.94,"NDC")
    pAUC.SetFillColor(ROOT.kWhite)
    pAUC.SetBorderSize(0)
    pAUC.SetTextFont(43)
    pAUC.SetTextSize(30)
    pAUC.SetTextAlign(31)
    aucText = "AUC: "
    for i,effAucDict in enumerate(effAucList):
        if effAucDict["auc"]:
            aucText+="%.1f %%"%effAucDict["auc"]
            if i<(len(effAucList)-1):
                aucText+="/"
    pAUC.AddText(aucText)
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
        50, 0, 1.0, 50, 0.0006, math.exp(1.1*math.log(ymax))
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
        
        
basepath = "/vols/cms/mkomm/BPH/training4/"

trainFiles = []
testFiles = []

for l in os.listdir(basepath):
    filePath = os.path.join(basepath,l)
    if re.match("train_\w+.tfrecord",l):
        trainFiles.append(filePath)

            
    if re.match("test_\w+.tfrecord",l):
        testFiles.append(filePath)
       
def atoi(text):
    return int(text) if text.isdigit() else text
        
def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]
         
trainFiles = sorted(trainFiles,key=natural_keys)
testFiles = sorted(testFiles,key=natural_keys)


'''
trainFiles = trainFiles[0:20]
testFiles = testFiles[0:50]
'''
#print trainFiles
        
print "found ",len(trainFiles),"/",len(testFiles)," train/test files"
#sys.exit(1)


batchSizeTrain = 2000
batchSizeTest = 5000

def find_nearest_index(array,value):
    return (numpy.abs(array - value)).argmin()


def setup_model(learning_rate):
    net = Network()
    #net = NetworkDense()
    #net = NetworkFullDense()


    inputs = keras.layers.Input(shape=(50,25))
    #bkgEstimate = keras.layers.Input(shape=(1,))
    prediction = net.getDiscriminant(inputs)
    '''
    def myLoss(y,x):
        signalEstimate = tf.reduce_sum(y*x)
        backgroundEstimate = tf.reduce_sum(y*(1.-x))
        return -1.*signalEstimate/tf.sqrt(signalEstimate+backgroundEstimate+1e-2)
    '''  
        
    model = keras.models.Model(inputs=inputs,outputs=prediction)
    opt = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(
        opt, 
        loss=[keras.losses.binary_crossentropy],
        metrics=['accuracy'],
        loss_weights=[1.]
    )
    return model
    
def train_test_cycle(epoch,model,input_queue,doTrain=False,doPredict=False,doRegularization=True,doWeight=True,doVeto=True,mode="Training",regTargets={}):

    sess = K.get_session()
    
    step = 0
    totalLoss = 0
    
    if doPredict:
        nTotal = 0
        nSignal = 0
        nAcc = 0
        nAccSignal = 0
        
        scoreTest = []
        truthTest = []
        
        histSignal = ROOT.TH1F("signalTrain"+str(epoch)+str(random.random()),"",10000,0,1)
        histSignal.Sumw2()
        histBackground = ROOT.TH1F("backgroundTrain"+str(epoch)+str(random.random()),"",10000,0,1)
        histBackground.Sumw2()
        reversedArr = ((1.-numpy.logspace(-3,-0.3,51))[::-1]).astype(dtype=numpy.float32)
        bMass = ROOT.TH2F("dataMass"+str(epoch)+str(random.random()),"",200,numpy.linspace(4.5,6.5,201,dtype=numpy.float32),50,reversedArr)
        bMass.Sumw2()
        mcEff = {"ref":0.}
        bkgEff = {"ref":0.}
        for wpBin in range(bMass.GetYaxis().GetNbins()):
            mcEff["wp"+str(wpBin+1)] = 0.
            bkgEff["wp"+str(wpBin+1)] = 0.
    
    try:
        while(True):
            batchVal = sess.run(input_queue)
            batchSize = batchVal['truth'].shape[0]
            
            for k in batchVal.keys():
                numpy.nan_to_num(batchVal[k], copy=False)
                #numpy.clip(batchVal[k], -100, 100)

            
                
            if doVeto:
                '''
                veto = (batchVal["truth"][:,0]<0.5)*(
                    (batchVal["veto"][:,0]>4.)*(batchVal["veto"][:,0]<5.)*1.+\
                    (batchVal["veto"][:,0]>5.6)*1.
                )+(batchVal["truth"][:,0]>0.5)*1.
                '''
                
                veto = (batchVal["truth"][:,0]<0.5)*(
                    (numpy.logical_or((batchVal["veto"][:,0]<5.),(batchVal["veto"][:,0]>5.6)))*1.)+\
                    (batchVal["truth"][:,0]>0.5)*1.
                #veto = numpy.ones(batchSize)
            else:
                veto = numpy.ones(batchSize)
            if doWeight:
                signalBatchSum = numpy.sum(batchVal["truth"][:,0])
                backgroundVetoSum = batchSize-numpy.sum(veto) #number of ignored events
                signalWeight = 1. if signalBatchSum==0 else 1.*batchSize/signalBatchSum
                backgroundWeight = 1. if signalBatchSum==batchSize else 1.*batchSize/(batchSize-signalBatchSum-backgroundVetoSum)
                weight = batchVal["truth"][:,0]*signalWeight+(1-batchVal["truth"][:,0])*backgroundWeight
            else:
                weight = numpy.ones(batchSize)
            weight*=veto
                
            
            if doRegularization and (step>10 or epoch>1):
                for _ in range(4):
                    feedDict = {
                        K.learning_phase(): 0,
                        model.targets[0]:batchVal["truth"],
                        model.sample_weights[0]: weight,
                        model.inputs[0]:batchVal["features"]
                    }

                    predictionVal,inputGradientsPredictionVal,inputGradientsLossVal = sess.run(
                        [
                            regTargets["predictionFct"],
                            regTargets["predictionGradients"],
                            regTargets["lossGradients"]
                        ],
                        feed_dict=feedDict
                    )    
                    feedDict[model.inputs[0]] = batchVal["features"]-inputGradientsPredictionVal[0]
                    predictionVal2 = sess.run(regTargets["predictionFct"],feed_dict=feedDict) 
                    scale = 7.*numpy.multiply((1.1-predictionVal),numpy.square(predictionVal))
                    scale*=numpy.abs(numpy.random.normal(1.,0.5,size=scale.shape))
                    changeInv = 1./(predictionVal-predictionVal2+1e-2)*scale
                    changeInv*=batchVal["truth"]
                    inputGradientsLossVal[0]*=numpy.abs(numpy.random.normal(0.,1.,size=inputGradientsLossVal[0].shape))+1.
                    #randomly move signal prediction towards 0
                    batchVal["features"]-=numpy.einsum('ijk,i->ijk',inputGradientsPredictionVal[0]*0.01,changeInv[:,0])
                    #randomly move signal to higher loss
                    batchVal["features"]+=numpy.einsum('ijk,i->ijk',inputGradientsLossVal[0],batchVal["truth"][:,0])
            

            step += 1
            if doTrain:
                result = model.train_on_batch(batchVal["features"],batchVal["truth"],sample_weight=weight)
                loss = result[0]
                totalLoss += loss
            else:
                result = model.test_on_batch(batchVal["features"],batchVal["truth"],sample_weight=weight)
                loss = result[0]
                totalLoss += loss   
            

            if doPredict:
                predict = model.predict_on_batch(batchVal["features"])
                #print predict
                for i in range(batchSize):
                    if batchVal["truth"][i][0]>0.5:
                        nTotal+=1
                        scoreTest.append(predict[i][0])
                        truthTest.append(1)
                        nSignal+=1
                        histSignal.Fill(predict[i][0])
                        if predict[i][0]>0.5:
                            nAccSignal+=1
                            nAcc+=1
                        for wpBin in range(bMass.GetYaxis().GetNbins()):
                            wpValue = bMass.GetYaxis().GetBinCenter(wpBin+1)
                            if predict[i][0]>wpValue:
                                mcEff["wp"+str(wpBin+1)]+=1.
                        if batchVal["refSel"][i,0]>0.5:
                            mcEff["ref"]+=1.

                    else:
                        if veto[i]>0.5:
                            nTotal+=1
                            scoreTest.append(predict[i][0])
                            truthTest.append(0)
                            histBackground.Fill(predict[i][0])
                            if predict[i][0]<0.5:
                                nAcc+=1
                        for wpBin in range(bMass.GetYaxis().GetNbins()):
                            wpValue = bMass.GetYaxis().GetBinCenter(wpBin+1)
                            if predict[i][0]>wpValue:
                                bMass.Fill(batchVal["veto"][i,0],wpValue)
                                if veto[i]>0.5:
                                    bkgEff["wp"+str(wpBin+1)]+=1.
                        if batchVal["refSel"][i,0]>0.5 and veto[i]>0.5:
                            bkgEff["ref"]+=1.
                                
                        if batchVal["refSel"][i,0]>0.5:
                            bMass.Fill(batchVal["veto"][i,0],-1)
                                
                
                    
                        
            if step%10==0:
                if doPredict:
                    print "%s step %i-%i: loss=%.4f, acc=%.3f%%, accSignal=%.3f%%"%(
                        mode,epoch,step,loss,
                        100.*nAcc/nTotal if nTotal>0 else 0,
                        100.*nAccSignal/nSignal if nSignal>0 else 0,
                    )
                else:
                    print "%s step %i-%i: loss=%.4f"%(
                        mode,epoch,step,loss
                    )
            
    except tf.errors.OutOfRangeError:
        print('%s done for %d steps.' % (mode,step))
    
    if doPredict:
        for wpBin in range(bMass.GetYaxis().GetNbins()):
            mcEff["wp"+str(wpBin+1)]/=nSignal
            bkgEff["wp"+str(wpBin+1)]/=(nTotal-nSignal)
        mcEff["ref"]/=nSignal
        bkgEff["ref"]/=(nTotal-nSignal)

    result = {
        "steps":step,
        "loss":totalLoss,
    }
    
    if doPredict:
        result["histSignal"] = histSignal
        result["histBackground"] = histBackground
        result["acc"] = 100.*nAcc/nTotal if nTotal>0 else 0
        result["accSignal"] = 100.*nAccSignal/nSignal if nSignal>0 else 0
        result["score"] = scoreTest
        result["truth"] = truthTest
        result["bMass"] = bMass
        result["mcEff"] = mcEff
        result["bkgEff"] = bkgEff
    
    return result

outputFolder = "result_peak"

fstat = open(os.path.join(outputFolder,"model_stat.txt"),"w")
fstat.close()

'''
print "-"*70
trainBatch = Generator(trainFiles)
testBatch = Generator(testFiles)
print "Signal: %i/%i (train/test)"%(trainBatch.nSignal,testBatch.nSignal)
print "Background: %i/%i (train/test)"%(trainBatch.nBackground,testBatch.nBackground)
print "S/B: %.2f%%/%.2f%% (train/test)"%(100.*trainBatch.nSignal/trainBatch.nBackground,100.*testBatch.nSignal/testBatch.nBackground)
print "-"*70
'''

learning_rate = 0.005
    
previousLoss = 1e10
for epoch in range(1,51):

    model = setup_model(learning_rate)
    if epoch==1:
        model.summary()

    trainBatch = input_pipeline(trainFiles,batchSizeTrain)
    trainBatchForRoc = input_pipeline(trainFiles,batchSizeTrain)
    #only evaluate every 5th epoch on full test sample
    if epoch%2==0:
        testBatch = input_pipeline(testFiles,batchSizeTest)
    else:
        testBatch = input_pipeline(testFiles[0:max(1,len(testFiles)/10)],batchSizeTest)
    
    
    classLossFct = model.total_loss #includes also regularization loss
    inputGradientsLoss = tf.gradients(classLossFct,model.inputs)
    
    predictionFct = model.outputs[0]
    inputGradientsPrediction = tf.gradients(predictionFct,model.inputs)
    
    regTargets = {
        "classLossFct":classLossFct,
        "lossGradients":inputGradientsLoss,
        "predictionFct":predictionFct,
        "predictionGradients":inputGradientsPrediction
    }
    
    sess = K.get_session()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    
    if epoch>1:
        model.load_weights(os.path.join(outputFolder,"weights_%i.hdf5"%(epoch-1)))
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    

    resultTrain = train_test_cycle(epoch,model,trainBatch,doTrain=True,doPredict=False,doRegularization=True,doWeight=True,mode="Training",regTargets=regTargets)
    
    model.save_weights(os.path.join(outputFolder,"weights_%i.hdf5"%epoch))
    
    resultTrainPerf = train_test_cycle(epoch,model,trainBatchForRoc,doTrain=False,doPredict=True,doRegularization=False,doWeight=False,mode="TrainPerf")
    resultTestPerf = train_test_cycle(epoch,model,testBatch,doTrain=False,doPredict=True,doRegularization=False,doWeight=False,mode="TestPerf")
    #sys.exit(1)
    
    
    bgEffTrain,sigEffTrain,thresTrain = metrics.roc_curve(resultTrainPerf["truth"], resultTrainPerf["score"])
    bgEffTest,sigEffTest,thresTest = metrics.roc_curve(resultTestPerf["truth"], resultTestPerf["score"])
    aucTrain = 100.*metrics.auc(bgEffTrain,sigEffTrain)
    aucTest = 100.*metrics.auc(bgEffTest,sigEffTest)
    
    if epoch%2==0:
        perfInfoList,perfInfoRef = fitBMass(os.path.join(outputFolder,"roc_epoch%i"%epoch),resultTestPerf["bMass"])
        #print "ref",resultTestPerf["mcEff"]["ref"],perfInfoRef["s"]/math.sqrt(perfInfoRef["s"]+perfInfoRef["b"])
        significance = []
        significanceErr = []
        mcEff = []
        for iperf,perfInfo in enumerate(perfInfoList):
            if perfInfo["w"]>thresTest[5] or perfInfo["w"]<thresTest[-5]:
                continue
            sig = 1.*perfInfo["s"]/math.sqrt(perfInfo["s"]+perfInfo["b"])
            sigEff = 1.*math.sqrt(
                ((2*perfInfo["b"]+perfInfo["s"])/(2*(perfInfo["s"]+perfInfo["b"])**(3./2)))**2*perfInfo["serr"]**2+\
                (perfInfo["s"]/(2*(perfInfo["s"]+perfInfo["b"])**(3./2)))**2*perfInfo["berr"]**2
            )
            if sigEff/sig>1:
                continue
            mcEff.append(1.*resultTestPerf["mcEff"]["wp"+str(perfInfo["wpBin"]+1)])
            significance.append(sig)
            significanceErr.append(sigEff)
        #print mcEff
        #print significance
        #print significanceErr
        
        refSignificance = perfInfoRef["s"]/math.sqrt(perfInfoRef["s"]+perfInfoRef["b"])
        
        if len(significance)>5:
            significance = numpy.array(significance)
            significanceErr = numpy.array(significanceErr)
            mcEff = numpy.array(mcEff)
            
            cvPerf = ROOT.TCanvas("cvperf"+str(epoch)+str(random.random()),"",800,700)
            cvPerf.SetMargin(0.15,0.08,0.17,0.1)
            axis = ROOT.TH2F("axis"+str(epoch)+str(random.random()),";MC efficiency;S/#sqrt{S+B}",
                50,0,1,
                50,min(significance)*0.7,max(max(significance),refSignificance)*1.2
            )
            axis.Draw("AXIS")
            graphPerf = ROOT.TGraphErrors(len(mcEff),mcEff,significance,numpy.zeros(len(mcEff)),significanceErr)
            graphPerf.SetLineWidth(2)
            graphPerf.SetLineColor(ROOT.kOrange+7)
            graphPerf.SetMarkerSize(1.1)
            graphPerf.SetMarkerStyle(21)
            graphPerf.SetMarkerColor(ROOT.kOrange+7)
            graphPerf.Draw("PLSame")
            
            markerRef = ROOT.TMarker(resultTestPerf["mcEff"]["ref"],refSignificance,20)
            markerRef.SetMarkerSize(1.3)
            markerRef.SetMarkerColor(ROOT.kBlue+1)
            markerRef.Draw("Same")
            
            
            cvPerf.Print(os.path.join(outputFolder,"dist_epoch%i_perf.pdf"%epoch))
   
    drawROC(os.path.join(outputFolder,"roc_epoch%i"%epoch),[
        {"sigEff":sigEffTrain,"bgEff":bgEffTrain,"auc":aucTrain},
        {"sigEff":sigEffTest,"bgEff":bgEffTest,"auc":aucTest}
    ],ref=[resultTestPerf["mcEff"]["ref"],resultTestPerf["bkgEff"]["ref"]])
    

    resultTrainPerf["histSignal"].Rebin(250)
    resultTrainPerf["histBackground"].Rebin(250)
    
    resultTestPerf["histSignal"].Rebin(250)
    resultTestPerf["histBackground"].Rebin(250)
    
    drawDist(
        os.path.join(outputFolder,"dist_epoch%i"%epoch),
        [resultTrainPerf["histSignal"],resultTestPerf["histSignal"]],
        [resultTrainPerf["histBackground"],resultTestPerf["histBackground"]]
    )
        
    avgLossTrain = 1.*resultTrainPerf["loss"]/resultTrainPerf["steps"]
    avgLossTest = 1.*resultTestPerf["loss"]/resultTestPerf["steps"]

        
    print "-"*70
    
    print "Epoch %i summary: lr=%.3e, loss=%.3f/%.3f, acc=%.2f%%/%.2f%%, accSignal=%.2f%%/%.2f%%, AUC=%.2f%%/%.2f%% (train/test)"%(
        epoch,
        learning_rate,
        avgLossTrain,avgLossTest,
        resultTrainPerf["acc"],resultTestPerf["acc"],
        resultTrainPerf["accSignal"],resultTestPerf["accSignal"],
        aucTrain,aucTest
    ) 
    
    fstat = open(os.path.join(outputFolder,"model_stat.txt"),"a")
    fstat.write("%11.4e,%8.4f,%8.4f,%8.4f,%8.4f,%8.4f,%8.4f,%8.4f,%8.4f\n"%(
        learning_rate,
        avgLossTrain,avgLossTest,
        resultTrainPerf["acc"],resultTestPerf["acc"],
        resultTrainPerf["accSignal"],resultTestPerf["accSignal"],
        aucTrain,aucTest
    ))
    fstat.close()
    
    if (avgLossTrain>previousLoss):
        learning_rate *= 0.8
        print "Reducing learning rate to: %.4e"%learning_rate
        previousLoss = 0.5*(previousLoss+avgLossTrain)
    else:
        previousLoss = avgLossTrain
    print "-"*70
    
    coord.request_stop()
    coord.join(threads)


    K.clear_session()
    
    

    

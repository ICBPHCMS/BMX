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

from sklearn import metrics

from network import Network

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
    
    ncombinations = 200
    nfeatures = 24
    nscales = 2
    featureList = {
        'truth': tf.FixedLenFeature([1], tf.float32),
        'features': tf.FixedLenFeature([ncombinations*nfeatures], tf.float32),
        'genIndex': tf.FixedLenFeature([ncombinations+1], tf.float32),
        'scales': tf.FixedLenFeature([ncombinations*nscales], tf.float32),
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
        maxThreads = 6
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
          
                   

def drawROC(name,effAucList,signalName="Signal",backgroundName="Background",style=1):
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
         
        
#trainFiles = trainFiles[0:1]
#testFiles = testFiles[0:1]

        
print "found ",len(trainFiles),"/",len(testFiles)," train/test files"
#sys.exit(1)


batchSizeTrain = 5000
batchSizeTest = 10000


def setup_model(learning_rate):
    net = Network()
    #net = NetworkDense()
    #net = NetworkFullDense()


    inputs = keras.layers.Input(shape=(200,24))
    prediction = net.getDiscriminant(inputs)
        
    model = keras.models.Model(inputs=inputs,outputs=prediction)
    opt = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(
        opt, 
        loss=[keras.losses.binary_crossentropy],
        metrics=['accuracy'],
        loss_weights=[1.]
    )
    return model
    
def train_test_cycle(epoch,model,input_queue,doTrain=False,doPredict=False,doRegularization=False,doWeight=True,mode="Training",regTargets={}):

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
        
    
    try:
        while(True):
            batchVal = sess.run(input_queue)
            batchSize = batchVal['truth'].shape[0]
            
            for k in batchVal.keys():
                numpy.nan_to_num(batchVal[k], copy=False)
                
            if doWeight:
                signalBatchSum = sum(batchVal["truth"])  
                signalWeight = 1. if signalBatchSum==0 else 1.*batchSize/signalBatchSum
                backgroundWeight = 1. if signalBatchSum==batchSize else 1.*batchSize/(batchSize-signalBatchSum)
                weight = batchVal["truth"][:,0]*signalWeight+(1-batchVal["truth"][:,0])*backgroundWeight
            else:
                weight = numpy.ones(batchSize)
            
            if doRegularization and (step>10 or epoch>1):
                for _ in range(2):
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
                    scale = (1.1-predictionVal)
                    scale*=numpy.abs(numpy.random.normal(1.,0.2,size=scale.shape))
                    changeInv = 1./(predictionVal-predictionVal2+1e-2)*scale
                    changeInv*=batchVal["truth"]
                    inputGradientsLossVal[0]*=numpy.abs(numpy.random.normal(0.,0.5,size=inputGradientsLossVal[0].shape))+0.5
                    #randomly move signal prediction towards 0
                    batchVal["features"]-=numpy.einsum('ijk,i->ijk',inputGradientsPredictionVal[0]*0.005,changeInv[:,0])
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
                    nTotal+=1
                    scoreTest.append(predict[i][0])
                    if batchVal["truth"][i][0]>0.5:
                        truthTest.append(1)
                        nSignal+=1
                        histSignal.Fill(predict[i][0])
                        if predict[i][0]>0.5:
                            nAccSignal+=1
                            nAcc+=1

                    else:
                        truthTest.append(0)
                        histBackground.Fill(predict[i][0])
                        if predict[i][0]<0.5:
                            nAcc+=1
                        
                        
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
    
    return result

outputFolder = "result_comb17"

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
    if epoch%5==0:
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

    resultTrain = train_test_cycle(epoch,model,trainBatch,doTrain=True,doPredict=True,doRegularization=True,doWeight=True,mode="Training",regTargets=regTargets)
    
    model.save_weights(os.path.join(outputFolder,"weights_%i.hdf5"%epoch))
    
    resultTrainPerf = train_test_cycle(epoch,model,trainBatchForRoc,doTrain=False,doPredict=True,doRegularization=False,doWeight=False,mode="TrainPerf")
    resultTestPerf = train_test_cycle(epoch,model,testBatch,doTrain=False,doPredict=True,doRegularization=False,doWeight=False,mode="TestPerf")
    

    bgEffTrain,sigEffTrain,thresTrain = metrics.roc_curve(resultTrainPerf["truth"], resultTrainPerf["score"])
    bgEffTest,sigEffTest,thresTest = metrics.roc_curve(resultTestPerf["truth"], resultTestPerf["score"])
    aucTrain = 100.*metrics.auc(bgEffTrain,sigEffTrain)
    aucTest = 100.*metrics.auc(bgEffTest,sigEffTest)
   
    drawROC(os.path.join(outputFolder,"roc_epoch%i"%epoch),[
        {"sigEff":sigEffTrain,"bgEff":bgEffTrain,"auc":aucTrain},
        {"sigEff":sigEffTest,"bgEff":bgEffTest,"auc":aucTest}
    ])
    

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
    
    

    

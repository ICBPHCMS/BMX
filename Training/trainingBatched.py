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
            reader_batch = max(10,int(batchSize/20.))
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
          
                    
class Network:
    def __init__(self,regLoss=1e-6):
        self.convLayers = [
            keras.layers.Conv1D(32, 1, strides=1, activation=None,
                kernel_regularizer=keras.regularizers.l2(regLoss)
            ),
            keras.layers.LeakyReLU(alpha=0.1), 
            #drop same features (=filters) for all combinations
            keras.layers.Dropout(0.2,noise_shape=[1,1,32]),
            keras.layers.Conv1D(32, 1, strides=1, activation=None,
                kernel_regularizer=keras.regularizers.l2(regLoss)
            ),
            keras.layers.LeakyReLU(alpha=0.1), 
            #drop same features (=filters) for all combinations
            keras.layers.Dropout(0.2,noise_shape=[1,1,32]),
            keras.layers.Conv1D(16, 1, strides=1, activation='tanh', 
                kernel_regularizer=keras.regularizers.l2(regLoss)
            ),
        ]

        self.lstmLayers = [
            keras.layers.LSTM(50, activation='tanh', recurrent_activation='hard_sigmoid',
                kernel_regularizer=keras.regularizers.l2(regLoss),
                implementation=2,
                recurrent_dropout=0.05,
                go_backwards=True
            ),
            keras.layers.Dropout(0.1),
        ]

        self.bypassLayersLSTM = [
            keras.layers.Lambda(lambda l:l[:,0:5,:]),
            keras.layers.Flatten(),
            #drop complete bypass
            keras.layers.Dropout(0.8,noise_shape=[1,1]),
        ]
        
        self.bypassLayersDense = [
            keras.layers.Dropout(0.8,noise_shape=[1,1]),
        ]
        
        self.predictLayers = [
            keras.layers.Dense(100, activation=None,
                kernel_regularizer=keras.regularizers.l2(regLoss)
            ),
            keras.layers.LeakyReLU(alpha=0.1),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(50, activation=None,
                kernel_regularizer=keras.regularizers.l2(regLoss)
            ),
            keras.layers.LeakyReLU(alpha=0.1),
            keras.layers.Dropout(0.1),
        ]
        
        self.finalLayers = [
            keras.layers.Dense(100, activation=None,
                kernel_regularizer=keras.regularizers.l2(regLoss)
            ),
            keras.layers.LeakyReLU(alpha=0.1),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(1, activation='sigmoid',
                kernel_regularizer=keras.regularizers.l2(regLoss)
            )
        ]

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
            
        bypassLSTM = self.bypassLayersLSTM[0](conv)
        for layer in self.bypassLayersLSTM[1:]:
            bypassLSTM = layer(bypassLSTM)
            
        features = keras.layers.Concatenate()([lstm,bypassLSTM])
        
        predict = self.predictLayers[0](features)
        for layer in self.predictLayers[1:]:
            predict = layer(predict)
        
        bypassDense = self.bypassLayersDense[0](features)
        for layer in self.bypassLayersDense[1:]:
            bypassDense = layer(bypassDense)
        
        final = keras.layers.Concatenate()([predict,bypassDense])
        final = self.finalLayers[0](final)
        for layer in self.finalLayers[1:]:
            final = layer(final)
        return final
        
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
            keras.layers.Dense(50, activation=None,
                kernel_regularizer=keras.regularizers.l2(regLoss)
            ),
            keras.layers.LeakyReLU(alpha=0.1),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(30, activation=None,
                kernel_regularizer=keras.regularizers.l2(regLoss)
            ),
            keras.layers.LeakyReLU(alpha=0.1),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(30, activation=None,
                kernel_regularizer=keras.regularizers.l2(regLoss)
            ),
            keras.layers.LeakyReLU(alpha=0.1),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(30, activation=None,
                kernel_regularizer=keras.regularizers.l2(regLoss)
            ),
            keras.layers.LeakyReLU(alpha=0.1),
            keras.layers.Dropout(0.3),
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
        
class NetworkFullDense:
    def __init__(self,regLoss=1e-6):
        self.convLayers = [
            keras.layers.Conv1D(32, 1, strides=1, activation=None,
                kernel_regularizer=keras.regularizers.l2(regLoss)
            ),
            keras.layers.LeakyReLU(alpha=0.1), 
            #drop same features (=filters) for all combinations
            keras.layers.Dropout(0.2,noise_shape=[1,1,32]),
            keras.layers.Conv1D(32, 1, strides=1, activation=None,
                kernel_regularizer=keras.regularizers.l2(regLoss)
            ),
            keras.layers.LeakyReLU(alpha=0.1), 
            #drop same features (=filters) for all combinations
            keras.layers.Dropout(0.2,noise_shape=[1,1,32]),
            keras.layers.Conv1D(16, 1, strides=1, activation='tanh', 
                kernel_regularizer=keras.regularizers.l2(regLoss)
            ),
        ]
        
        self.predictLayers = [
            keras.layers.Dense(200, activation=None,
                kernel_regularizer=keras.regularizers.l2(regLoss)
            ),
            keras.layers.LeakyReLU(alpha=0.1),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(100, activation=None,
                kernel_regularizer=keras.regularizers.l2(regLoss)
            ),
            keras.layers.LeakyReLU(alpha=0.1),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(50, activation=None,
                kernel_regularizer=keras.regularizers.l2(regLoss)
            ),
            keras.layers.LeakyReLU(alpha=0.1),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(50, activation=None,
                kernel_regularizer=keras.regularizers.l2(regLoss)
            ),
            keras.layers.LeakyReLU(alpha=0.1),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(1, activation='sigmoid',
                kernel_regularizer=keras.regularizers.l2(regLoss)
            )
        ]
        
    
    def getDiscriminant(self,x):
        conv = self.convLayers[0](x)
        for layer in self.convLayers[1:]:
            conv = layer(conv)
        predict = keras.layers.Flatten()(conv)
        predict = self.predictLayers[0](predict)
        for layer in self.predictLayers[1:]:
            predict = layer(predict)
        return predict
        
'''
def getROCErrors(truthTest, scoreTest,bag=20):
    
    bgEffDist = numpy.zeros((len(bgEff),bag))
    for n in range(bag):
        subIndices = numpy.multiply((numpy.random.uniform(0,1,len(sigEff))>0.5)*2-1,numpy.linspace(0,len(sigEff)-1,len(sigEff),dtype=numpy.int32))
        sigEffSub = sigEff[numpy.extract(subIndices>=0,subIndices)]
        bgEffSub = bgEff[numpy.extract(subIndices>=0,subIndices)]
        
        bgEff,sigEff,thres = metrics.roc_curve(truthTest, scoreTest)
        
        g = ROOT.TGraph(len(sigEffSub),sigEffSub,bgEffSub)
        #g.SetBit(ROOT.TGraph.kIsSortedX)
        bgEffAlt = numpy.zeros(len(bgEff))
        for i in range(len(bgEff)):
            bgEffDist[i][n]=g.Eval(sigEff[i])
    bgMean = numpy.mean(bgEffDist,axis=1)
    bgErr = numpy.std(bgEffDist,axis=1)
    
    return bgMean,bgErr
'''

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
    
    #bgMean, bgErr = getROCErrors(sigEff,bgEff)

    #### draw here
    graphF = ROOT.TGraph(len(sigEff),numpy.array(sigEff),numpy.array(bgEff))
    graphF.SetLineWidth(0)
    graphF.SetFillColor(ROOT.kOrange+10)
    #graphF.Draw("SameF")

    graphL = ROOT.TGraphErrors(len(sigEff),numpy.array(sigEff),numpy.array(bgEff))
    graphL.SetLineColor(ROOT.kOrange+7)
    graphL.SetLineWidth(3)
    graphL.SetLineStyle(style)
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


batchSizeTrain = 2000
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
    


outputFolder = "result_comb4"

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
    testBatch = input_pipeline(testFiles,batchSizeTest)
    

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
    
    
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = K.get_session()
    sess.run(init_op)
    
    if epoch>1:
        model.load_weights(os.path.join(outputFolder,"weights_%i.hdf5"%(epoch-1)))
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    try:
        while(True):
            trainBatchVal = sess.run(trainBatch)
            for k in trainBatchVal.keys():
                numpy.nan_to_num(trainBatchVal[k], copy=False)
                
            signalBatchSum = sum(trainBatchVal["truth"])  
            signalweight = 1. if signalBatchSum==0 else 1.*batchSizeTrain/signalBatchSum
            backgroundweight = 1. if signalBatchSum==batchSizeTrain else 1.*batchSizeTrain/(batchSizeTrain-signalBatchSum)
            weight = trainBatchVal["truth"][:,0]*signalweight+(1-trainBatchVal["truth"][:,0])*backgroundweight
            
            if stepTrain>10 or epoch>1:
                for _ in range(3):
                    feedDict = {
                        K.learning_phase(): 0,
                        model.targets[0]:trainBatchVal["truth"],
                        model.sample_weights[0]: weight,
                        model.inputs[0]:trainBatchVal["features"]
                    }

                    classLossVal,inputGradientsVal = sess.run([classLossFct,inputGradients],feed_dict=feedDict)         
                    direction = numpy.abs(numpy.random.normal(0,1.,trainBatchVal["features"].shape[0]))+1.
                    shifts=numpy.einsum('ijk,i->ijk',inputGradientsVal[0],numpy.multiply(direction,trainBatchVal["truth"][:,0]))
                    trainBatchVal["features"]+=shifts
            
            stepTrain += 1
            result = model.train_on_batch(trainBatchVal["features"],trainBatchVal["truth"],sample_weight=weight)
            predict = model.predict_on_batch(trainBatchVal["features"])
            loss = result[0]
            
            #lossComb = sum(result[batchSize+1:2*batchSize+1])/batchSize
            totalLossTrain+=loss
            #totalLossCombTrain+=lossComb 
            
            #print predict
            for i in range(batchSizeTrain):
                if trainBatchVal["truth"][i][0]>0.5:
                    nSignalTrain+=1
                    histSignalTrain.Fill(predict[i][0])
                    if predict[i][0]>0.5:
                        nAccSignalTrain+=1
                        nAccTrain+=1

                else:
                    histBackgroundTrain.Fill(predict[i][0])
                    if predict[i][0]<0.5:
                        nAccTrain+=1
                        
                        
            if stepTrain%10==0:
                print "Training step %i-%i: loss=%.4f, acc=%.3f%%, accSignal=%.3f%%"%(
                    epoch,stepTrain,loss,
                    100.*nAccTrain/stepTrain/batchSizeTrain,
                    100.*nAccSignalTrain/nSignalTrain if nSignalTrain>0 else 0,
                )
            
    except tf.errors.OutOfRangeError:
        print('Done training for %d steps.' % (stepTrain))
    
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
    
    try:
        while(True):
            testBatchVal = sess.run(testBatch)
            for k in testBatchVal.keys():
                numpy.nan_to_num(testBatchVal[k], copy=False)
                
            signalBatchSum = sum(testBatchVal["truth"])  
            signalweight = 1. if signalBatchSum==0 else 1.*batchSizeTest/signalBatchSum
            backgroundweight = 1. if signalBatchSum==batchSizeTest else 1.*batchSizeTest/(batchSizeTest-signalBatchSum)
            weight = testBatchVal["truth"][:,0]*signalweight+(1-testBatchVal["truth"][:,0])*backgroundweight
                
            stepTest += 1
            result = model.test_on_batch(testBatchVal["features"],testBatchVal["truth"],sample_weight=weight)
            predict = model.predict_on_batch(testBatchVal["features"])
            loss = result[0]
            #lossComb = sum(result[batchSize+1:2*batchSize+1])/batchSize
            totalLossTest+=loss
            #totalLossCombTest+=lossComb
            
            
            for i in range(batchSizeTest):
                scoreTest.append(predict[i][0])
                if testBatchVal["truth"][i][0]>0.5:
                    nSignalTest+=1
                    histSignalTest.Fill(predict[i][0])
                    truthTest.append(1)
                    if predict[i][0]>0.5:
                        nAccSignalTest+=1
                        nAccTest+=1
                    
                else:
                    truthTest.append(0)
                
                    histBackgroundTest.Fill(predict[i][0])
                    if predict[i][0]<0.5:
                        nAccTest+=1

            if stepTest%10==0:
                print "Testing step %i-%i: loss=%.4f, acc=%.3f%%, accSignal=%.3f%%"%(
                    epoch,stepTest,loss,
                    100.*nAccTest/stepTest/batchSizeTest,
                    100.*nAccSignalTest/nSignalTest if nSignalTest>0 else 0,
                )
                
    except tf.errors.OutOfRangeError:
        print('Done testing for %d steps.' % (stepTest))
            
    bgEffTest,sigEffTest,thres = metrics.roc_curve(truthTest, scoreTest)
    aucTest = metrics.auc(bgEffTest,sigEffTest)
    #sigEffTest, bgRejTest, bgEffTest = getROC(histSignalTest,histBackgroundTest)
    #aucTest = getAUC(sigEffTest, bgRejTest)
    drawROC(os.path.join(outputFolder,"roc_epoch%i"%epoch),sigEffTest,bgEffTest,auc=aucTest)
    
    histSignalTrain.Rebin(250)
    histBackgroundTrain.Rebin(250)
    
    histSignalTest.Rebin(250)
    histBackgroundTest.Rebin(250)
    
    drawDist(
        os.path.join(outputFolder,"dist_epoch%i"%epoch),
        [histSignalTrain,histSignalTest],
        [histBackgroundTrain,histBackgroundTest]
    )
        
    avgLossTrain = totalLossTrain/stepTrain
    avgLossTest = totalLossTest/stepTest
        
    print "-"*70
    
    print "Epoch %i summary: lr=%.3e, loss=%.3f/%.3f, acc=%.2f%%/%.2f%%, accSignal=%.2f%%/%.2f%% (train/test), AUC=%.2f%%"%(
        epoch,
        learning_rate,
        avgLossTrain,avgLossTest,
        100.*nAccTrain/stepTrain/batchSizeTrain,100.*nAccTest/stepTest/batchSizeTest,
        100.*nAccSignalTrain/nSignalTrain,100.*nAccSignalTest/nSignalTest,
        100.*aucTest
    ) 
    
    fstat = open(os.path.join(outputFolder,"model_stat.txt"),"a")
    fstat.write("%11.4e,%8.4f,%8.4f,%8.4f,%8.4f,%8.4f,%8.4f,%8.4f\n"%(
        learning_rate,avgLossTrain,avgLossTest,
        100.*nAccTrain/stepTrain/batchSizeTrain,100.*nAccTest/stepTest/batchSizeTest,
        100.*nAccSignalTrain/nSignalTrain,100.*nAccSignalTest/nSignalTest,
        100.*aucTest
    ))
    fstat.close()
    
    if (avgLossTrain>previousLoss):
        learning_rate *= 0.8
        print "Reducing learning rate to: %.4e"%lr
        previousLoss = 0.5*(previousLoss+avgLossTrain)
    else:
        previousLoss = avgLossTrain
    print "-"*70
    
    coord.request_stop()
    coord.join(threads)


    K.clear_session()
    
    

    

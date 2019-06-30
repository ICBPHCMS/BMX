import h5py
import ROOT
import numpy
import os
import random

ROOT.gROOT.SetBatch(True)
ROOT.gROOT.SetStyle("Plain")
ROOT.gStyle.SetOptFit(0)
ROOT.gStyle.SetOptDate(0)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptFile(0)
ROOT.gStyle.SetOptTitle(0)


ROOT.gStyle.SetCanvasBorderMode(0)
ROOT.gStyle.SetCanvasColor(ROOT.kWhite)

ROOT.gStyle.SetPadBorderMode(0)
ROOT.gStyle.SetPadColor(ROOT.kWhite)
ROOT.gStyle.SetGridColor(ROOT.kBlack)
ROOT.gStyle.SetGridStyle(2)
ROOT.gStyle.SetGridWidth(1)

ROOT.gStyle.SetPadTopMargin(0.07)
ROOT.gStyle.SetPadRightMargin(0.04)
ROOT.gStyle.SetPadBottomMargin(0.14)
ROOT.gStyle.SetPadLeftMargin(0.14)

ROOT.gStyle.SetFrameBorderMode(0)
ROOT.gStyle.SetFrameBorderSize(0)
ROOT.gStyle.SetFrameFillColor(0)
ROOT.gStyle.SetFrameFillStyle(0)
ROOT.gStyle.SetFrameLineColor(1)
ROOT.gStyle.SetFrameLineStyle(1)
ROOT.gStyle.SetFrameLineWidth(0)

ROOT.gStyle.SetEndErrorSize(2)
ROOT.gStyle.SetErrorX(0.)
ROOT.gStyle.SetMarkerStyle(20)

ROOT.gStyle.SetHatchesSpacing(0.9)
ROOT.gStyle.SetHatchesLineWidth(2)

ROOT.gStyle.SetTitleColor(1, "XYZ")
ROOT.gStyle.SetTitleFont(43, "XYZ")
ROOT.gStyle.SetTitleSize(33, "XYZ")
ROOT.gStyle.SetTitleXOffset(1.135)
ROOT.gStyle.SetTitleOffset(1.4, "YZ")

ROOT.gStyle.SetLabelColor(1, "XYZ")
ROOT.gStyle.SetLabelFont(43, "XYZ")
ROOT.gStyle.SetLabelSize(29, "XYZ")

ROOT.gStyle.SetAxisColor(1, "XYZ")
ROOT.gStyle.SetAxisColor(1, "XYZ")
ROOT.gStyle.SetStripDecimals(True)
ROOT.gStyle.SetNdivisions(1005, "X")
ROOT.gStyle.SetNdivisions(506, "Y")

ROOT.gStyle.SetPadTickX(1)
ROOT.gStyle.SetPadTickY(1)

ROOT.gStyle.SetPaperSize(8.0*1.35,6.7*1.35)
ROOT.TGaxis.SetMaxDigits(3)
ROOT.gStyle.SetLineScalePS(2)

ROOT.gStyle.SetPalette(1)
ROOT.gStyle.SetPaintTextFormat(".1f")

#fitting mass distribution using RooFit
def fitMass(name,massHist,title=""):
    w = ROOT.RooWorkspace("w"+str(name)+str(random.random()))
    w.factory("x[4., 6.5]")
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
    plot.SetXTitle("K#kern[-0.4]{ }J/#Psi(#rightarrow#mu#mu) mass (GeV)")
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
    plot.Draw()
    pText = ROOT.TPaveText(1-cv.GetRightMargin(),1-cv.GetTopMargin()+0.03,1-cv.GetRightMargin(),1-cv.GetTopMargin()+0.03,"NDC")
    pText.SetBorderSize(0)
    pText.SetFillColor(ROOT.kWhite)
    pText.SetTextSize(30)
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
    
    

basepath = "/vols/cms/mkomm/BPH/KmumuDataOnlyPR51"

files = []
for f in os.listdir(basepath):
    if f.endswith(".hdf5"):
        files.append(os.path.join(basepath,f))
        
hist = ROOT.TH1F("hist",";B mass (GeV); Events",160,4.,8.)
hist.Sumw2()

#loop over files
for f in files:
    data = h5py.File(f)
    #the reconstructed B mass per event and triplet
    bmass = data['bmass'][()]
    #the reconstructed dilepton mass per event and triplet
    llmass = data['llmass'][()]
    #the number of triplets per event
    nTriples = data['nCombinations'][()]
    #reference selection (need to train neural network to replace this)
    referenceSelection = data['refSel'][()]
    
    #loop over events per file
    for ievent in range(bmass.shape[0]):
        #loop over triplets
        for itriplet in range(nTriples[ievent]):
            # check if triplet is selected by reference
            if (referenceSelection[ievent,itriplet]>0):
            
                #select J/Psi mass window
                if (llmass[ievent,itriplet]>2.9 and llmass[ievent,itriplet]<3.1):
                    hist.Fill(bmass[ievent,itriplet])
                    
                    #break loop over triplets => only put 1 triplet per event into histogram
                    break
        
fitMass("bmass",hist)


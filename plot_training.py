import os,sys
sys.argv.append("-b")
import ROOT as rt
import numpy as np

#ntraining = 46527
#ntraining = 101191
ntraining =  100232
nevents_per_iteration = 8

iters_per_point = 20
test_iters_per_point = 100
nevents_per_epoch = float(ntraining)/float(nevents_per_iteration)

f = open(sys.argv[1],'r')

out = rt.TFile("plots_training.root", "recreate")

loss_pts = []
acc_pts = []
train_acc_pts = []

niter = 0
niter_test = 0
niter_trainacc = 0
loss_max = 0
for l in f:
    if "solver.cpp:244" in l and "Train net output #1: loss3/loss3" in l:
        # TRAINING LOSS
        loss = float( l.strip().split("=")[1].split("(")[0].strip() )
        loss_pts.append( (niter,loss) )
        niter += iters_per_point
        if loss_max<loss:
            loss_max = loss
    elif "Test net output #2: loss3/top-1" in l:
        # VAL ACC
        acc = float( l.strip().split("=")[1] )
        acc_pts.append( (niter_test,acc) )
        niter_test += test_iters_per_point
    elif "Train net output #2: loss3/top-1" in l:
        # TRAIN ACC
        train_acc = float( l.strip().split("=")[1] )
        train_acc_pts.append( (niter_trainacc, train_acc ) )
        niter_trainacc += iters_per_point

gloss = rt.TGraph( len(loss_pts) )
for n,pt in enumerate(loss_pts):
    gloss.SetPoint(n, float(pt[0])/nevents_per_epoch, pt[1])

gacc = rt.TGraph( len(acc_pts) )
for n,pt in enumerate(acc_pts):
    gacc.SetPoint(n,float(pt[0])/nevents_per_epoch, pt[1])

gacc_train = rt.TGraph( len(train_acc_pts) )
for n,pt in enumerate(train_acc_pts):
    gacc_train.SetPoint(n,float(pt[0])/nevents_per_epoch, pt[1])

gloss.Write("gloss")
gacc.Write("gacc_test")
gacc_train.Write("gacc_train")

c = rt.TCanvas("c","c",1200,600)
c.Divide(2,1)
c.cd(1).SetLogy(1)
c.cd(1).SetGridx(1)
c.cd(1).SetGridy(1)
gloss.Draw("ALP")
gloss.SetTitle("Loss Curve;epochs;loss")
gloss.GetYaxis().SetRangeUser(0.1e-2,1.1*loss_max)
c.cd(2)
c.cd(2).SetGridx(1)
c.cd(2).SetGridy(1)
gacc_train.Draw("ALP")
gacc_train.GetYaxis().SetRangeUser(0,1)
gacc_train.SetTitle("Accuracy;epochs;accuracy")
gacc.SetLineColor(rt.kRed)
gacc.Draw("LP")
c.SaveAs("training_plot.png")


print "number of points: ",len(loss_pts)," nepochs=",float(niter)/nevents_per_epoch

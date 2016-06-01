import caffe
import numpy as np
import ROOT as rt
from math import log
from array import array
from larcv import larcv
import time

# This script setups caffe in TEST mode

gpu_id = 1
caffe.set_device(gpu_id)
caffe.set_mode_gpu()

prototxt = "bvlc_googlenet_test.prototxt"
#model = "snapshot_rmsprop_googlenet_iter_100000.caffemodel"
model = "snapshot_rmsprop_googlenet_iter_32500.caffemodel"
out_tag = "bnb2"
#rootfile = "/mnt/disk1/production/v04/train_sample/val_filtered.root"
#rootfile="/mnt/disk1/production/v04/adcscale/data_extbnb/extbnb_part00.root"
#rootfile="/mnt/disk1/production/v04/adcscale/data_bnb/extbnb_part00.root"
rootfile="/mnt/disk1/production/v04/adcscale/data_bnb/bnb_part00.root"


net = caffe.Net( prototxt, model, caffe.TEST )
input_shape = net.blobs["data"].data.shape
batch_size = input_shape[0]
binlabels = {0:"background",1:"neutrino"}
classlabels = binlabels.keys()

nevents = 10000
store_pe = True
draw_pmt_wfm = False

# setup input
# ROOT data
ioman = larcv.IOManager(larcv.IOManager.kREAD,"IOMan2")
ioman.add_in_file( rootfile )
ioman.initialize()

print "Network Ready: Batch Size=",batch_size
print "[ENTER] to continue."
raw_input()


# setup output

out = rt.TFile("out_%s_netanalysis.root"%(out_tag), "RECREATE" )
entry = array('i',[0])
label = array('i',[0])
nuprob = array('f',[0.0])
winpe  = array('f',[0.0])

tree = rt.TTree("net","net output")
tree.Branch("entry",entry,"entry/I")
tree.Branch("label",label,"label/I")
tree.Branch("nuprob",nuprob,"nuprob/F")
tree.Branch("winpe",winpe,"winpe/F")

nbatches = nevents/batch_size
if nevents%batch_size!=0:
    nbatches += 1
filler = larcv.ThreadFillerFactory.get_filler("test")

ibatch = 0
ievents = 0
inu = 0

if draw_pmt_wfm:
    c = rt.TCanvas("c","c",800,600)
    c.Draw()

while ibatch<nbatches:
    print "batch ",ibatch," of ",nbatches
    keys = []

    net.forward()
    labels =  net.blobs["label"].data
    probs = net.blobs["probt"].data
    processed = filler.processed_entries()
    print processed.size()

    for ientry,evlabel,prob in zip(processed,labels,probs):
        label[0] = int(evlabel)
        nuprob[0] = prob[1]
        entry[0] = ientry
        print ientry,evlabel,prob
        
        if store_pe:
            ioman.read_entry( ientry )
            peimg = ioman.get_data( larcv.kProductImage2D, "pmt" )
            pmtwfms = peimg.Image2DArray()[0]
            pmtnd = larcv.as_ndarray( pmtwfms )
            # slice off beam window
            if draw_pmt_wfm:
                beamwin = np.copy( pmtnd[:,190:310] )
            else:
                beamwin = pmtnd[:,190:310]
            # remove pedestal
            beamwin -= (2048+10) # 0.5 pe-ish threshold
            # remove undershooot
            beamwin[ beamwin<0 ] = 0.0
            # sum pe
            chsum = np.sum( beamwin, axis=1 )
            chsum /= 100.0
            maxch = np.argmax( chsum )
            totsum = np.sum( chsum )
            winpe[0] = totsum
            print entry[0],label[0],nuprob[0],winpe[0]
            if draw_pmt_wfm:
                h = rt.TH1D("hwfm","",120,0,120)
                print chsum
                print "max channel integral: ",chsum[maxch]
                print "total sum: ",totsum
                for i in range(0,120):
                    h.SetBinContent( i+1, pmtnd[maxch,190+i]-2048.0 )
                h.Draw()
                c.Update()
                raw_input()

        else:
            winpe[0] = 0.0

        tree.Fill()
        ievents += 1
        if nuprob[0]>0.5:
            inu+=1

    print "nu fraction: ",float(inu)/float(ievents)
    ibatch += 1
    
out.Write()
    

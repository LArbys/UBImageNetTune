import sys,os
import caffe
import numpy as np
import ROOT as rt
from math import log
from larcv import larcv
import cv2
import time

# Params
gpu_id = 0
nbatches = 1
save_data_blobs = False

outfolder = "images_placedin_extcrop"
os.popen("mkdir -p %s"%(outfolder))
os.popen("rm %s/*"%(outfolder))

caffe.set_mode_gpu()
caffe.set_device(gpu_id)

lowerbounds = [0.5,0.45,0.5]
upperbounds = [4.5,4.55,4.5]

#caffe.set_mode_cpu()

eventlist = []
with open("eventlist_ext_crop.txt",'r') as f:
    lines = f.readlines()
    for l in lines:
        if l.strip()!='' and int(l.strip()) not in eventlist:
            eventlist.append( int(l.strip()) )

#eventlist = range(0,1)

print "EVENTLIST: ",len(eventlist)

box_width = 107
image_cols = 856
image_rows = 756
offset = [0,0]
origins = []
for ix in range(offset[0],image_cols,box_width/2):
    for iy in range(offset[1],image_rows,box_width/2):
        origins.append( (ix,iy) )
nboxes = len(origins)
print "NUM boxes: ",len(origins)

# network setup

prototxt = "bvlc_googlenet_memlayer.prototxt"
#model    = "snapshot_rmsprop_googlenet_iter_32500.caffemodel"
model    = "snapshot_rmsprop_googlenet_iter_62500.caffemodel"
rootfile = "/mnt/disk1/production/v04/train_sample/val_filtered.root"
#rootfile = "/mnt/disk1/production/v04/adcscale/data_bnb/bnb_part00.root"


net = caffe.Net( prototxt, model, caffe.TEST )

binlabels = {0:"background",1:"neutrino"}
classlabels = binlabels.keys()

input_shape = net.blobs["data"].data.shape
batch_size  = net.blobs["data"].data.shape[0]

# ROOT data
ioman = larcv.IOManager(larcv.IOManager.kREAD,"IOPLACEIN")
ioman.add_in_file( rootfile )
ioman.initialize()

print "INPUT SHAPE: ",input_shape
print "box steps: ",image_rows,image_cols
print "[ENTER] to continue."
raw_input()


# setup output
out = rt.TFile("output_maskanalysis.root", "RECREATE" )

input_labels = np.zeros( (input_shape[0],), dtype=np.float32 )

# work space arrays
orig_data = np.zeros( (input_shape[1],input_shape[2],input_shape[3]), dtype=np.float32 )
data = np.zeros( input_shape, dtype=np.float32 )
label = np.ones( (input_shape[0],), dtype=np.float32 )
prob_array = np.zeros( (input_shape[2],input_shape[3],3), dtype=np.float )
norm_array = np.zeros( (input_shape[2],input_shape[3],3), dtype=np.float )
net.set_input_arrays( data, label )

outofentries = False
eventlist.sort()
for entry in eventlist:
    # one image at a time
    ioman.read_entry(entry)
    print "Process entry: ",entry
    print "EVENT LIST remaining: ",len(eventlist)

    # clear the image
    data[...] = 0.0
    label[...] = 1.0
    orig_data[...] = 0.0
    
    # get the images
    event_images = ioman.get_data(larcv.kProductImage2D,"tpc")
    event_rois   = ioman.get_data(larcv.kProductROI, "tpc")

    # copy image data into data
    images = {}
    for img in event_images.Image2DArray():
        imgnd = larcv.as_ndarray(img)
        imgnd = np.transpose( imgnd, (1,0) )
        imgnd -= lowerbounds[img.meta().plane()]
        imgnd[ imgnd<0 ] = 0.0
        imgnd[ imgnd>upperbounds[img.meta().plane()] ] = upperbounds[img.meta().plane()]
        x1 = int( 0.5*( input_shape[2]-image_rows ) )
        y1 = int( 0.5*( input_shape[3]-image_cols ) )
        orig_data[img.meta().plane(),:,:] = imgnd[x1:x1+image_rows,y1:y1+image_cols]
        images[img.meta().plane()] = img

    # get ROI
    event_label = 1
    bnb_roi = None
    for roi in event_rois.ROIArray():
        if roi.Type()==larcv.kROICosmic:
            event_label = 0
            break
        elif roi.Type()==larcv.kROIBNB:
            event_label = 1
            bnb_roi = roi
            break
    label *= float(event_label)

    # fill data, get original prob
    for i in range(0,batch_size):
        data[i,...] = orig_data[...]
        if i==batch_size-1 and batch_size>1:
            data[i,...] = 0.0
    print "RUN ORIGINAL FORWARD"
    net.forward()

    orig_probs = net.blobs["probt"].data[0]
    print "SCORE: ",orig_probs

    if batch_size>1:
        baseline_probs = net.blobs["probt"].data[-1]
    else:
        data[0,...] = 0.0
        net.forward()
        baseline_probs = net.blobs["probt"].data[0]
        
    print "Event",entry,": original prob=",orig_probs," baseline=",baseline_probs," label=",label

    # now clear out and do place-in analysis
    nchannels = [0,1,2]
    prob_array[...] = 0.0
    norm_array[...] = 0.0
    
    for ichannel in nchannels:

        data[...] = 0.0

        nbatches = (nboxes)/batch_size
        if (nboxes)%batch_size!=0:
            nbatches += 1

        # we do multiple crops for each image
        for ibatch in range(nbatches):
            start = ibatch*batch_size
            end   = (ibatch+1)*batch_size
            if end>nboxes:
                end = nboxes

            # now make blanked out regions
            nfilled = 0
            for origin in origins[start:end]:
                data[nfilled,:,:,:] = 0.0
                x1 = origin[0]-offset[0]
                x2 = np.minimum(x1 + box_width,data.shape[2]-offset[0])
                y1 = origin[1]
                y2 = np.minimum(y1 + box_width,data.shape[3])
                data[nfilled,ichannel,x1:x2,y1:y2] += orig_data[ichannel,x1:x2,y1:y2]
                input_labels[nfilled] = event_label
                nfilled += 1

            print "pushed batch ",ibatch," of ",nbatches," channel=",ichannel
            net.forward()
            for iorigin,origin in enumerate(origins[start:end]):
                probs = net.blobs["probt"].data[iorigin]
                print "origin[%d]: "%(iorigin),origin," prob=",probs
                # past into result matrix if prob is big
                x1 = origin[0]
                x2 = np.minimum(x1 + box_width,data.shape[2])
                y1 = origin[1]
                y2 = np.minimum(y1 + box_width,data.shape[3])
                prob_array[x1:x2,y1:y2,ichannel] += (probs[1]-baseline_probs[1])*128/4.0
                norm_array[x1:x2,y1:y2,ichannel] += 1.0
            
            # batches over for channel
            # normal array
            #idx = norm_array[:,:,ichannel] > 0
            #prob_array[:,:,ichannel][idx] /= norm_array[:,:,ichannel][idx]

    # make output image
    output_img = np.zeros( prob_array.shape )
    for i in range(0,3):
        output_img[:,:,i] = orig_data[i,:,:]*255.0
    output_img += prob_array
    
    # save
    if bnb_roi is not None:
        colors = {0:(255,0,0),
                  1:(0,255,0),
                  2:(0,0,255)}
        for meta in bnb_roi.BB():
            p = int(meta.plane())
            print p,meta.plane()
            img = images[p]
            x = meta.min_x() - img.meta().min_x()
            y = meta.min_y() - img.meta().min_y()
            dw_i = img.meta().cols()/( img.meta().max_x()-img.meta().min_x() )
            dh_i = img.meta().rows()/( img.meta().max_y()-img.meta().min_y() )

            w_b = meta.max_x()-meta.min_x()
            h_b = meta.max_y()-meta.min_y()

            x1 = int(x*dw_i)
            y1 = int(y*dw_i)
            x2 = int( (x+w_b)*dw_i )
            y2 = int( (y+h_b)*dh_i )
            
            cv2.rectangle( output_img, 
                           (x1,y1),
                           (x2,y2),
                           colors[p], 2 )
    cv2.imwrite( "%s/entry_%d.PNG"%(outfolder,entry), output_img )

    
    

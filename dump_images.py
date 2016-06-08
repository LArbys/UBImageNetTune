import sys,os
import numpy as np
import ROOT as rt
from math import log
from larcv import larcv
import cv2
import time

# This is just a list of entry numbers from a 'net' tree that we wish to look at.
eventfile = "eventlist_bnb_99_winpe100.txt"
outfolder = "images_bnb_99_winpe100"
os.popen("mkdir -p %s"%(outfolder))
os.popen("rm %s/*"%(outfolder))

# The LArCV Root file
rootfile = "/mnt/disk1/production/v04/train_sample/pmt_weight_bnb.root"

# Bounds to help setup the contrast
lowerbounds = [0.3,0.35,0.3]
upperbounds = [4.5,4.55,4.5]

# Make a list of integers which contain the entry numbers from the text files above
eventlist = []
with open(eventfile,'r') as f:
    lines = f.readlines()
    for l in lines:
        if l.strip()!='' and int(l.strip()) not in eventlist:
            eventlist.append( int(l.strip()) )

#eventlist = range(0,1)

print "EVENTLIST: ",len(eventlist)


# Setup the IOManager
ioman = larcv.IOManager(larcv.IOManager.kREAD,"IOPLACEIN") # constructor
ioman.add_in_file( rootfile ) # specify rootfile(s) to open
ioman.initialize() # init

# details of the shape of our input image size
image_cols = 856
image_rows = 756
input_shape = (20, 3, 756, 856)

print "[ENTER] to continue."
raw_input()


# setup output
out = rt.TFile("output_maskanalysis.root", "RECREATE" )

eventlist.sort()
for entry in eventlist:
    # load data for a given entry number
    ioman.read_entry(entry)
    print "Process entry: ",entry
    print "EVENT LIST remaining: ",len(eventlist)
    
    # get the images
    event_images = ioman.get_data(larcv.kProductImage2D,"tpc")   # gets the TPC image
    pmtweight_images = ioman.get_data(larcv.kProductImage2D,"pmtweight") # gets the PMT-weighted TPC image
    event_rois   = ioman.get_data(larcv.kProductROI, "tpc") # gets the Region-of-Interest list, container for particle meta-data

    # is this a cosmic or neutrino?
    # go through ROI list
    truth_label = 1
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
    truth_label *= float(event_label)

    orig_img = np.zeros( (input_shape[2],input_shape[3],input_shape[1]) )
    pmt_img = np.zeros( (input_shape[2],input_shape[3],input_shape[1]) )

    # copy image data into data
    for iimg in range(0,3):
        img  = event_images.Image2DArray()[iimg]
        pmtimg  = pmtweight_images.Image2DArray()[iimg]
        imgnd = larcv.as_ndarray(img)
        imgnd = np.transpose( imgnd, (1,0) )
        imgnd -= lowerbounds[img.meta().plane()]
        imgnd[ imgnd<0 ] = 0.0
        imgnd[ imgnd>upperbounds[img.meta().plane()] ] = upperbounds[img.meta().plane()]

        pmtnd = larcv.as_ndarray(pmtimg)
        pmtnd = np.transpose( pmtnd, (1,0) )


        x1 = int( 0.5*( input_shape[2]-image_rows ) )
        y1 = int( 0.5*( input_shape[3]-image_cols ) )
        orig_img[:,:,img.meta().plane()] = imgnd[x1:x1+image_rows,y1:y1+image_cols]*100
        pmt_img[:,:,img.meta().plane()]  = pmtnd[x1:x1+image_rows,y1:y1+image_cols]*100


    cv2.imwrite( "%s/entry_%d.PNG"%(outfolder,entry), orig_img )
    #cv2.imwrite( "%s/entry_%d_pmt.PNG"%(outfolder,entry), pmt_img )

    
    

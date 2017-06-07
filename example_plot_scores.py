# we import the python modules we need
import os         # methods to unix commands/file system paths
import sys        # python system methods
import ROOT as rt # ROOT python bindings

# check command line arguments
# -------------------------------------
# 1) are they the right number?
if len(sys.argv)!=2:
    # if not the right number of arguments, we provide a hint to run the python script
    print "usage: python plot_scores.py [net output root file]"
    sys.exit(-1) # stop program    
# 2) get the arguments:
rootfile_name = sys.argv[1] # filename of ROOT file

# check if path points to file and if it exists:
if not os.path.isfile(rootfile_name) or not os.path.exists(rootfile_name):
    raise ValueError("Could not find the root file") # instead of stopping script, we throw an exception

# setup the ROOT file
# -------------------------------------

#open the root file:
rfile = rt.TFile( rootfile_name, "OPEN" )

# we get the TTree:
# note, for any file, to see what is in it go into the ROOT prompt and type
# > .ls
# for the type of file we're assuming (the one output by analyze_test_data.py) you'll get this:
"""
twongjirad@tmw-Blade:~/working/uboone/ubimagenettune$ root out_valid_netanalysis.root 
root [0] 
Attaching file out_valid_netanalysis.root as _file0...
(TFile *) 0x55bfd0f29010
root [1] .ls
TFile**        out_valid_netanalysis.root
TFile*         out_valid_netanalysis.root
KEY: TTree     net;1           net output
root [2]
"""
# we want the 'net' TTree
net = rfile.Get("net")

# the branches of the tree (that is the columns of data it stores) can
# be found also by looking into the prompt and doing
"""
root [2] net->Print()
******************************************************************************
*Tree    :net       : net output                                             *
*Entries :    10000 : Total =          162828 bytes  File  Size =      84112 *
*        :          : Tree compression factor =   1.92                       *
******************************************************************************
*Br    0 :entry     : entry/I                                                *
*Entries :    10000 : Total  Size=      40616 bytes  File Size  =      14157 *
*Baskets :        2 : Basket Size=      32000 bytes  Compression=   2.84     *
*............................................................................*
*Br    1 :label     : label/I                                                *
*Entries :    10000 : Total  Size=      40616 bytes  File Size  =        382 *
*Baskets :        2 : Basket Size=      32000 bytes  Compression= 105.08     *
*............................................................................*
*Br    2 :nuprob    : nuprob/F                                               *
*Entries :    10000 : Total  Size=      40622 bytes  File Size  =      36610 *
*Baskets :        2 : Basket Size=      32000 bytes  Compression=   1.10     *
*............................................................................*
*Br    3 :winpe     : winpe/F                                                *
*Entries :    10000 : Total  Size=      40616 bytes  File Size  =      32318 *
*Baskets :        2 : Basket Size=      32000 bytes  Compression=   1.24     *
*............................................................................*
"""

# now we loop through the events and make a histogram of the scores
# -----------------------------------------------------------------

# GOAL: we want plot a histogram for true neutrino events and true cosmic events.
# the score output by the network for each events is in the 'nuprob' branch.
# the flag for neutrino or cosmic events is in 'label'. label==0 if cosmic. label==1 if neutrino

# first we have to create the histograms we want to make
hnuprob_nu     = rt.TH1D("hnuprob_nu","Neutrino Score;score;counts",100,0,1.0)
hnuprob_cosmic = rt.TH1D("hnuprob_bg","Neutrino Score;score;counts",100,0,1.0)

# counter for entry number
ientry = 0

# the TTree acts as our bridge between the data on disk
# and the data stored in variables we can use in our program

# ask the tree to read entry 0.
# tree returns the number of bytes read from disk
# note: if bytes_read==0. We have reached the end of the file. we use this condition to control the event loop.
bytes_read = net.GetEntry(ientry)

while bytes_read>0:
    #print ientry,net.label
    
    # we fill either the neutrino or cosmic histogram depending on the label variable
    if net.label==0:
        hnuprob_cosmic.Fill( net.nuprob )
    elif net.label==1:
        hnuprob_nu.Fill( net.nuprob )

    # increment the entry counter
    ientry += 1
        
    # we get the next event
    bytes_read = net.GetEntry( ientry )

# now we want to make a plot with our histograms
# ----------------------------------------------

# first make a canvas to draw on
canv = rt.TCanvas("canv","Neutrino Scores",800,600)

# now we draw the neutrino histogram on it
hnuprob_nu.Draw()

# we want to overlay the cosmic histogram on it
hnuprob_cosmic.Draw("same")

# we need to make them different colors to know which one is which
hnuprob_nu.SetLineColor( rt.kRed )
hnuprob_cosmic.SetLineColor( rt.kBlack )

# we should make a nice label to indicate what is what
legend = rt.TLegend( 0.4, 0.9, 0.6, 0.7 ) # arguments give corners of legend, (x1,y1,x2,y2), in fraction of x and y axis. e.g. 0.5=half way up axis
legend.AddEntry( hnuprob_nu, "neutrino", "L" ) # 'L' is for line, which is what is drawn on legend. Look up other options in online ROOT documentation, if interested.
legend.AddEntry( hnuprob_cosmic, "cosmic", "L" )

# overlay legend into our plot
legend.Draw("same")

# update the canvas and draw it
canv.Update()
canv.Draw()

# pause the program so we can enjoy the plot
# note, we should be able to interact with the axis, move the legend
print "[enter] to continue"
raw_input()

# let's save the plot
canv.SaveAs("myplot.png")


    


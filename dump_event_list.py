import os,sys
import ROOT as rt

out = open('eventlist.txt','w')

fname = sys.argv[1]

fin = rt.TFile(fname,"open")
tree = fin.Get("net")

for n in range(0,tree.GetEntries()):
    tree.GetEntry(n)
    if tree.label==0 and tree.nuprob>0.99 and tree.winpe>100:
        print >> out,tree.entry

out.close()
        

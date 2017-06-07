// ROOT Macro Example
// To run this

/*
twongjirad@tmw-Blade:~/working/uboone/ubimagenettune$ root
root [0] .L example_plot_scores.C
warning: expression result unused [-Wunused-value]
root [1] plot_scores("out_test_netanalysis.root")
*/

void plot_scores(std::string rootfile_name) {

  // setup the ROOT file
  // -------------------------------------

  //open the root file:
  TFile rfile( rootfile_name.c_str(), "OPEN" );

  // we get the TTree:
  // note, for any file, to see what is in it go into the ROOT prompt and type
  // > .ls
  // for the type of file we're assuming (the one output by analyze_test_data.py) you'll get this:
  /*
    twongjirad@tmw-Blade:~/working/uboone/ubimagenettune$ root out_valid_netanalysis.root 
    root [0] 
    Attaching file out_valid_netanalysis.root as _file0...
    (TFile *) 0x55bfd0f29010
    root [1] .ls
    TFile**        out_valid_netanalysis.root
    TFile*         out_valid_netanalysis.root
    KEY: TTree     net;1           net output
    root [2]
  */

  // we want the 'net' TTree
  TTree* net = (TTree*)rfile.Get("net");

  // check its OK
  if ( net==NULL ) {
    std::cout << "Tree is bad." << std::endl;
    return;
  }

  // the branches of the tree (that is the columns of data it stores) can
  // be found also by looking into the prompt and doing
  /*
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
  */

  // setup the branches to output data values for each entry to specific variables
  int label;
  float nuprob;
  net->SetBranchAddress("label",&label);
  net->SetBranchAddress("nuprob",&nuprob);

  // now we loop through the events and make a histogram of the scores
  // -----------------------------------------------------------------

  // GOAL: we want plot a histogram for true neutrino events and true cosmic events.
  // the score output by the network for each events is in the 'nuprob' branch.
  // the flag for neutrino or cosmic events is in 'label'. label==0 if cosmic. label==1 if neutrino

  // first we have to create the histograms we want to make
  TH1D hnuprob_nu("hnuprob_nu","Neutrino Score;score;counts",100,0,1.0);
  TH1D hnuprob_cosmic("hnuprob_bg","Neutrino Score;score;counts",100,0,1.0);

  // counter for entry number
  unsigned long ientry = 0;

  // the TTree acts as our bridge between the data on disk
  // and the data stored in variables we can use in our program

  // ask the tree to read entry 0.
  // tree returns the number of bytes read from disk
  // note: if bytes_read==0. We have reached the end of the file. we use this condition to control the event loop.
  bytes_read = net->GetEntry(ientry);

  while (bytes_read>0) {
    //std::cout <<  ientry << " " << label << std::endl;
    
    // we fill either the neutrino or cosmic histogram depending on the label variable
    if (label==0)
      hnuprob_cosmic.Fill( nuprob );
    else if (label==1)
      hnuprob_nu.Fill( nuprob );

    // increment the entry counter
    ientry++;
        
    /// we get the next event
    bytes_read = net->GetEntry( ientry );
  }

  // now we want to make a plot with our histograms
  // ----------------------------------------------

  // first make a canvas to draw on
  TCanvas* canv = new TCanvas("canv","Neutrino Scores",800,600); // note this a pointer which gets asigned a new object from the heap
  // if 'heap' is an unfamilar word, you might want to look up something like 'stack versus heap c++' on google.
  // this script is not the place to describe this important distinction about how your variables reside in memory.

  // now we draw the neutrino histogram on it
  hnuprob_nu.Draw();

  // we want to overlay the cosmic histogram on it
  hnuprob_cosmic.Draw("same");

  // we need to make them different colors to know which one is which
  hnuprob_nu.SetLineColor( kRed );
  hnuprob_cosmic.SetLineColor( kBlack );

  // we should make a nice label to indicate what is what
  TLegend legend ( 0.4, 0.9, 0.6, 0.7 ); // arguments give corners of legend, (x1,y1,x2,y2), in fraction of x and y axis. e.g. 0.5=half way up axis
  // note that the legend object wants a pointer to the histograms.
  // pointers are variables that stores not the object itself, but an address to the location in memory where the object resides.
  // ex.
  // TH1D h; // a histogram object
  // &h; // the address of the object. If you print this value to screen (cout << &h << endl;) you'll get something like 0x1fdf034, which is an address to memory location.
  // TH1D* p_h; // a pointer. it holds addresses
  // p_h = &h; // assigning the address to the pointer
  // *p_h; // this returns the object itself (* is the 'dereferencing operator')
  TH1D* p_hnuprob_nu = &hnuprob_nu;
  legend.AddEntry( p_hnuprob_nu, "neutrino", "L" ); // 'L' is for line, which is what is drawn on legend. Look up other options in online ROOT documentation, if interested.
  legend.AddEntry( &hnuprob_cosmic, "cosmic", "L" ); // we don't have to make intermeidate pointer variable, we can just pass the address as an argument

  // overlay legend into our plot
  legend.Draw("same");

  // update the canvas and draw it
  canv->Update(); // note that when you want to use the functions of a class pointed to by pointer, you use the '->' not '.'
  // the above is short hand for
  (*canv).Update(); // i.e. the pointer is dereferenced to the object, then the fucntion Update() is called.
  
  canv->Draw();

  // pause the program so we can enjoy the plot
  // note, we should be able to interact with the axis, move the legend
  std::cout <<  "[enter] to continue" << std::endl;
  std::cin.get();

  // let's save the plot
  canv->SaveAs("myplot.png");

  // we're done. Note that we have to free any variable created by 'new'
  delete canv;

  return;
}


    


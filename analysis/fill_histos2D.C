#include "fill_histos.h"
#include "helper_h2d.h"

TObjArray* Get_HistArray_h2d(const char* file_nm) {

  TObjArray* HistArray = new TObjArray(100); // Hist. Array

  // If data file exists, process it with RDataFame
  if(!gSystem -> AccessPathName(file_nm)){

    cout << "\nProcessing data file: " << file_nm << endl;
    
    // Check input root file and list all trees
    TFile* file = new TFile(file_nm);
    
    check_trees(file);

    // Fill histos

    for (int i = 0; i < 10; i ++) {
      // chi-2 selection
      TH2D* h2d = (TH2D*)file -> Get(hist_type + TString("_") + mc_names[i]);
      TH2D* h2d_good = (TH2D*)file -> Get(hist_type + TString("_good_") + mc_names[i]);
      TH2D* h2d_bad = (TH2D*)file -> Get(hist_type + TString("_bad_") + mc_names[i]);
      
      //cout << h2d -> GetName() << endl;

      // bdt selection
      TH2D* h2d_bdt = (TH2D*)file -> Get(hist_type + TString("_BDT_") + mc_names[i]);
      //cout << h2d_bdt -> GetName() << endl;

      // bdt best
      TH2D* h2d_bdt_good = (TH2D*)file -> Get(hist_type + TString("_BDT_good_") + mc_names[i]);
      //cout << h2d_bdt_good -> GetName() << endl;

      // bdt discarded
      TH2D* h2d_bdt_bad = (TH2D*)file -> Get(hist_type + TString("_BDT_bad_") + mc_names[i]);
      //cout << h2d_bdt_bad -> GetName() << endl;

      // Fill HistArray_m3pi
      HistArray -> Add(h2d);
      HistArray -> Add(h2d_good);
      HistArray -> Add(h2d_bad);
      
      HistArray -> Add(h2d_bdt);
      HistArray -> Add(h2d_bdt_good);
      HistArray -> Add(h2d_bdt_bad);
      
    }
    
  }// end input checks

  return HistArray;

}

int fill_histos2D(const char* input_filename = "./output_with_bdt.root") {

  gErrorIgnoreLevel = kError;
  TGaxis::SetMaxDigits(4);
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  gStyle->SetFitFormat("6.4g");

  TObjArray* HistArray_h2d = Get_HistArray_h2d(input_filename); //new TObjArray(100); // Hist. Array


  return 0;

}

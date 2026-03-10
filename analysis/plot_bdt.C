#include "helper.h"

TCanvas *cv_plot(TH1D* h_all, TH1D* h_bdt, TH1D* h_good_bdt, TH1D* h_bad_bdt, TString y_title, TString x_title) {

  TCanvas *cv0 = new TCanvas("c1", "Invariant mass of 3pi", 700, 600);
    cv0 -> SetLeftMargin(0.1);
    cv0 -> SetBottomMargin(0.1);//0.007

    //double ymax = h_all -> GetBinContent(h_all -> GetMaximumBin());
    const double ymax = 300.;
    const double xmin = 650., xmax = 1000.;
    
    h_all -> GetYaxis() -> SetTitle(y_title);
    h_all -> GetYaxis() -> CenterTitle();
    h_all -> GetYaxis() -> SetTitleSize(0.04);
    h_all -> GetYaxis() -> SetNdivisions(505);
    h_all -> GetYaxis() -> SetRangeUser(0.01, ymax * 1.5); 
    h_all -> GetXaxis() -> SetTitle(x_title);
    h_all -> GetXaxis() -> CenterTitle();
    h_all -> GetXaxis() -> SetTitleSize(0.04);
    h_all -> GetXaxis() -> SetRangeUser(xmin, xmax); 
   
    h_all -> Draw("E");
    h_good_bdt -> Draw("Same");
    h_bad_bdt -> Draw("Same");
    h_bdt -> Draw("Same");
    //gPad->SetLogy(1); 
      
    TLegend *legd_cv = new TLegend(0.5, 0.55, 0.9, 0.9);
      
    legd_cv -> SetTextFont(132);
    legd_cv -> SetFillStyle(0);
    legd_cv -> SetBorderSize(0);
    legd_cv -> SetNColumns(1);
    
    legd_cv -> AddEntry(h_all, "KLOE Selected", "l");
    legd_cv -> AddEntry(h_good_bdt, "BDT Selected", "l");
    legd_cv -> AddEntry(h_bad_bdt, "BDT Comb. BKG", "f");
    
    legd_cv -> Draw("Same");
    
    legtextsize(legd_cv, 0.04);
    
    return cv0;
}

void plot_bdt(const char* input_filename = "./output_main_bdt.root") {

  gErrorIgnoreLevel = kError;
  TGaxis::SetMaxDigits(4);
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  gStyle->SetFitFormat("6.4g");
  
  cout << "Plotting BDT results ... " << endl;

  // Load input root file: output_with_bdt.root
  std::cout << "Loading input " << input_filename << std::endl;  // Added std::

  // If data file exists, process it with RDataFame
  if(!gSystem -> AccessPathName(input_filename)){

    cout << "\nProcessing data file: " << input_filename << endl;
    
    // Open the root file
    TFile* file = TFile::Open(input_filename);
    if (!file || file -> IsZombie()){
      cout << "Error: Cannot open file " << input_filename << endl;
      return;
    }// end open root file
    
    // Loop over keys
    TIter next_tree(file -> GetListOfKeys());
    
    TString objnm_tree, classnm_tree;
    
    TKey *key;
    
    int ch_nb = 0;
    
    while ((key = (TKey *) next_tree())) { // key loop

      
      objnm_tree = key -> GetName();
      classnm_tree = key -> GetClassName();
      key -> GetSeekKey();

      cout << "classnm = " << classnm_tree << ", objnm = " << objnm_tree << endl;
      
    }

    // Get histos
    file -> cd(); // Make sure we're in the output file
    // TISR3PI_SIG
    TH1D* hM3pi_TISR3PI_SIG = (TH1D*)file -> Get("hM3pi_TISR3PI_SIG");
    TH1D* hM3pi_BDT_TISR3PI_SIG = (TH1D*)file -> Get("hM3pi_BDT_TISR3PI_SIG");
    TH1D* hM3pi_BDT_good_TISR3PI_SIG = (TH1D*)file -> Get("hM3pi_BDT_good_TISR3PI_SIG");
    TH1D* hM3pi_BDT_bad_TISR3PI_SIG = (TH1D*)file -> Get("hM3pi_BDT_bad_TISR3PI_SIG");

    format_h(hM3pi_TISR3PI_SIG, 4, 2);
    format_h(hM3pi_BDT_TISR3PI_SIG, 2, 2);
    format_h(hM3pi_BDT_good_TISR3PI_SIG, 3, 2);
    formatfill_h(hM3pi_BDT_bad_TISR3PI_SIG, 2, 3001);

    //
    TH1D* hM3pi_TDATA = (TH1D*)file -> Get("hM3pi_TDATA");
    TH1D* hM3pi_BDT_TDATA = (TH1D*)file -> Get("hM3pi_BDT_TDATA");
    TH1D* hM3pi_BDT_good_TDATA = (TH1D*)file -> Get("hM3pi_BDT_good_TDATA");
    TH1D* hM3pi_BDT_bad_TDATA = (TH1D*)file -> Get("hM3pi_BDT_bad_TDATA");

    format_h(hM3pi_TDATA, 1, 2);
    format_h(hM3pi_BDT_TDATA, 2, 2);
    format_h(hM3pi_BDT_good_TDATA, 3, 2);
    formatfill_h(hM3pi_BDT_bad_TDATA, 2, 3001);

    // Plot
    TCanvas *cv_3pi = cv_plot(hM3pi_TDATA, hM3pi_BDT_TDATA, hM3pi_BDT_good_TDATA, hM3pi_BDT_bad_TDATA, "Events", "M_{3#pi} [MeV/c^{2}]");
    
    //int nentries = outtree -> GetEntries();
    //cout << "Tree has " << nentries << " entries" << endl;
    
    
    // Set branch addres for input features
    //double 
    
  }// end check input file existence.
    
    
}

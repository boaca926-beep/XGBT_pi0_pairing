#include "helper.h"
#include "helper_m3pi.h"
//#include "helper_mgg.h"

double getbinwidth(TH1D* h) {
  Int_t binsize=0;
  double width=0.;
  double xmax=0., xmin=0.;
  xmax = h->GetXaxis()->GetXmax(); //cout<<xmax<<endl;
  xmin = h->GetXaxis()->GetXmin(); //cout<<xmin<<endl;
  binsize=h->GetNbinsX(); //cout<<"binsize = " << binsize << endl;
  width=(xmax-xmin)/binsize; //cout<<width<<endl;

  return width;
}


TCanvas *plot_corr(TH2D* h2d, const TString hist_type, const TString cv_nm, const TString pt_str) {

  TH1D *h2d_projx = h2d -> ProjectionX();
  TH1D *h2d_projy = h2d -> ProjectionY();
  
  double binwidth_x = getbinwidth(h2d_projx);
  double binwidth_y = getbinwidth(h2d_projy);
  
  TCanvas * cv2d =  new TCanvas("cv2d_" + hist_type, cv_nm, 0, 0, 700, 700);

  cv2d -> SetBottomMargin(0.15);//0.007
  cv2d -> SetLeftMargin(0.15);
  cv2d -> SetRightMargin(0.15);

  //h2d -> SetMinimum(10);

  h2d -> GetXaxis() -> SetNdivisions(5);
  h2d -> GetXaxis() -> SetTitle("M^{KLOE}_{3#pi} " + TString::Format("Events/[%0.2f", binwidth_x) + " MeV/c^{2}]"); //SetTitle(x_label + " " + x_unit);
  h2d -> GetXaxis() -> SetTitleOffset(1.2);
  h2d -> GetXaxis() -> SetTitleSize(0.06);
  h2d -> GetXaxis() -> CenterTitle();
  h2d -> GetXaxis() -> SetLabelSize(0.06);
  h2d -> GetXaxis() -> SetLabelOffset(0.01);
  //h2d -> GetXaxis() -> SetRangeUser(0.2, 0.6);
  
  h2d -> GetYaxis() -> SetTitle("M^{BDT}_{3#gamma} " + TString::Format("Events/[%0.2f", binwidth_y) + " MeV/c^{2}]"); //SetTitle(x_label + " " + x_unit);
  h2d -> GetYaxis() -> SetLabelOffset(0.01);
  h2d -> GetYaxis() -> SetTitleOffset(1.2);
  h2d -> GetYaxis() -> SetLabelSize(0.05);
  h2d -> GetYaxis() -> SetTitleSize(0.06);
  h2d -> GetYaxis() -> CenterTitle();

  h2d -> Draw("COLZ");
  //h2d -> Draw("TEXT0COLZ");

  gPad->SetLogz(1);

  //char display[50];

  //sprintf(display, "#frac{N_{sig}}{N_{data}-N_{bkg}}");
  
  //TPaveText *pt = new TPaveText(pt1_x0, 0.8, pt1_x1, 0.85, "NDC");

  //pt -> SetTextSize(0.08);
  //pt -> SetFillColor(0);
  //pt -> SetTextAlign(12);
  
  //pt -> AddText("Relative Error [%]");
  //pt -> AddText(pt_str);
  
  //pt -> Draw("Same");
  //line1 -> Draw("Same");
  //line2 -> Draw("Same");

  //h2d -> GetZaxis() -> SetLabelSize(0.045);
  
  return cv2d;
  
}

TLegend* set_legend(const double x1, const double x2, const double y1, const double y2){

  TLegend *legd_cv = new TLegend(x1, x2, y1, y2);

  legd_cv -> SetTextFont(132);
  legd_cv -> SetFillStyle(0);
  legd_cv -> SetBorderSize(0);
  legd_cv -> SetNColumns(1);

  return legd_cv;
}

TPaveText* set_pt(const double x1, const double x2, const double y1, const double y2) {

  // Create TPaveText
  TPaveText* pt = new TPaveText(x1, x2, y1, y2, "NDC");  // Coordinates in NDC (0-1)
  pt->SetFillColor(0);           // White background
  pt->SetTextColor(kBlack);      // Black text
  pt->SetTextSize(0.04);         // Text size
  //pt->SetBorderSize(1);        // Border size (0 = no border)
  pt->SetTextAlign(12);          // Left aligned, vertically centered
  
  return pt;
}



TCanvas* plot_kine_var(TObjArray* HList, const TString var_type, const TString mc_type, const TString cv_title, const TString x_title, const TString unit, const double xmin, const double xmax){

  TH1D *h1d_kloe = (TH1D *) HList -> FindObject(var_type + "_" + mc_type);
  format_h(h1d_kloe, kBlack, 2);
  
  TH1D *h1d_bdt = (TH1D *) HList -> FindObject(var_type + "_BDT_" + mc_type);
  format_h(h1d_bdt, kBlue, 2);
  
  TH1D *h1d_bdt_good = (TH1D *) HList -> FindObject(var_type + "_BDT_good_" + mc_type);
  format_h(h1d_bdt_good, kGreen, 2);
  
  TH1D *h1d_bdt_bad = (TH1D *) HList -> FindObject(var_type + "_BDT_bad_" + mc_type);
  format_h(h1d_bdt_bad, kRed, 2);
  
  // calculate scaling factor
  
  TCanvas* cv =  new TCanvas("cv_" + var_type, cv_title, 0, 0, 1200, 800);

  const double binwidth = getbinwidth(h1d_bdt_good);

  const double ymax = h1d_bdt -> GetBinContent(h1d_bdt -> GetMaximumBin()) * 1.5;

  h1d_bdt -> GetYaxis() -> SetTitle(TString::Format("Events/%0.2f " + unit, binwidth));
  h1d_bdt -> GetYaxis() -> CenterTitle();
  h1d_bdt -> GetYaxis() -> SetTitleSize(0.04);
  h1d_bdt -> GetYaxis() -> SetNdivisions(505);
  h1d_bdt -> GetYaxis() -> SetRangeUser(0.1, ymax);
  h1d_bdt -> GetXaxis() -> SetTitle(x_title + " " + unit);
  h1d_bdt -> GetXaxis() -> CenterTitle();
  h1d_bdt -> GetXaxis() -> SetTitleSize(0.04);
  h1d_bdt -> GetXaxis() -> SetRangeUser(xmin, xmax); 
  
  
  cv -> SetBottomMargin(0.15);//0.007
  cv -> SetLeftMargin(0.15);
  cv -> SetRightMargin(0.15);

  TLegend *legd_cv = set_legend(0.5, 0.6, 0.9, 0.9);
  
  // Create Legend
  if (mc_type == "TDATA") {
    h1d_bdt -> Draw("E");
    h1d_kloe -> Draw("ESame");
    h1d_bdt_good -> Draw("ESame");
    h1d_bdt_bad -> Draw("ESame");

     }
  else{// MC
    TH1D *h1d_kloe_good = (TH1D *) HList -> FindObject(var_type + "_good_" + mc_type);
    //format_h(h1d_kloe_good, 42, 2);
    formatfill_h(h1d_kloe_good, 7, 3001);

    TH1D *h1d_kloe_bad = (TH1D *) HList -> FindObject(var_type + "_bad_" + mc_type);
    //format_h(h1d_bdt_bad, kRed, 2);
    formatfill_h(h1d_kloe_bad, kBlack, 3001);

    h1d_bdt -> Draw("Hist");
    h1d_kloe -> Draw("HistSame");
    h1d_kloe_good -> Draw("HistSame");
    h1d_kloe_bad -> Draw("HistSame");
    
    h1d_bdt_good -> Draw("HistSame");
    h1d_bdt_bad -> Draw("HistSame");

    legd_cv -> AddEntry(h1d_kloe, "#chi^{2}_{m_{#gamma#gamma}} Selection " + TString::Format("(%0.0f)", h1d_kloe -> GetEntries()), "lep");
    legd_cv -> AddEntry(h1d_kloe_good, "#chi^{2}_{m_{#gamma#gamma}} Best " + TString::Format("(%0.0f)", h1d_kloe_good -> GetEntries()), "f");
    legd_cv -> AddEntry(h1d_kloe_bad, "#chi^{2}_{m_{#gamma#gamma}} Discarded " + TString::Format("(%0.0f)", h1d_kloe_bad -> GetEntries()), "f");
  
  }

  
  legd_cv -> AddEntry(h1d_bdt, "BDT Selection " + TString::Format("(%0.0f)", h1d_bdt -> GetEntries()), "lep");
  legd_cv -> AddEntry(h1d_bdt_good, "BDT Best #pi^{0}(#gamma#gamma) " + TString::Format("(%0.0f)", h1d_bdt_good -> GetEntries()), "lep");
  legd_cv -> AddEntry(h1d_bdt_bad, "BDT Discarded " + TString::Format("(%0.0f)", h1d_bdt_bad -> GetEntries()), "lep");
  
  legtextsize(legd_cv, 0.03);

  legd_cv -> Draw("Same");
  

  return cv;
  
}
  
//
void check_trees(TFile* file) {

  if (!file || file -> IsZombie()){
    cout << "Error: Cannot open file " << file -> GetName() << endl;
      return;
    }// end open root file

    TIter next_tree(file -> GetListOfKeys());

    TString objnm_tree, classnm_tree;
    
    TKey *key;
    
    while ( (key = (TKey *) next_tree() ) ) {// start tree while loop
      
      objnm_tree   =  key -> GetName();
      classnm_tree = key -> GetClassName();
      key -> GetSeekKey();

      if (classnm_tree == "TH2D") { // list all TH2D
      //if (classnm_tree == "TH1D") { // list all TH1D
	cout << "classnm = " << classnm_tree << ", objnm = " << objnm_tree << endl;
      }
      
      TTree *tree_tmp = (TTree*)file -> Get(objnm_tree);
      
    }
    
    // Get histos
    file -> cd(); // Make sure we're in the output file

}

int fill_histos(const char* input_filename = "./output_with_bdt.root") {

  gErrorIgnoreLevel = kError;
  TGaxis::SetMaxDigits(4);
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  gStyle->SetFitFormat("6.4g");

  TObjArray* HistArray_m3pi = new TObjArray(100); // Hist. Array
    
  // If data file exists, process it with RDataFame
  if(!gSystem -> AccessPathName(input_filename)){

    cout << "\nProcessing data file: " << input_filename << endl;
    
    // Check input root file and list all trees
    TFile* file = new TFile(input_filename);
    
    check_trees(file);

    const char* mc_names[] = {"TDATA",
			      "TEEG",
			      "TOMEGAPI",
			      "TKPM",
			      "TKSL",
			      "TRHOPI",
			      "TETAGAM",
			      "TBKGREST",
			      "T3PIGAM",
			      "TISR3PI_SIG"
    };
    
    // Fill histos

    for (int i = 0; i < 10; i ++) {
      // chi-2 selection
      TH1D* hm3pi = (TH1D*)file -> Get(hist_type + TString("_") + mc_names[i]);
      TH1D* hm3pi_good = (TH1D*)file -> Get(hist_type + TString("_good_") + mc_names[i]);
      TH1D* hm3pi_bad = (TH1D*)file -> Get(hist_type + TString("_bad_") + mc_names[i]);
      
      //cout << hm3pi -> GetName() << endl;

      // bdt selection
      TH1D* hm3pi_bdt = (TH1D*)file -> Get(hist_type + TString("_BDT_") + mc_names[i]);
      //cout << hm3pi_bdt -> GetName() << endl;

      // bdt best
      TH1D* hm3pi_bdt_good = (TH1D*)file -> Get(hist_type + TString("_BDT_good_") + mc_names[i]);
      //cout << hm3pi_bdt_good -> GetName() << endl;

      // bdt discarded
      TH1D* hm3pi_bdt_bad = (TH1D*)file -> Get(hist_type + TString("_BDT_bad_") + mc_names[i]);
      //cout << hm3pi_bdt_bad -> GetName() << endl;

      // Fill HistArray_m3pi
      HistArray_m3pi -> Add(hm3pi);
      HistArray_m3pi -> Add(hm3pi_good);
      HistArray_m3pi -> Add(hm3pi_bad);
      
      HistArray_m3pi -> Add(hm3pi_bdt);
      HistArray_m3pi -> Add(hm3pi_bdt_good);
      HistArray_m3pi -> Add(hm3pi_bdt_bad);

      // Fill
      
    }
    
  }// end input checks

  //==================================== Plotting =================================
  
  TLine *line11 = new TLine(650., 0., 650., 1500.); // vertical left
  line11 -> SetLineColor(42);
  line11 -> SetLineWidth(4);
  
  TLine *line22 = new TLine(900., 0., 900., 1500.); // vertical right
  line22 -> SetLineColor(42);
  line22 -> SetLineWidth(4);

  const int list_size = 3;

  //const TString ch_type[list_size] = {"TETAGAM"};

  const TString ch_type[list_size] = {"TDATA",
				      "TETAGAM",
				      "TISR3PI_SIG"
  };
  
  for (int i = 0; i < list_size; i++) {
    //
    TCanvas* cv_M3pi = plot_kine_var(HistArray_m3pi, hist_type, ch_type[i], cv_title, x_title, unit, xmin, xmax);

    TPaveText* pt_cut = set_pt(0.1, 0.92, 0.9, 0.98);
    pt_cut -> SetTextColor(42);
    pt_cut -> AddText(Form(pt_cut_text, omega_mass[0], omega_mass[1]));
    //pt_cut -> Draw("same");
    
    TPaveText* pt_tmp = set_pt(0.2, 0.8, 0.3, 0.85);
    pt_tmp -> AddText(ch_type[i]);
    pt_tmp -> Draw("same");
    
    //line11 -> Draw("Same");
    //line22 -> Draw("Same");
    
    cv_M3pi -> SetLogy(1);  // Turn off log scale
    
    cv_M3pi -> Update();
    cv_M3pi -> Modified();
    cv_M3pi -> SaveAs(hist_type + "_" + ch_type[i] + "_full_mass_range.pdf");

    //
    //TCanvas* cv2d_corr = plot_corr("h2dM3pi_kloeBDT_corr_TDATA", "TDATA", "", "Data");
    
  }
    
  return 0;
  
}

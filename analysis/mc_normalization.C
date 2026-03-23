#include "helper.h"
#include "sfw2d.h"

const double pt1_x0 = 0.2;
const double pt1_x1 = 0.3;



TList *HSFW1D = new TList();

void set_cv(TH1D* h1d, const TString x_title, const TString unit, const double xmin, const double xmax, const double binwidth) {

  const double ymax = h1d -> GetBinContent(h1d -> GetMaximumBin()) * 1.5;

  h1d -> GetYaxis() -> SetTitle(TString::Format("Events/%0.2f " + unit, binwidth));
  h1d -> GetYaxis() -> CenterTitle();
  h1d -> GetYaxis() -> SetTitleSize(0.04);
  h1d -> GetYaxis() -> SetNdivisions(505);
  h1d -> GetYaxis() -> SetRangeUser(0.1, ymax);
  h1d -> GetXaxis() -> SetTitle(x_title + " " + unit);
  h1d -> GetXaxis() -> CenterTitle();
  h1d -> GetXaxis() -> SetTitleSize(0.04);
  h1d -> GetXaxis() -> SetRangeUser(xmin, xmax); 
  

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
  pt->SetTextSize(0.05);         // Text size
  //pt->SetBorderSize(1);        // Border size (0 = no border)
  pt->SetTextAlign(12);          // Left aligned, vertically centered
  
  return pt;
}

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

TCanvas* plot_kine_compr(TObjArray* HList, const double sf_tmp, const TString var_type, const TString cv_title){

  TH1D *h1d_bdt_data = (TH1D *) HList -> FindObject(var_type + "_BDT_TDATA");

  TH1D *h1d_bdt_good_sig = (TH1D *) HList -> FindObject(var_type + "_BDT_good_TISR3PI_SIG_TMP");
  h1d_bdt_good_sig -> Scale(sf_tmp);
  formatfill_h(h1d_bdt_good_sig, 3, 3001);
  
  TH1D *h1d_bdt_bad_sig = (TH1D *) HList -> FindObject(var_type + "_BDT_bad_TISR3PI_SIG_TMP");
  h1d_bdt_bad_sig -> Scale(sf_tmp);
  format_h(h1d_bdt_bad_sig, 3, 2);
  
  TH1D* h1d_bdt_eeg = (TH1D*)HList -> FindObject(var_type + "_BDT_TEEG");
  
  TH1D* h1d_bdt_omegapi = (TH1D*)HList -> FindObject(var_type + "_BDT_TOMEGAPI");
  
  TH1D* h1d_bdt_kpm = (TH1D*)HList -> FindObject(var_type + "_BDT_TKPM");
  
  TH1D* h1d_bdt_ksl = (TH1D*)HList -> FindObject(var_type + "_BDT_TKSL");
  
  TH1D* h1d_bdt_rhopi = (TH1D*)HList -> FindObject(var_type + "_BDT_TRHOPI");
  
  TH1D* h1d_bdt_etagam = (TH1D*)HList -> FindObject(var_type + "_BDT_TETAGAM");
    
  TH1D *h1d_bdt_bkgrest = (TH1D*)HList -> FindObject(var_type + "_BDT_TBKGREST");
  
  //hM_gg_BDT_good_MCSUM -> Add(hM_gg_BDT_good_TEEG, 1.);
  //hM_gg_BDT_good_MCSUM -> Add(hM_gg_BDT_good_TOMEGAPI, 1.);
  //hM_gg_BDT_good_MCSUM -> Add(hM_gg_BDT_good_TKPM, 1.);
  //hM_gg_BDT_good_MCSUM -> Add(hM_gg_BDT_good_TKSL, 1.);
  //hM_gg_BDT_good_MCSUM -> Add(hM_gg_BDT_good_TRHOPI, 1.);
  //hM_gg_BDT_good_MCSUM -> Add(hM_gg_BDT_good_TETAGAM, 1.);
  //hM_gg_BDT_good_MCSUM -> Add(hM_gg_BDT_good_TMCREST, 1.);
  
  TCanvas* cv =  new TCanvas("cv_" + var_type, cv_title, 0, 0, 1200, 800);

  cv -> SetBottomMargin(0.15);//0.007
  cv -> SetLeftMargin(0.15);
  cv -> SetRightMargin(0.15);

  h1d_bdt_data -> Draw("E");
  h1d_bdt_good_sig -> Draw("HistSame");
  h1d_bdt_bad_sig -> Draw("HistSame");
  h1d_bdt_eeg -> Draw("HistSame");
  h1d_bdt_omegapi -> Draw("HistSame");
  h1d_bdt_kpm -> Draw("HistSame");
  h1d_bdt_ksl -> Draw("HistSame");
  h1d_bdt_rhopi -> Draw("HistSame");
  h1d_bdt_etagam -> Draw("HistSame");
  h1d_bdt_bkgrest -> Draw("HistSame");
  
  
  return cv;
}

TCanvas* plot_kine_var(TObjArray* HList, const double sf_tmp, const TString var_type, const TString cv_title){

  /*
  TH1D* hM_gg_TDATA = (TH1D*)file -> Get("hM_gg_TDATA");
    
  TH1D* hM_gg_BDT_TDATA = (TH1D*)file -> Get("hM_gg_BDT_TDATA");
  
  TH1D* hM_gg_BDT_good_TDATA = (TH1D*)file -> Get("hM_gg_BDT_good_TDATA");
  
  TH1D* hM_gg_BDT_bad_TDATA = (TH1D*)file -> Get("hM_gg_BDT_bad_TDATA");
  
  TH1D* hM_gg_BDT_good_T3PIGAM = (TH1D*)file -> Get("hM_gg_BDT_good_T3PIGAM");
  TH1D* hM_gg_BDT_good_TISR3PI_SIG = (TH1D*)file -> Get("hM_gg_BDT_good_TISR3PI_SIG");
  
  TH1D* hM_gg_BDT_good_TEEG = (TH1D*)file -> Get("hM_gg_BDT_good_TEEG");
  
  TH1D* hM_gg_BDT_good_TOMEGAPI = (TH1D*)file -> Get("hM_gg_BDT_good_TOMEGAPI");
  
  TH1D* hM_gg_BDT_good_TKPM = (TH1D*)file -> Get("hM_gg_BDT_good_TKPM");
  
  TH1D* hM_gg_BDT_good_TKSL = (TH1D*)file -> Get("hM_gg_BDT_good_TKSL");
  
  TH1D* hM_gg_BDT_good_TRHOPI = (TH1D*)file -> Get("hM_gg_BDT_good_TRHOPI");
  
  TH1D* hM_gg_BDT_good_TETAGAM = (TH1D*)file -> Get("hM_gg_BDT_good_TETAGAM");
  
  TH1D* hM_gg_BDT_good_TMCREST = (TH1D*)file -> Get("hM_gg_BDT_good_TBKGREST");
    
  TH1D *hist_bkgrest = (TH1D *) Hlist.FindObject("hist_bkgrest");
  
  */
  
  TH1D *h1d_kloe_data = (TH1D *) HList -> FindObject(var_type + "_TDATA");
  TH1D *h1d_bdt_data = (TH1D *) HList -> FindObject(var_type + "_BDT_TDATA");
  TH1D *h1d_bdt_good_data = (TH1D *) HList -> FindObject(var_type + "_BDT_good_TDATA");
  TH1D *h1d_bdt_bad_data = (TH1D *) HList -> FindObject(var_type + "_BDT_bad_TDATA");

  TCanvas* cv =  new TCanvas("cv_" + var_type, cv_title, 0, 0, 1200, 800);

  cv -> SetBottomMargin(0.15);//0.007
  cv -> SetLeftMargin(0.15);
  cv -> SetRightMargin(0.15);

  h1d_kloe_data -> Draw("E");
  h1d_bdt_data -> Draw("ESame");
  h1d_bdt_good_data -> Draw("ESame");
  h1d_bdt_bad_data -> Draw("ESame");


  return cv;
  
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

  char display[50];

  //sprintf(display, "#frac{N_{sig}}{N_{data}-N_{bkg}}");
  
  TPaveText *pt = new TPaveText(pt1_x0, 0.8, pt1_x1, 0.85, "NDC");

  pt -> SetTextSize(0.08);
  pt -> SetFillColor(0);
  pt -> SetTextAlign(12);
  
  //pt -> AddText("Relative Error [%]");
  pt -> AddText(pt_str);
  
  pt -> Draw("Same");
  line1 -> Draw("Same");
  line2 -> Draw("Same");

  h2d -> GetZaxis() -> SetLabelSize(0.045);
  
  gPad -> SetLogz();
  
  return cv2d;
  
}

TCanvas *plot_sfw(const TString hist_type, const TString cv_nm, TH2D *h2d, const TString pt_str)
{

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
  h2d -> GetXaxis() -> SetTitle("M_{2#pi} " + TString::Format("Events/[%0.2f", binwidth_x) + " MeV/c^{2}]"); //SetTitle(x_label + " " + x_unit);
  h2d -> GetXaxis() -> SetTitleOffset(1.2);
  h2d -> GetXaxis() -> SetTitleSize(0.06);
  h2d -> GetXaxis() -> CenterTitle();
  h2d -> GetXaxis() -> SetLabelSize(0.06);
  h2d -> GetXaxis() -> SetLabelOffset(0.01);
  //h2d -> GetXaxis() -> SetRangeUser(0.2, 0.6);
  
  h2d -> GetYaxis() -> SetTitle("E_{#gamma_{3}} " + TString::Format("Events/[%0.2f", binwidth_y) + " MeV]"); //SetTitle(x_label + " " + x_unit);
  h2d -> GetYaxis() -> SetLabelOffset(0.01);
  h2d -> GetYaxis() -> SetTitleOffset(1.2);
  h2d -> GetYaxis() -> SetLabelSize(0.05);
  h2d -> GetYaxis() -> SetTitleSize(0.06);
  h2d -> GetYaxis() -> CenterTitle();

  h2d -> Draw("COLZ");
  //h2d -> Draw("TEXT0COLZ");

  gPad->SetLogz(1);
  
  char display[50];

  //sprintf(display, "#frac{N_{sig}}{N_{data}-N_{bkg}}");
  
  TPaveText *pt = new TPaveText(pt1_x0, 0.7, pt1_x1, 0.85, "NDC");

  pt -> SetTextSize(0.1);
  pt -> SetFillColor(0);
  pt -> SetTextAlign(12);
  
  //pt -> AddText("Relative Error [%]");
  pt -> AddText(pt_str);
  
  pt -> Draw("Same");
  h2d -> GetZaxis() -> SetLabelSize(0.045);
  
  gPad -> SetLogz();
  //gPad -> Modified();
  //gPad -> Update();

  return cv2d;
  
}

TCanvas *cv_plot(const TString hist_type, TH1D* h_kloe, TH1D* h_good_kloe, TH1D* h_bad_kloe, TH1D* h_bdt, TH1D* h_good_bdt, TH1D* h_bad_bdt, TString y_title, TString x_title, const TString sample_type, const TString cv_title) {

  TCanvas *cv = new TCanvas("cv_" + hist_type, cv_title, 1000, 800);
  cv -> SetLeftMargin(0.1);
  cv -> SetBottomMargin(0.1);//0.007
  
  double ymax = h_kloe -> GetBinContent(h_kloe -> GetMaximumBin());
  const double xmin = 0., xmax = 1000.;
  
  //const double ymax = 300.;
  //const double xmin = 650., xmax = 1000.;
  
  h_kloe -> GetYaxis() -> SetTitle(y_title);
  h_kloe -> GetYaxis() -> CenterTitle();
  h_kloe -> GetYaxis() -> SetTitleSize(0.04);
  h_kloe -> GetYaxis() -> SetNdivisions(505);
  h_kloe -> GetYaxis() -> SetRangeUser(0.1, ymax * 1.5); 
  h_kloe -> GetXaxis() -> SetTitle(x_title);
  h_kloe -> GetXaxis() -> CenterTitle();
  h_kloe -> GetXaxis() -> SetTitleSize(0.04);
  h_kloe -> GetXaxis() -> SetRangeUser(xmin, xmax); 
  
  //gPad->SetLogy(1); 
  
  TLegend *legd_cv = new TLegend(0.6, 0.65, 0.85, 0.9);
  
  legd_cv -> SetTextFont(132);
  legd_cv -> SetFillStyle(0);
  legd_cv -> SetBorderSize(0);
  legd_cv -> SetNColumns(1);

  legd_cv -> AddEntry(h_kloe, "#chi^{2}_{m_{#gamma#gamma}} Selection", "lep");
  
  
  if (sample_type == "Data") {
    h_kloe -> Draw("E");
    //h_good_kloe -> Draw("Same");
    //h_bad_kloe -> Draw("Same");
    
    //h_bdt -> Draw("ESame");
    h_good_bdt -> Draw("ESame");
    h_bad_bdt -> Draw("ESame");
    line11 -> Draw("Same");
    line22 -> Draw("Same");

    //legd_cv -> AddEntry(h_bdt, "BDT Selection", "lep");
    legd_cv -> AddEntry(h_good_bdt, "BDT Selected", "lep");
    legd_cv -> AddEntry(h_bad_bdt, "BDT Discarded", "lep");
    
  }
  else if (sample_type == "MC") {
    h_kloe -> Draw();
    h_good_kloe -> Draw("Same");
    h_bad_kloe -> Draw("Same");
    
    h_bdt -> Draw("Same");
    h_good_bdt -> Draw("Same");
    h_bad_bdt -> Draw("Same");

    legd_cv -> AddEntry(h_good_kloe, "#chi^{2}_{m_{#gamma#gamma}} Good", "lep");
    legd_cv -> AddEntry(h_bad_kloe, "#chi^{2}_{m_{#gamma#gamma}} Bad", "f");

    //legd_cv -> AddEntry(h_bdt, "BDT Selection", "lep");
    legd_cv -> AddEntry(h_good_bdt, "BDT Selected", "lep");
    legd_cv -> AddEntry(h_bad_bdt, "BDT Discarded", "f");
    
  }
  
  
  legd_cv -> Draw("Same");
  
  legtextsize(legd_cv, 0.04);
  
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

      //if (classnm_tree == "TH2D") { // list all TH2D
      if (classnm_tree == "TH1D") { // list all TH1D
	//cout << "classnm = " << classnm_tree << ", objnm = " << objnm_tree << endl;
      }
      
      TTree *tree_tmp = (TTree*)file -> Get(objnm_tree);
      
    }
    
    // Get histos
    file -> cd(); // Make sure we're in the output file

}

void mc_normalization(const char* input_filename = "./output_main_bdt.root") {

  gErrorIgnoreLevel = kError;
  TGaxis::SetMaxDigits(4);
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  gStyle->SetFitFormat("6.4g");

  cout << "Plotting BDT results ... " << endl;

  // Load input root file: output_with_bdt.root
  std::cout << "Loading input " << input_filename << std::endl;  // Added std::

  const int xbins = 200;
  const double xmin = 400.;
  const double xmax = 1000.;
    
  // If data file exists, process it with RDataFame
  if(!gSystem -> AccessPathName(input_filename)){

    cout << "\nProcessing data file: " << input_filename << endl;
    
    // Check input root file and list all trees
    TFile* file = new TFile(input_filename);
    check_trees(file);
    
    // TH2D
    TH2D* h2dM3pi_kloeBDT_corr_TDATA = (TH2D *) file -> Get("h2dM3pi_kloeBDT_corr_TDATA");
    TH2D* h2dM3pi_kloeBDT_corr_TISR3PI_SIG = (TH2D *) file -> Get("h2dM3pi_kloeBDT_corr_TISR3PI_SIG");
    TH2D* h2dM3pi_kloeBDT_corr_TETAGAM = (TH2D *) file -> Get("h2dM3pi_kloeBDT_corr_TETAGAM");

    /*
    TH2D *h2d_sfw_BDT_good_TDATA = (TH2D *) file -> Get("h2d_sfw_BDT_good_TDATA");
    TH2D *h2d_sfw_BDT_good_TISR3PI_SIG = (TH2D *) file -> Get("h2d_sfw_BDT_good_TISR3PI_SIG"); // sig mc 1
    TH2D *h2d_sfw_BDT_good_TEEG = (TH2D *) file -> Get("h2d_sfw_BDT_good_TEEG"); // bkg 1, mc 2
    TH2D *h2d_sfw_BDT_good_TOMEGAPI = (TH2D *) file -> Get("h2d_sfw_BDT_good_TOMEGAPI"); // bkg 2, mc 3
    TH2D *h2d_sfw_BDT_good_TKPM = (TH2D *) file -> Get("h2d_sfw_BDT_good_TKPM"); // bkg 3, mc 4
    TH2D *h2d_sfw_BDT_good_TKSL = (TH2D *) file -> Get("h2d_sfw_BDT_good_TKSL"); // bkg 4, mc 5
    //TH2D *h2d_sfw_BDT_good_T3PIGAM = (TH2D *) file -> Get("h2d_sfw_BDT_good_T3PIGAM");
    TH2D *h2d_sfw_BDT_good_TRHOPI = (TH2D *) file -> Get("h2d_sfw_BDT_good_TRHOPI"); // bkg 5, mc 6
    TH2D *h2d_sfw_BDT_good_TETAGAM = (TH2D *) file -> Get("h2d_sfw_BDT_good_TETAGAM"); // bkg 6, mc 7
    TH2D *h2d_sfw_BDT_good_TBKGREST = (TH2D *) file -> Get("h2d_sfw_BDT_good_TBKGREST"); // bkg 7, mc 8
    */

    // KLOE Selection
    TH2D *h2d_sfw_BDT_good_TDATA = (TH2D *) file -> Get("h2d_sfw_good_TDATA");
    TH2D *h2d_sfw_BDT_good_TISR3PI_SIG = (TH2D *) file -> Get("h2d_sfw_good_TISR3PI_SIG"); // sig mc 1
    TH2D *h2d_sfw_BDT_good_TEEG = (TH2D *) file -> Get("h2d_sfw_good_TEEG"); // bkg 1, mc 2
    TH2D *h2d_sfw_BDT_good_TOMEGAPI = (TH2D *) file -> Get("h2d_sfw_good_TOMEGAPI"); // bkg 2, mc 3
    TH2D *h2d_sfw_BDT_good_TKPM = (TH2D *) file -> Get("h2d_sfw_good_TKPM"); // bkg 3, mc 4
    TH2D *h2d_sfw_BDT_good_TKSL = (TH2D *) file -> Get("h2d_sfw_good_TKSL"); // bkg 4, mc 5
    //TH2D *h2d_sfw_BDT_good_T3PIGAM = (TH2D *) file -> Get("h2d_sfw_good_T3PIGAM");
    TH2D *h2d_sfw_BDT_good_TRHOPI = (TH2D *) file -> Get("h2d_sfw_good_TRHOPI"); // bkg 5, mc 6
    TH2D *h2d_sfw_BDT_good_TETAGAM = (TH2D *) file -> Get("h2d_sfw_good_TETAGAM"); // bkg 6, mc 7
    TH2D *h2d_sfw_BDT_good_TBKGREST = (TH2D *) file -> Get("h2d_sfw_good_TBKGREST"); // bkg 7, mc 8
    
    TH2D * h2d_sfw_BDT_good_MCREST;
    TH2D * h2d_sfw_BDT_good_MCSUM;

    // h2d_sfw_BDT_good_MCREST
    h2d_sfw_BDT_good_MCREST = (TH2D*) h2d_sfw_BDT_good_TBKGREST -> Clone();
    h2d_sfw_BDT_good_MCREST -> Add(h2d_sfw_BDT_good_TKPM, 1.);
    h2d_sfw_BDT_good_MCREST -> Add(h2d_sfw_BDT_good_TRHOPI, 1.);
    h2d_sfw_BDT_good_MCREST -> SetName("h2d_sfw_BDT_good_MCREST");
    
    // h2d_sfw_BDT_good_MCSUM
    h2d_sfw_BDT_good_MCSUM = (TH2D*) h2d_sfw_BDT_good_TISR3PI_SIG -> Clone();
    h2d_sfw_BDT_good_MCSUM -> Add(h2d_sfw_BDT_good_TEEG, 1.);
    h2d_sfw_BDT_good_MCSUM -> Add(h2d_sfw_BDT_good_TOMEGAPI, 1.);
    h2d_sfw_BDT_good_MCSUM -> Add(h2d_sfw_BDT_good_TKSL, 1.);
    h2d_sfw_BDT_good_MCSUM -> Add(h2d_sfw_BDT_good_TETAGAM, 1.);
    h2d_sfw_BDT_good_MCSUM -> Add(h2d_sfw_BDT_good_MCREST, 1.);
    h2d_sfw_BDT_good_MCSUM -> SetName("h2d_sfw_BDT_good_MCSUM");

    // mc sum no etagam
    TH2D *h2d_sfw_BDT_good_MCSUM_NOETA = (TH2D *) h2d_sfw_BDT_good_TEEG -> Clone();
    h2d_sfw_BDT_good_MCSUM_NOETA -> Add(h2d_sfw_BDT_good_TKSL, 1.);
    h2d_sfw_BDT_good_MCSUM_NOETA -> Add(h2d_sfw_BDT_good_TOMEGAPI, 1.);
    h2d_sfw_BDT_good_MCSUM_NOETA -> Add(h2d_sfw_BDT_good_MCREST, 1.);
    h2d_sfw_BDT_good_MCSUM_NOETA -> SetName("h2d_sfw_BDT_good_MCSUM_NOETA");

    // TH1D
    // M3pi
    TObjArray* HM3pi = new TObjArray(100); // Hist. Array
    
    TH1D* hM3pi_TISR3PI_SIG = (TH1D*)file -> Get("hM3pi_TISR3PI_SIG");
    format_h(hM3pi_TISR3PI_SIG, 1, 2);
    
    TH1D* hM3pi_good_TISR3PI_SIG = (TH1D*)file -> Get("hM3pi_good_TISR3PI_SIG");
    formatfill_h(hM3pi_good_TISR3PI_SIG, 4, 3001);

    TH1D* hM3pi_bad_TISR3PI_SIG = (TH1D*)file -> Get("hM3pi_bad_TISR3PI_SIG");
    format_h(hM3pi_bad_TISR3PI_SIG, 4, 2);
    
    TH1D* hM3pi_BDT_TISR3PI_SIG = (TH1D*)file -> Get("hM3pi_BDT_TISR3PI_SIG");
    format_h(hM3pi_BDT_TISR3PI_SIG, 2, 2);
    
    TH1D* hM3pi_BDT_good_TISR3PI_SIG = (TH1D*)file -> Get("hM3pi_BDT_good_TISR3PI_SIG");
    HM3pi -> Add(hM3pi_BDT_good_TISR3PI_SIG);
    //format_h(hM3pi_BDT_good_TISR3PI_SIG, 4, 2);
    
    TH1D* hM3pi_BDT_bad_TISR3PI_SIG = (TH1D*)file -> Get("hM3pi_BDT_bad_TISR3PI_SIG");
    //formatfill_h(hM3pi_BDT_bad_TISR3PI_SIG, 4, 3001);

    TH1D* hM3pi_BDT_good_TISR3PI_SIG_TMP = (TH1D*)hM3pi_BDT_good_TISR3PI_SIG -> Clone("hM3pi_BDT_good_TISR3PI_SIG_TMP"); 
    HM3pi -> Add(hM3pi_BDT_good_TISR3PI_SIG_TMP);
    
    TH1D* hM3pi_BDT_bad_TISR3PI_SIG_TMP = (TH1D*)hM3pi_BDT_bad_TISR3PI_SIG -> Clone("hM3pi_BDT_bad_TISR3PI_SIG_TMP");
    HM3pi -> Add(hM3pi_BDT_bad_TISR3PI_SIG_TMP);

    //
    TH1D* hM3pi_BDT_good_T3PIGAM = (TH1D*)file -> Get("hM3pi_BDT_good_T3PIGAM");
    HM3pi -> Add(hM3pi_BDT_good_T3PIGAM);

    // Create a list to add all histos
    TObjArray* HistArray_m3pi = new TObjArray(100); // Hist. Array
    
    const TString kine_type = "hM3pi";
    const char* names[] = {"TDATA",
			   "good_TDATA",
			   "bad_TDATA",
			   "BDT_TDATA",
			   "BDT_good_TDATA",
			   "BDT_bad_TDATA",
			   "BDT_TEEG",
			   "BDT_TOMEGAPI",
			   "BDT_TKPM",
			   "BDT_TKSL",
			   "BDT_TRHOPI",
			   "BDT_TETAGAM",
			   "BDT_TBKGREST",
			   "BDT_T3PIGAM",
			   "BDT_good_T3PIGAM",
			   "BDT_bad_T3PIGAM",
			   "BDT_TISR3PI_SIG",
			   "BDT_good_TISR3PI_SIG",
			   "BDT_bad_TISR3PI_SIG"
    };


    for (int i = 0; i < 19; i ++) {
      TH1D* hm3pi = (TH1D*)file -> Get(kine_type + "_" + names[i]);
      cout << hm3pi -> GetName() << endl;
      HistArray_m3pi -> Add(hm3pi);
    }

    //  
    TH1D* hM3pi_TDATA = (TH1D*)file -> Get("hM3pi_TDATA");
    format_h(hM3pi_TDATA, 1, 2);
    HM3pi -> Add(hM3pi_TDATA);
    
    TH1D* hM3pi_good_TDATA = (TH1D*)file -> Get("hM3pi_good_TDATA");
    TH1D* hM3pi_bad_TDATA = (TH1D*)file -> Get("hM3pi_bad_TDATA");
    formatfill_h(hM3pi_bad_TDATA, 4, 3001);
    format_h(hM3pi_good_TDATA, 4, 2);

    TH1D* hM3pi_BDT_TDATA = (TH1D*)file -> Get("hM3pi_BDT_TDATA");
    format_h(hM3pi_BDT_TDATA, 4, 2);
    HM3pi -> Add(hM3pi_BDT_TDATA);
      
    TH1D* hM3pi_BDT_good_TDATA = (TH1D*)file -> Get("hM3pi_BDT_good_TDATA");
    format_h(hM3pi_BDT_good_TDATA, 3, 2);
    HM3pi -> Add(hM3pi_BDT_good_TDATA);

    TH1D* hM3pi_BDT_bad_TDATA = (TH1D*)file -> Get("hM3pi_BDT_bad_TDATA");
    format_h(hM3pi_BDT_bad_TDATA, 2, 2);
    HM3pi -> Add(hM3pi_BDT_bad_TDATA);

    //
    TH1D* hM3pi_BDT_TEEG = (TH1D*)file -> Get("hM3pi_BDT_TEEG");
    format_h(hM3pi_BDT_TEEG, 5, 2);
    HM3pi -> Add(hM3pi_BDT_TEEG);

    TH1D* hM3pi_BDT_TOMEGAPI = (TH1D*)file -> Get("hM3pi_BDT_TOMEGAPI");
    format_h(hM3pi_BDT_TOMEGAPI, 6, 2);
    HM3pi -> Add(hM3pi_BDT_TOMEGAPI);

    TH1D* hM3pi_BDT_TKPM = (TH1D*)file -> Get("hM3pi_BDT_TKPM");
    format_h(hM3pi_BDT_TKPM, 7, 2);
    HM3pi -> Add(hM3pi_BDT_TKPM);

    TH1D* hM3pi_BDT_TKSL = (TH1D*)file -> Get("hM3pi_BDT_TKSL");
    format_h(hM3pi_BDT_TKSL, 8, 2);
    HM3pi -> Add(hM3pi_BDT_TKSL);

    TH1D* hM3pi_BDT_TRHOPI = (TH1D*)file -> Get("hM3pi_BDT_TRHOPI");
    format_h(hM3pi_BDT_TRHOPI, 9, 2);
    HM3pi -> Add(hM3pi_BDT_TRHOPI);

    TH1D* hM3pi_BDT_TETAGAM = (TH1D*)file -> Get("hM3pi_BDT_TETAGAM");
    format_h(hM3pi_BDT_TETAGAM, 11, 2);
    HM3pi -> Add(hM3pi_BDT_TETAGAM);

    TH1D* hM3pi_BDT_TBKGREST = (TH1D*)file -> Get("hM3pi_BDT_TBKGREST");
    format_h(hM3pi_BDT_TBKGREST, 42, 2);
    HM3pi -> Add(hM3pi_BDT_TBKGREST);
   //cout << hM3pi_BDT_TBKGREST -> GetEntries() << endl;

    // Check the current axis ranges
    cout << "X-axis range: " << hM3pi_BDT_TBKGREST->GetXaxis()->GetXmin() 
	 << " to " << hM3pi_BDT_TBKGREST->GetXaxis()->GetXmax() << endl;
    
    // Check where the entries actually are
    cout << "Mean: " << hM3pi_BDT_TBKGREST->GetMean() << endl;
    cout << "RMS: " << hM3pi_BDT_TBKGREST->GetRMS() << endl;
    cout << "Min bin content: " << hM3pi_BDT_TBKGREST->GetMinimum() << endl;
    cout << "Max bin content: " << hM3pi_BDT_TBKGREST->GetMaximum() << endl;

    //TH1D* hM_gg_BDT_good_TMCREST = (TH1D*)file -> Get("hM_gg_BDT_good_TBKGREST");
    //format_h(hM_gg_BDT_good_TMCREST, 11, 2);
    
    // M_gg
    TObjArray* HM_gg = new TObjArray(100); // Hist. Array
    
    TH1D* hM_gg_TDATA = (TH1D*)file -> Get("hM_gg_TDATA");
    format_h(hM_gg_TDATA, 1, 2);
    HM_gg -> Add(hM_gg_TDATA);
    
    TH1D* hM_gg_BDT_TDATA = (TH1D*)file -> Get("hM_gg_BDT_TDATA");
    format_h(hM_gg_BDT_TDATA, 4, 2);
    HM_gg -> Add(hM_gg_BDT_TDATA);
    
    TH1D* hM_gg_BDT_good_TDATA = (TH1D*)file -> Get("hM_gg_BDT_good_TDATA");
    format_h(hM_gg_BDT_good_TDATA, 3, 2);
    HM_gg -> Add(hM_gg_BDT_good_TDATA);
    
    TH1D* hM_gg_BDT_bad_TDATA = (TH1D*)file -> Get("hM_gg_BDT_bad_TDATA");
    format_h(hM_gg_BDT_bad_TDATA, 2, 2);
    HM_gg -> Add(hM_gg_BDT_bad_TDATA);
    
    TH1D* hM_gg_BDT_good_T3PIGAM = (TH1D*)file -> Get("hM_gg_BDT_good_T3PIGAM");
    TH1D* hM_gg_BDT_good_TISR3PI_SIG = (TH1D*)file -> Get("hM_gg_BDT_good_TISR3PI_SIG");
    formatfill_h(hM_gg_BDT_good_T3PIGAM, 4, 3001);

    TH1D* hM_gg_BDT_bad_TISR3PI_SIG = (TH1D*)file -> Get("hM_gg_BDT_bad_TISR3PI_SIG");
    
    TH1D* hM_gg_BDT_good_TEEG = (TH1D*)file -> Get("hM_gg_BDT_good_TEEG");
    format_h(hM_gg_BDT_good_TEEG, 5, 2);

    TH1D* hM_gg_BDT_good_TOMEGAPI = (TH1D*)file -> Get("hM_gg_BDT_good_TOMEGAPI");
    format_h(hM_gg_BDT_good_TOMEGAPI, 6, 2);

    TH1D* hM_gg_BDT_good_TKPM = (TH1D*)file -> Get("hM_gg_BDT_good_TKPM");
    format_h(hM_gg_BDT_good_TKPM, 7, 2);

    TH1D* hM_gg_BDT_good_TKSL = (TH1D*)file -> Get("hM_gg_BDT_good_TKSL");
    format_h(hM_gg_BDT_good_TKSL, 8, 2);

    TH1D* hM_gg_BDT_good_TRHOPI = (TH1D*)file -> Get("hM_gg_BDT_good_TRHOPI");
    format_h(hM_gg_BDT_good_TRHOPI, 9, 2);

    TH1D* hM_gg_BDT_good_TETAGAM = (TH1D*)file -> Get("hM_gg_BDT_good_TETAGAM");
    format_h(hM_gg_BDT_good_TETAGAM, 10, 2);

    TH1D* hM_gg_BDT_good_TMCREST = (TH1D*)file -> Get("hM_gg_BDT_good_TBKGREST");
    format_h(hM_gg_BDT_good_TMCREST, 11, 2);
    
    // Get sfw for ISR3PI_SIG
    const double sf_tmp = hM_gg_BDT_good_T3PIGAM -> GetEntries() / hM_gg_BDT_good_TISR3PI_SIG -> GetEntries();
    cout << "sf_tmp = " << sf_tmp << endl;
    
    TH1D* hM_gg_BDT_good_TISR3PI_SIG_TMP = (TH1D*)hM_gg_BDT_good_TISR3PI_SIG -> Clone("hM_gg_BDT_good_TISR3PI_SIG_TMP"); // MC 1
    hM_gg_BDT_good_TISR3PI_SIG_TMP -> Scale(sf_tmp);
    format_h(hM_gg_BDT_good_TISR3PI_SIG_TMP, 4, 2);

    TH1D* hM_gg_BDT_bad_TISR3PI_SIG_TMP = (TH1D*)hM_gg_BDT_bad_TISR3PI_SIG -> Clone("hM_gg_BDT_bad_TISR3PI_SIG_TMP");
    
    TH1D* hM_gg_BDT_good_MCSUM = (TH1D*) hM_gg_BDT_good_TISR3PI_SIG_TMP -> Clone("hM_gg_BDT_good_MCSUM");
    hM_gg_BDT_good_MCSUM -> Add(hM_gg_BDT_good_TEEG, 1.);
    hM_gg_BDT_good_MCSUM -> Add(hM_gg_BDT_good_TOMEGAPI, 1.);
    hM_gg_BDT_good_MCSUM -> Add(hM_gg_BDT_good_TKPM, 1.);
    hM_gg_BDT_good_MCSUM -> Add(hM_gg_BDT_good_TKSL, 1.);
    hM_gg_BDT_good_MCSUM -> Add(hM_gg_BDT_good_TRHOPI, 1.);
    hM_gg_BDT_good_MCSUM -> Add(hM_gg_BDT_good_TETAGAM, 1.);
    hM_gg_BDT_good_MCSUM -> Add(hM_gg_BDT_good_TMCREST, 1.);
    
    format_h(hM_gg_BDT_good_MCSUM, 2, 2);

    //TCanvas* cv_tmp =  new TCanvas("cv_tmp", "", 0, 0, 700, 700);

    //hM_gg_TDATA -> Draw("E");
    //hM_gg_BDT_TDATA -> Draw("ESame");
    //hM_gg_BDT_good_TDATA -> Draw("ESame");
    //hM_gg_BDT_bad_TDATA -> Draw("ESame");
    
    //hM_gg_BDT_good_TISR3PI_SIG_TMP -> Draw("HistSame");
    //hM_gg_BDT_good_T3PIGAM -> Draw("HistSame");
    //hM_gg_BDT_good_TEEG -> Draw("HistSame");
    //hM_gg_BDT_good_TOMEGAPI -> Draw("HistSame");
    //hM_gg_BDT_good_TKPM -> Draw("HistSame");
    //hM_gg_BDT_good_TKSL -> Draw("HistSame");
    //hM_gg_BDT_good_TRHOPI -> Draw("HistSame");
    //hM_gg_BDT_good_TETAGAM -> Draw("HistSame");
    
    
    //hM_gg_BDT_good_MCSUM -> Draw("HistSame");

    //==================================== Plotting M_gg (Data)=================================
    TCanvas* cv_M_gg = plot_kine_var(HM_gg, sf_tmp, "hM_gg", "Invariant mass of gamma-gamma");
    //cv_M_gg -> SetTitle("Invariant mass of gamma-gamma");
    
    // Create TPaveText, add text lines, and draw
    //TPaveText* pt_cut = set_pt(0.2, 0.8, 0.3, 0.8);
    TPaveText* pt_cut = set_pt(0.1, 0.92, 0.9, 0.98);
    pt_cut -> SetTextColor(42);
    pt_cut -> AddText(Form("M^{BDT}_{3#pi}#in[%0.0f, %0.0f] MeV/c^{2}", 650., 900.));
    pt_cut -> Draw("same");

    TPaveText* pt_mgg1 = set_pt(0.2, 0.8, 0.3, 0.85);
    pt_mgg1 -> AddText("Data");
    pt_mgg1 -> Draw("same");
    

    // CV attribute
    const double binwidth_mgg = getbinwidth(hM_gg_TDATA);
    set_cv(hM_gg_TDATA, "M_{#gamma#gamma}", "[MeV/c^{2}]", 50., 300., binwidth_mgg);

    // Create Legend
    TLegend *legd_cv = set_legend(0.6, 0.7, 0.9, 0.9);
    legd_cv -> AddEntry(hM_gg_TDATA, "#chi^{2}_{m_{#gamma#gamma}} Selection", "lep");
    legd_cv -> AddEntry(hM_gg_BDT_TDATA, "BDT Selection", "lep");
    legd_cv -> AddEntry(hM_gg_BDT_good_TDATA, "BDT best #pi^{0}(#gamma#gamma)", "lep");
    legd_cv -> AddEntry(hM_gg_BDT_bad_TDATA, "BDT Discarded", "lep");
    legd_cv -> Draw("Same");
    legtextsize(legd_cv, 0.04);
    
    // Update the canvas to show changes
    cv_M_gg -> Update();
    cv_M_gg -> Modified();
    cv_M_gg -> SaveAs("cv_Mgg.pdf");
    cv_M_gg -> Close();
    //cv_M_gg -> SetLogy(1);  // Turn off log scale
    
    //==================================== Plotting M3pi=================================
    TCanvas* cv_M3pi = plot_kine_var(HM3pi, sf_tmp, "hM3pi", "Invariant mass of M3pi");

    const double binwidth_m3pi = getbinwidth(hM3pi_TDATA);
    set_cv(hM3pi_TDATA, "M_{3#pi}", "[MeV/c^{2}]", 0., 1000., binwidth_m3pi);

    pt_cut -> Draw("same");
    pt_mgg1 -> Draw("same");
    
    legd_cv -> Draw("Same");
    line11 -> Draw("Same");
    line22 -> Draw("Same");

    cv_M3pi -> SetLogy(1);  // Turn off log scale
    
    cv_M3pi -> Update();
    cv_M3pi -> Modified();
    cv_M3pi -> SaveAs("cv_M3pi.pdf");
    //cv_M3pi -> Close();
    
    
    //==================================== Plotting M3pi MC-Data comparison=================================
    /*
    hM3pi_BDT_TDATA -> Write();
    hM3pi_BDT_good_TISR3PI_SIG_TMP -> Write();
    hM3pi_BDT_bad_TISR3PI_SIG_TMP -> Write();
    hM3pi_BDT_TEEG -> Write();
    hM3pi_BDT_TOMEGAPI -> Write();
    hM3pi_BDT_TKPM -> Write();
    hM3pi_BDT_TKSL -> Write();
    hM3pi_BDT_TRHOPI -> Write();
    hM3pi_BDT_TETAGAM -> Write();
    hM3pi_BDT_TBKGREST -> Write();
    */
    
    // Plot M3pi MC-Data comparsion
    TCanvas* cv_M3pi_compr = plot_kine_compr(HM3pi, sf_tmp, "hM3pi", "M3pi MC-Data comparsion");
    
    set_cv(hM3pi_BDT_TDATA, "M_{3#pi}", "[MeV/c^{2}]", 650., 900., binwidth_m3pi);

    TH1D* h1d_bdt_mcsum = (TH1D*) hM3pi_BDT_bad_TISR3PI_SIG_TMP -> Clone("h1d_bdt_mcsum");
    h1d_bdt_mcsum -> Add(hM3pi_BDT_good_TISR3PI_SIG_TMP, 1.);
    h1d_bdt_mcsum -> Add(hM3pi_BDT_TEEG, 1.);
    h1d_bdt_mcsum -> Add(hM3pi_BDT_TOMEGAPI, 1.);
    h1d_bdt_mcsum -> Add(hM3pi_BDT_TKPM, 1.);
    h1d_bdt_mcsum -> Add(hM3pi_BDT_TKSL, 1.);
    h1d_bdt_mcsum -> Add(hM3pi_BDT_TRHOPI, 1.);
    h1d_bdt_mcsum -> Add(hM3pi_BDT_TETAGAM, 1.);
    h1d_bdt_mcsum -> Add(hM3pi_BDT_TBKGREST, 1.);

    format_h(h1d_bdt_mcsum, 2, 2);
    
    
    h1d_bdt_mcsum -> Draw("HistSame");
  
    TPaveText* pt_m3pi_compr = set_pt(0.6, 0.8, 0.7, 0.8);
    pt_m3pi_compr -> AddText("BDT Selection");
    pt_m3pi_compr -> Draw("same");

    // Create Legend
    TLegend *legd_cv1 = set_legend(0.2, 0.3, 0.6, 0.9);
    legd_cv1 -> AddEntry(hM3pi_BDT_TDATA, "Data", "lep");
    legd_cv1 -> AddEntry(hM3pi_BDT_good_TISR3PI_SIG_TMP, "BDT best #pi^{0}(#gamma#gamma) (Signal)", "f");
    legd_cv1 -> AddEntry(hM3pi_BDT_bad_TISR3PI_SIG_TMP, "BDT discarded (Signal)", "l");
    legd_cv1 -> AddEntry(hM3pi_BDT_TEEG, "Bhabha", "l");
    legd_cv1 -> AddEntry(hM3pi_BDT_TOMEGAPI, "#omega#pi", "l");
    legd_cv1 -> AddEntry(hM3pi_BDT_TKPM, "K#bar{K}", "l");
    legd_cv1 -> AddEntry(hM3pi_BDT_TKSL, "K_{S}K_{L}", "l");
    legd_cv1 -> AddEntry(hM3pi_BDT_TRHOPI, "#rho#pi", "l");
    legd_cv1 -> AddEntry(hM3pi_BDT_TETAGAM, "#eta#gamma", "l");
    legd_cv1 -> AddEntry(hM3pi_BDT_TBKGREST, "Others", "l");
    legd_cv1 -> AddEntry(h1d_bdt_mcsum, "MC sum", "l");
    
    legd_cv1 -> Draw("Same");
    
    legtextsize(legd_cv1, 0.04);

    //cv_M3pi_compr -> SaveAs("cv_M3pi_compr.pdf");
    cv_M3pi_compr -> Close();
    
    //==================================== MC normalization =================================

    // define and initialize variable variables
    double nb_data = 0.;
    double nb_isr3pi = 0.;
    double nb_eeg = 0.;
    double nb_omegapi = 0.;
    double nb_kpm = 0.;
    double nb_ksl = 0.;
    double nb_rhopi = 0.;
    double nb_etagam = 0.;
    double nb_mcrest = 0.;
    double nb_mc = 0.;

    // fractions
    double fisr3pi = 0.,   fisr3pi_err = 0.;
    double feeg = 0.,      feeg_err = 0.;
    double fomegapi = 0.,  fomegapi_err = 0.;
    double fkpm = 0.,      fkpm_err = 0.;
    double fksl = 0.,      fksl_err = 0.;
    double frhopi = 0.,    frhopi_err = 0.;
    double fetagam = 0.,   fetagam_err = 0.;
    double fmcrest = 0.,   fmcrest_err = 0.;

    // scaling factors
    double isr3pi_sfw = 0,  isr3pi_sfw_err = 0; 
    double eeg_sfw = 0,     eeg_sfw_err = 0; 
    double omegapi_sfw = 0, omegapi_sfw_err = 0; 
    double kpm_sfw = 0,     kpm_sfw_err = 0; 
    double ksl_sfw = 0,     ksl_sfw_err = 0; 
    double rhopi_sfw = 0,   rhopi_sfw_err = 0; 
    double etagam_sfw = 0,  etagam_sfw_err = 0; 
    double mcrest_sfw  = 0, mcrest_sfw_err = 0; 

    // define tree
    TFile *f_output = new TFile("sfw2d.root", "recreate");
    f_output -> cd();  // Set as current directory
    
    TSFW2D -> SetAutoSave(0);
    
    TSFW2D -> Branch("Br_nb_data", &nb_data, "Br_nb_data/D");
    TSFW2D -> Branch("Br_nb_isr3pi", &nb_isr3pi, "Br_nb_isr3pi/D");
    TSFW2D -> Branch("Br_nb_eeg", &nb_eeg, "Br_nb_eeg/D");
    TSFW2D -> Branch("Br_nb_omegapi", &nb_omegapi, "Br_nb_omegapi/D");
    TSFW2D -> Branch("Br_nb_kpm", &nb_kpm, "Br_nb_kpm/D");
    TSFW2D -> Branch("Br_nb_ksl", &nb_ksl, "Br_nb_ksl/D");
    TSFW2D -> Branch("Br_nb_rhopi", &nb_rhopi, "Br_nb_rhopi/D");
    TSFW2D -> Branch("Br_nb_etagam", &nb_etagam, "Br_nb_etagam/D");
    TSFW2D -> Branch("Br_nb_mcrest", &nb_mcrest, "Br_nb_mcrest/D");

    /*
      TH2D *h2d_sfw_BDT_good_TDATA       // data, mc 0
      TH2D *h2d_sfw_BDT_good_TISR3PI_SIG // sig, mc 1
      TH2D *h2d_sfw_BDT_good_TEEG        // bkg1, mc 2
      TH2D *h2d_sfw_BDT_good_TOMEGAPI    // bkg2, mc 3
      TH2D *h2d_sfw_BDT_good_TKPM        // bkg3, mc 4
      TH2D *h2d_sfw_BDT_good_TKSL        // bkg4, mc 5
      TH2D *h2d_sfw_BDT_good_TRHOPI      // bkg5, mc 6
      TH2D *h2d_sfw_BDT_good_TETAGAM     // bkg6, mc 7
      TH2D *h2d_sfw_BDT_good_TBKGREST    // bkg7, mc 8
      h2d_sfw_BDT_good_MCSUM
    */

    //h2d_sfw_BDT_good_TDATA -> Draw();
    //h2d_sfw_BDT_good_TISR3PI_SIG -> Draw();
    //h2d_sfw_BDT_good_TEEG -> Draw();
    //h2d_sfw_BDT_good_TOMEGAPI -> Draw();
    //h2d_sfw_BDT_good_TKPM -> Draw();
    //h2d_sfw_BDT_good_TKSL -> Draw();
    //h2d_sfw_BDT_good_TRHOPI -> Draw();
    //h2d_sfw_BDT_good_TETAGAM -> Draw();
    //h2d_sfw_BDT_good_TBKGREST -> Draw();
    //h2d_sfw_BDT_good_MCSUM -> Draw();
    
    // loop over histos
    int evnt_indx = 0;
    
    for (int i = 1; i <= h2d_sfw_BDT_good_TDATA -> ProjectionX() -> GetNbinsX(); i ++ ) {

      for (int j = 1; j <= h2d_sfw_BDT_good_TDATA -> ProjectionY() -> GetNbinsX(); j ++ ) {

	evnt_indx += 1;
	
	// data
	nb_data = h2d_sfw_BDT_good_TDATA -> GetBinContent(i, j);
	nb_data_sum += nb_data;

	//cout << i << ", " << j << ", " << nb_data << endl;

	// isr3pi
	nb_isr3pi = h2d_sfw_BDT_good_TISR3PI_SIG -> GetBinContent(i, j);
	nb_isr3pi_sum += nb_isr3pi;

	// eeg
	nb_eeg = h2d_sfw_BDT_good_TEEG -> GetBinContent(i, j);
	nb_eeg_sum += nb_eeg;

	// omegapi
	nb_omegapi = h2d_sfw_BDT_good_TOMEGAPI -> GetBinContent(i, j);
	nb_omegapi_sum += nb_omegapi;

	// kpm
	nb_kpm = h2d_sfw_BDT_good_TKPM -> GetBinContent(i, j);
	nb_kpm_sum += nb_kpm;

	// ksl
	nb_ksl = h2d_sfw_BDT_good_TKSL -> GetBinContent(i, j);
	nb_ksl_sum += nb_ksl;
	  
	// rhopi
	nb_rhopi = h2d_sfw_BDT_good_TRHOPI -> GetBinContent(i, j);
	nb_rhopi_sum += nb_rhopi;

	// etagam
	nb_etagam = h2d_sfw_BDT_good_TETAGAM -> GetBinContent(i, j);
	nb_etagam_sum += nb_etagam;

	// mcrest
	nb_mcrest = h2d_sfw_BDT_good_TBKGREST-> GetBinContent(i, j);
	nb_mcrest_sum += nb_mcrest;

	// mcsum
	nb_mc = h2d_sfw_BDT_good_MCSUM -> GetBinContent(i, j);
	nb_mcsum += nb_mc;

	//
	//if (evnt_indx > 1e3) break;

	TSFW2D -> Fill();
      }
      
    }

    cout << "nb_data_sum = " << nb_data_sum << "\n"
	 << "nb_isr3pi_sum = " << nb_isr3pi_sum << "\n"
    	 << "nb_eeg_sum = " << nb_eeg_sum << "\n"
	 << "nb_omegapi_sum = " << nb_omegapi_sum << "\n"
	 << "nb_kpm_sum = " << nb_kpm_sum << "\n"
	 << "nb_ksl_sum = " << nb_ksl_sum << "\n"
	 << "nb_rhopi_sum = " << nb_rhopi_sum << "\n"
	 << "nb_etagam_sum = " << nb_etagam_sum << "\n"
	 << "nb_mcrest_sum = " << nb_mcrest_sum << "\n"
	 << "nb_mcsum = " << nb_mcsum << ", checked = " << nb_isr3pi_sum + nb_eeg_sum + nb_omegapi_sum + nb_kpm_sum + nb_ksl_sum + nb_rhopi_sum + nb_etagam_sum  + nb_mcrest_sum << "\n\n";

    nb_isr3pi_sum = h2d_sfw_BDT_good_TISR3PI_SIG -> GetEntries();
    nb_eeg_sum = h2d_sfw_BDT_good_TEEG -> GetEntries();
    nb_omegapi_sum = h2d_sfw_BDT_good_TOMEGAPI -> GetEntries();
    nb_kpm_sum = h2d_sfw_BDT_good_TKPM -> GetEntries();
    nb_ksl_sum = h2d_sfw_BDT_good_TKSL -> GetEntries();
    nb_rhopi_sum = h2d_sfw_BDT_good_TRHOPI -> GetEntries() + 1; // avoid zero
    nb_etagam_sum = h2d_sfw_BDT_good_TETAGAM -> GetEntries();
    nb_mcrest_sum = h2d_sfw_BDT_good_MCREST -> GetEntries();
      
    cout << "nb_mcsum = " << h2d_sfw_BDT_good_MCSUM -> GetEntries() << endl;
    cout << "1. nb_isr3pi_sum = " << h2d_sfw_BDT_good_TISR3PI_SIG -> GetEntries() << endl;
    cout << "2. nb_eeg_sum = " << h2d_sfw_BDT_good_TEEG -> GetEntries() << endl;
    cout << "3. nb_omegapi_sum = " << h2d_sfw_BDT_good_TOMEGAPI -> GetEntries() << endl;
    cout << "4. nb_kpm_sum = " << h2d_sfw_BDT_good_TKPM -> GetEntries() << endl;
    cout << "5. nb_ksl_sum = " << h2d_sfw_BDT_good_TKSL -> GetEntries() << endl;
    cout << "6. nb_rhopi_sum = " << h2d_sfw_BDT_good_TRHOPI -> GetEntries() << endl;
    cout << "7. nb_etagam_sum = " << h2d_sfw_BDT_good_TETAGAM -> GetEntries() << endl;
    cout << "8. nb_mcrest_sum = " << h2d_sfw_BDT_good_MCREST -> GetEntries() << "\n\n";
    
    double fisr3pi_init  = h2d_sfw_BDT_good_TISR3PI_SIG -> GetEntries() / h2d_sfw_BDT_good_MCSUM -> GetEntries();
    double feeg_init     = h2d_sfw_BDT_good_TEEG -> GetEntries() / h2d_sfw_BDT_good_MCSUM -> GetEntries();
    double fomegapi_init = h2d_sfw_BDT_good_TOMEGAPI -> GetEntries() / h2d_sfw_BDT_good_MCSUM -> GetEntries();
    double fkpm_init     = h2d_sfw_BDT_good_TKPM -> GetEntries() / h2d_sfw_BDT_good_MCSUM -> GetEntries();
    double fksl_init     = h2d_sfw_BDT_good_TKSL -> GetEntries() / h2d_sfw_BDT_good_MCSUM -> GetEntries();
    double frhopi_init   = h2d_sfw_BDT_good_TRHOPI -> GetEntries() / h2d_sfw_BDT_good_MCSUM -> GetEntries();
    double fetagam_init  = h2d_sfw_BDT_good_TETAGAM -> GetEntries()  / h2d_sfw_BDT_good_MCSUM -> GetEntries();
    double fmcrest_init  = h2d_sfw_BDT_good_MCREST -> GetEntries()  / h2d_sfw_BDT_good_MCSUM -> GetEntries();
    
    cout << "1. fisr3pi_init = " << fisr3pi_init << "\n"
	 << "2. feeg_init = " << feeg_init << "\n"
	 << "3. fomegapi_init = " << fomegapi_init << "\n"
	 << "4. fkpm_init = " << fkpm_init << "\n"
	 << "5. fksl_init = " << fksl_init << "\n"
	 << "6. frhopi_init = " << frhopi_init << "\n"
	 << "7. fetagam_init = " << fetagam_init << "\n"
	 << "8. fmcrest_init = " << fmcrest_init << "\n"
	 << "f_sum = " << fisr3pi_init + feeg_init + fomegapi_init + fkpm_init + fksl_init + frhopi_init + fetagam_init + fmcrest_init << endl;
    
    //<< "nb_mcsum = " << nb_mcsum << ", checked = " << nb_isr3pi_sum + nb_eeg_sum + nb_omegapi_sum + nb_kpm_sum + nb_ksl_sum + nb_rhopi_sum + nb_etagam_sum + nb_mcrest_sum << "\n";

    // Fit

    const int para_nb_sfw2d = 7;
    
    TMinuit *gMinuit = new TMinuit(para_nb_sfw2d); // maximum number of parameters in ()
    gMinuit -> SetFCN(fcn_sfw2d);

    Double_t arglist[10];
    Int_t ierflg = 0;

    // Set print level
    //gMinuit -> SetPrintLevel(-1);

    gMinuit -> mnparm(0, "fisr3pi_ML",  fisr3pi_init,  0.001, 0., 1., ierflg);
    gMinuit -> mnparm(1, "feeg_ML",     feeg_init,     0.001, 0., 1., ierflg);
    gMinuit -> mnparm(2, "fomegapi_ML", fomegapi_init, 0.001, 0., 1., ierflg);
    gMinuit -> mnparm(3, "fkpm_ML",     fkpm_init,     0.001, 0., 1., ierflg);
    gMinuit -> mnparm(4, "fksl_ML",     fksl_init,     0.001, 0., 1., ierflg);
    //gMinuit -> mnparm(5, "frhopi_ML",   frhopi_init,   0.01, 0., 1., ierflg);
    gMinuit -> mnparm(5, "fetagam_ML",  fetagam_init,  0.001, 0., 1., ierflg);
    gMinuit -> mnparm(6, "fmcrest_ML",  fmcrest_init,  0.001, 0., 1., ierflg);
    
    gMinuit -> SetErrorDef(0.5);

    // ready for minimization step
    arglist[0] = 500;
    //gMinuit -> mnexcm("MIGRAD", arglist, 1, ierflg); // fit sfw2d

    // get sfw2d fit results

    gMinuit -> GetParameter(0, fisr3pi, fisr3pi_err);
    gMinuit -> GetParameter(1, feeg, feeg_err);
    gMinuit -> GetParameter(2, fomegapi, fomegapi_err);
    gMinuit -> GetParameter(3, fkpm, fkpm_err);
    gMinuit -> GetParameter(4, fksl, fksl_err);
    gMinuit -> GetParameter(5, fetagam, fetagam_err);
    gMinuit -> GetParameter(6, fmcrest, fmcrest_err);

    isr3pi_sfw = getscale(nb_data_sum, fisr3pi, nb_isr3pi_sum);
    isr3pi_sfw_err = GetScalError(nb_data_sum, nb_isr3pi_sum, fisr3pi, fisr3pi_err);

    eeg_sfw = getscale(nb_data_sum, feeg, nb_eeg_sum);
    eeg_sfw_err = GetScalError(nb_data_sum, nb_eeg_sum, feeg, feeg_err);

    omegapi_sfw = getscale(nb_data_sum, fomegapi, nb_omegapi_sum);
    omegapi_sfw_err = GetScalError(nb_data_sum, nb_omegapi_sum, fomegapi, fomegapi_err);

    kpm_sfw = getscale(nb_data_sum, fkpm, nb_kpm_sum);
    kpm_sfw_err = GetScalError(nb_data_sum, nb_kpm_sum, fkpm, fkpm_err);

    ksl_sfw = getscale(nb_data_sum, fksl, nb_ksl_sum);
    ksl_sfw_err = GetScalError(nb_data_sum, nb_ksl_sum, fksl, fksl_err);

    etagam_sfw = getscale(nb_data_sum, fetagam, nb_etagam_sum);
    etagam_sfw_err = GetScalError(nb_data_sum, nb_etagam_sum, fetagam, fetagam_err);
    
    mcrest_sfw  = getscale(nb_data_sum, fmcrest, nb_mcrest_sum);
    mcrest_sfw_err = GetScalError(nb_data_sum, nb_mcrest_sum, fmcrest, fmcrest_err);

    // write in the output file
    cout << "fisr3pi = " << fisr3pi << "+/-" << fisr3pi_err << "\n"
	 << "feeg = " << feeg << "+/-" << feeg_err << "\n"
	 << "fomegapi = " << fomegapi << "+/-" << fomegapi_err << "\n"
	 << "fkpm = " << fkpm << "+/-" << fkpm_err << "\n"   
 	 << "fksl = " << fksl << "+/-" << fksl_err << "\n"
	 << "fetagam = " << fetagam << "+/-" << fetagam_err << "\n"
	 << "fmcrest = " << fmcrest << "+/-" << fmcrest_err << "\n\n";

    cout << "isr3pi_sfw = " << isr3pi_sfw << "+/-" << isr3pi_sfw_err << "\n"
	 << "eeg_sfw = " << eeg_sfw << "+/-" << eeg_sfw_err << "\n"
	 << "omegapi_sfw = " << omegapi_sfw << "+/-" << omegapi_sfw_err << "\n"
	 << "kpm_sfw = " << kpm_sfw << "+/-" << kpm_sfw_err << "\n"    
	 << "ksl_sfw = " << ksl_sfw << "+/-" << ksl_sfw_err << "\n"
	 << "etagam_sfw = " << etagam_sfw << "+/-" << etagam_sfw_err << "\n"
	 << "mcrest_sfw = " << mcrest_sfw << "+/-" << mcrest_sfw_err << "\n";
      
      
    // << "const double etagam_sfw = " << etagam_sfw << ";\n"
    // << "const double mcrest_sfw = " << mcrest_sfw << ";\n";

    // Create scaled histograms
    TH1D* hM3pi_BDT_good_TISR3PI_SIG_SCALED = (TH1D*)hM3pi_BDT_good_TISR3PI_SIG -> Clone("hM3pi_BDT_good_TISR3PI_SIG_SCALED");
    hM3pi_BDT_good_TISR3PI_SIG_SCALED -> Scale(isr3pi_sfw);

    double orig_integral = hM3pi_BDT_good_TISR3PI_SIG->Integral();
    cout << "Original histogram: " << hM3pi_BDT_good_TISR3PI_SIG->GetName() << endl;
    cout << "  Integral: " << orig_integral << " events" << endl;
    cout << "  Entries: " << hM3pi_BDT_good_TISR3PI_SIG->GetEntries() << endl;
    
    cout << "Original name: " << hM3pi_BDT_good_TISR3PI_SIG->GetName() << endl;
    cout << "Original title: " << hM3pi_BDT_good_TISR3PI_SIG->GetTitle() << endl;
    cout << "Integral: " << hM3pi_BDT_good_TISR3PI_SIG->Integral() << endl;
    cout << "Entries: " << hM3pi_BDT_good_TISR3PI_SIG->GetEntries() << endl;
    cout << "X range: [" << hM3pi_BDT_good_TISR3PI_SIG->GetXaxis()->GetXmin() 
	 << ", " << hM3pi_BDT_good_TISR3PI_SIG->GetXaxis()->GetXmax() << "]" << endl;
    cout << "Maximum value: " << hM3pi_BDT_good_TISR3PI_SIG->GetMaximum() << endl;

    /*
    TCanvas *cv_scaled = new TCanvas("cv_scaled", "Scaled Signal", 1000, 800);
    cv_scaled->SetLeftMargin(0.15);
    cv_scaled->SetBottomMargin(0.15);
    cv_scaled->SetGrid();
    cv_scaled->SetLogy(0);  // Turn off log scale
    
    // Style the histogram
    hM3pi_BDT_good_TISR3PI_SIG_SCALED->SetTitle("Scaled Signal Histogram");
    hM3pi_BDT_good_TISR3PI_SIG_SCALED->GetXaxis()->SetTitle("M_{3#pi} [MeV/c^{2}]");
    hM3pi_BDT_good_TISR3PI_SIG_SCALED->GetXaxis()->SetTitleOffset(1.2);
    hM3pi_BDT_good_TISR3PI_SIG_SCALED->GetYaxis()->SetTitle("Events");
    hM3pi_BDT_good_TISR3PI_SIG_SCALED->GetYaxis()->SetTitleOffset(1.4);
    

    // Set colors to make it visible
    hM3pi_BDT_good_TISR3PI_SIG_SCALED->SetLineColor(kRed);
    hM3pi_BDT_good_TISR3PI_SIG_SCALED->SetLineWidth(2);
    hM3pi_BDT_good_TISR3PI_SIG_SCALED->SetFillColor(kRed-9);
    hM3pi_BDT_good_TISR3PI_SIG_SCALED->SetFillStyle(3001);
    
    // Set Y-axis range based on the scaled maximum (361)
    double max_val = hM3pi_BDT_good_TISR3PI_SIG_SCALED->GetMaximum();
    hM3pi_BDT_good_TISR3PI_SIG_SCALED->SetMaximum(max_val * 1.2);  // Add 20% headroom
    hM3pi_BDT_good_TISR3PI_SIG_SCALED->SetMinimum(0);
    
    // Draw
    hM3pi_BDT_good_TISR3PI_SIG_SCALED->Draw("HIST");
    
    // Update and save
    cv_scaled->Update();
    //cv_scaled->SaveAs("plots/scaled_signal.pdf");
    */
    
    // Print clone info
    cout << "\n========== CLONE INFO ==========" << endl;
    cout << "Clone integral: " << hM3pi_BDT_good_TISR3PI_SIG_SCALED->Integral() << endl;
    cout << "Clone entries: " << hM3pi_BDT_good_TISR3PI_SIG_SCALED->GetEntries() << endl;
    cout << "Clone maximum: " << hM3pi_BDT_good_TISR3PI_SIG_SCALED->GetMaximum() << endl;
    cout << "X range: [" << hM3pi_BDT_good_TISR3PI_SIG_SCALED->GetXaxis()->GetXmin()
	 << ", " << hM3pi_BDT_good_TISR3PI_SIG_SCALED->GetXaxis()->GetXmax() << "]" << endl;

    /*
    
    // Sum all components
    //TH2D *h_mc_total = (TH2D*)h_isr3pi_scaled->Clone("h_mc_total");
    //h_mc_total->Add(h_eeg_scaled);
    //h_mc_total->Add(h_omegapi_scaled);
    //h_mc_total->Add(h_kpm_scaled);
    //h_mc_total->Add(h_ksl_scaled);
    //h_mc_total->Add(h_etagam_scaled);
    //h_mc_total->Add(h_mcrest_scaled);
    */
    
    // Plot KLOE results, hM3pi MC and data
    //hM3pi_TDATA -> Draw("E");
    //hM3pi_bad_TDATA -> Draw("Same");
    
    //TCanvas *cv_m3pi_data = cv_plot("TDATA", hM3pi_TDATA, hM3pi_good_TDATA, hM3pi_bad_TDATA, hM3pi_BDT_TDATA, hM3pi_BDT_good_TDATA, hM3pi_BDT_bad_TDATA, "Events", "M_{3#pi} [MeV/c^{2}]", "Data", "Invariant mass of 3pi (Data)");
    
    //TCanvas *cv_m3pi_isr3pi = cv_plot("TISR3PI_SIG", hM3pi_TISR3PI_SIG, hM3pi_good_TISR3PI_SIG, hM3pi_bad_TISR3PI_SIG, hM3pi_BDT_TISR3PI_SIG, hM3pi_BDT_good_TISR3PI_SIG, hM3pi_BDT_bad_TISR3PI_SIG, "Events", "M_{3#pi} [MeV/c^{2}]", "MC", "Invariant mass of 3pi (Signal)");

    
    //int nentries = outtree -> GetEntries();
    //cout << "Tree has " << nentries << " entries" << endl;

    // Plot
    //TCanvas* cv2d_corr_data = plot_corr(h2dM3pi_kloeBDT_corr_TDATA, "TDATA", "", "Data");
    //TCanvas* cv2d_corr_sig = plot_corr(h2dM3pi_kloeBDT_corr_TISR3PI_SIG, "TISR3PI_SIG", "", "Signal");
    //TCanvas* cv2d_corr_etagam = plot_corr(h2dM3pi_kloeBDT_corr_TETAGAM, "TETAGAM", "", "#eta#gamma");

    
    //TCanvas *cv_signal = plot_sfw("TISR3PI_SIG", "Singal", h2d_sfw_BDT_good_TISR3PI_SIG, "Singal");
    //TCanvas *cv_etagam = plot_sfw("TETAGAM", "etagam", h2d_sfw_BDT_good_TETAGAM, "#eta#gamma");
    //TCanvas *cv_data = plot_sfw("TDATA", "data", h2d_sfw_BDT_good_TDATA, "Data");
    //TCanvas *cv_mcsum_noeta = plot_sfw("MCSUM_NOETA", "MC sum (no etagam)", h2d_sfw_BDT_good_MCSUM_NOETA, "Others");

    /*
    cv_m3pi_data -> SaveAs("./plots/cv_m3pi_data.pdf");
    cv_m3pi_isr3pi -> SaveAs("./plots/cv_m3pi_isr3pi.pdf");
      
    cv_signal -> SaveAs("./plots/cv_sfw2d_TISR3PI_SIG.pdf");
    cv_etagam -> SaveAs("./plots/cv_sfw2d_TETAGAM.pdf");
    cv_mcsum_noeta -> SaveAs("./plots/cv_sfw2d_TMCSUM_NOETA.pdf");
    cv_data -> SaveAs("./plots/cv_sfw2d_TDATA.pdf");
    */
    
    /*
    cv_m3pi_data -> SaveAs("cv_m3pi_data.pdf");
    cv_m3pi_isr3pi -> SaveAs("cv_m3pi_isr3pi.pdf");
      
    cv_signal -> SaveAs("cv_sfw2d_TISR3PI_SIG.pdf");
    cv_etagam -> SaveAs("cv_sfw2d_TETAGAM.pdf");
    cv_mcsum_noeta -> SaveAs("cv_sfw2d_TMCSUM_NOETA.pdf");
    cv_data -> SaveAs("cv_sfw2d_TDATA.pdf");
    */

    //hM3pi_BDT_good_TISR3PI_SIG_SCALED -> Write();
    //hM3pi_BDT_good_TDATA -> Write();
    //hM3pi_BDT_TDATA -> Write();
    //hM3pi_BDT_good_TISR3PI_SIG_TMP -> Write();
    //hM3pi_BDT_bad_TISR3PI_SIG_TMP -> Write();
    //hM3pi_BDT_TEEG -> Write();
    //hM3pi_BDT_TOMEGAPI -> Write();
    //hM3pi_BDT_TKPM -> Write();
    //hM3pi_BDT_TKSL -> Write();
    //hM3pi_BDT_TRHOPI -> Write();
    //hM3pi_BDT_TETAGAM -> Write();
    //hM3pi_BDT_TBKGREST -> Write();

    HistArray_m3pi -> Write();

    
    TSFW2D -> Write();
    f_output -> Close();
  
  }// end check input file existence.

  
  //return 0;
  
}

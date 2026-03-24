#include "fill_histos.h"
#include "helper_h2d.h"

TCanvas *plot_corr(TObjArray* HList, const TString var_type, const TString mc_type, const TString select_type, const TString cv_title, const TString x_title, const TString x_unit, const double xmin, const double xmax, const TString y_title, const TString y_unit, const double ymin, const double ymax) {

  TH2D *h2d = (TH2D *) HList -> FindObject(var_type + select_type + mc_type);
  
  TH1D *h2d_projx = h2d -> ProjectionX();
  TH1D *h2d_projy = h2d -> ProjectionY();
  
  double binwidth_x = getbinwidth(h2d_projx);
  double binwidth_y = getbinwidth(h2d_projy);
  
  TCanvas * cv2d =  new TCanvas("cv2d_" + var_type, cv_title, 0, 0, 700, 700);

  cv2d -> SetBottomMargin(0.15);//0.007
  cv2d -> SetLeftMargin(0.15);
  cv2d -> SetRightMargin(0.15);

  //h2d -> SetMinimum(10);

  h2d -> GetXaxis() -> SetNdivisions(5);
  h2d -> GetXaxis() -> SetTitle(x_title + " " + TString::Format("Events/%0.2f", binwidth_x) + " " + x_unit); //SetTitle(x_label + " " + x_unit);
  h2d -> GetXaxis() -> SetTitleOffset(1.);
  h2d -> GetXaxis() -> SetTitleSize(0.06);
  h2d -> GetXaxis() -> CenterTitle();
  h2d -> GetXaxis() -> SetLabelSize(0.06);
  h2d -> GetXaxis() -> SetLabelOffset(0.01);
  //h2d -> GetXaxis() -> SetRangeUser(0.2, 0.6);
  
  h2d -> GetYaxis() -> SetTitle(y_title + " " + TString::Format("Events/%0.2f", binwidth_y) + " " + y_unit); //SetTitle(x_label + " " + x_unit);
  h2d -> GetYaxis() -> SetLabelOffset(0.01);
  h2d -> GetYaxis() -> SetTitleOffset(1.2);
  h2d -> GetYaxis() -> SetLabelSize(0.05);
  h2d -> GetYaxis() -> SetTitleSize(0.06);
  h2d -> GetYaxis() -> CenterTitle();

  h2d -> Draw("COLZ");
  //h2d -> Draw("TEXT0COLZ");

  h2d -> GetZaxis() -> SetLabelSize(0.045);
  
  gPad->SetLogz(1);

  
  return cv2d;
  
}


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
      
      //cout << h2d_good -> GetName() << endl;

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

  // Plot
  for (int i = 0; i < list_size; i++) {
    
    TCanvas* cv2d_corr_data = plot_corr(HistArray_h2d, hist_type, ch_type[i], "_", cv_title, x_title, x_unit, xmin, xmax, y_title, y_unit, ymin, ymax);

    TH2D *h2d = (TH2D *) HistArray_h2d -> FindObject(hist_type + "_" + ch_type[i]);
  
    TPaveText *pt = set_pt(0.2, 0.86, 0.3, 0.86); //new TPaveText(0.2, 0.8, 0.3, 0.85, "NDC");
    
    //pt -> SetTextSize(0.08);
    //pt -> SetFillColor(0);
    //pt -> SetTextAlign(12);
    //pt -> AddText("Relative Error [%]");
    pt -> AddText(ch_type[i] + " BDT Selection (" + h2d -> GetEntries() + ")");
    
    pt -> Draw("Same");
    line1 -> Draw("Same");
    line2 -> Draw("Same");
    
    gPad -> SetLogz(1);

    cv2d_corr_data -> Update();
    cv2d_corr_data -> Modified();
    //cv2d_corr_data -> SaveAs("h2d_plots/cv2d_corr_bdt_" + ch_type[i] + "_full_mass_range" + ".pdf");
    cv2d_corr_data -> SaveAs("h2d_plots/cv2d_corr_bdt_" + ch_type[i] + "_mass_cut" + ".pdf");
    
    cv2d_corr_data -> Close();
    
  }
    
  
  return 0;

}

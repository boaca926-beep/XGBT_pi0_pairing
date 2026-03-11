#include "helper.h"

const double pt1_x0 = 0.4;
const double pt1_x1 = 0.7;

TList *HSFW1D = new TList();

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

TCanvas *plot_cv(const TString hist_type, const TString cv_nm, TH2D *h2d, const TString pt_str)
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

TCanvas *cv_plot(TH1D* h_kloe, TH1D* h_bdt, TH1D* h_good_bdt, TH1D* h_bad_bdt, TString y_title, TString x_title, const TString sample_type, const TString cv_title) {

  TCanvas *cv = new TCanvas("c1", cv_title, 700, 600);
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

    gPad->SetLogy(1); 
    
    TLegend *legd_cv = new TLegend(0.5, 0.65, 0.9, 0.9);
    
    legd_cv -> SetTextFont(132);
    legd_cv -> SetFillStyle(0);
    legd_cv -> SetBorderSize(0);
    legd_cv -> SetNColumns(1);
    
    legd_cv -> AddEntry(h_kloe, "KLOE", "lep");
    legd_cv -> AddEntry(h_bdt, "BDT Cut", "lep");
    legd_cv -> AddEntry(h_good_bdt, "BDT Good", "lep");
    
    if (sample_type == "Data") {
      h_kloe -> Draw("E");
      h_bdt -> Draw("ESame");
      h_good_bdt -> Draw("ESame");
      h_bad_bdt -> Draw("ESame");

      legd_cv -> AddEntry(h_bad_bdt, "BDT Comb. BKG", "lep");
    
    }
    else if (sample_type == "MC") {
      h_kloe -> Draw();
      h_bdt -> Draw("Same");
      h_good_bdt -> Draw("ESame");
      h_bad_bdt -> Draw("ESame");

      legd_cv -> AddEntry(h_bad_bdt, "BDT Comb. BKG", "f");
    
    }

    
    legd_cv -> Draw("Same");
    
    legtextsize(legd_cv, 0.04);
    
    return cv;
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
    
    // Open the root file
    //TFile* file = TFile::Open(input_filename);
    TFile* file = new TFile(input_filename);

    if (!file || file -> IsZombie()){
      cout << "Error: Cannot open file " << input_filename << endl;
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
	cout << "classnm = " << classnm_tree << ", objnm = " << objnm_tree << endl;
      }
      
      TTree *tree_tmp = (TTree*)file -> Get(objnm_tree);
      
    }
    
    // Get histos
    file -> cd(); // Make sure we're in the output file

    // TH2D
    TH2D *h2d_sfw_BDT_good_TEEG = (TH2D *) file -> Get("h2d_sfw_BDT_good_TEEG");
    TH2D *h2d_sfw_BDT_good_TDATA = (TH2D *) file -> Get("h2d_sfw_BDT_good_TDATA");
    TH2D *h2d_sfw_BDT_good_TISR3PI_SIG = (TH2D *) file -> Get("h2d_sfw_BDT_good_TISR3PI_SIG");
    TH2D *h2d_sfw_BDT_good_TOMEGAPI = (TH2D *) file -> Get("h2d_sfw_BDT_good_TOMEGAPI");
    TH2D *h2d_sfw_BDT_good_TKPM = (TH2D *) file -> Get("h2d_sfw_BDT_good_TKPM");
    TH2D *h2d_sfw_BDT_good_TKSL = (TH2D *) file -> Get("h2d_sfw_BDT_good_TKSL");
    //TH2D *h2d_sfw_BDT_good_T3PIGAM = (TH2D *) file -> Get("h2d_sfw_BDT_good_T3PIGAM");
    TH2D *h2d_sfw_BDT_good_TRHOPI = (TH2D *) file -> Get("h2d_sfw_BDT_good_TRHOPI");
    TH2D *h2d_sfw_BDT_good_TETAGAM = (TH2D *) file -> Get("h2d_sfw_BDT_good_TETAGAM");
    TH2D *h2d_sfw_BDT_good_TBKGREST = (TH2D *) file -> Get("h2d_sfw_BDT_good_TBKGREST");

    TH2D * h2d_sfw_BDT_good_MCREST;
    TH2D * h2d_sfw_BDT_good_MCSUM;

    // h2d_sfw_BDT_good_MCREST
    h2d_sfw_BDT_good_MCREST = (TH2D*) h2d_sfw_BDT_good_TBKGREST -> Clone();
    h2d_sfw_BDT_good_MCREST -> Add(h2d_sfw_BDT_good_TKPM, 1.);
    h2d_sfw_BDT_good_MCREST -> Add(h2d_sfw_BDT_good_TRHOPI, 1.);
    h2d_sfw_BDT_good_MCREST -> SetName("h2d_sfw_BDT_good_MCREST");
    
    // h2d_sfw_BDT_good_MCSUM
    h2d_sfw_BDT_good_MCSUM = (TH2D*) h2d_sfw_BDT_good_TEEG -> Clone();
    h2d_sfw_BDT_good_MCSUM -> Add(h2d_sfw_BDT_good_TOMEGAPI, 1.);
    h2d_sfw_BDT_good_MCSUM -> Add(h2d_sfw_BDT_good_TKSL, 1.);
    h2d_sfw_BDT_good_MCSUM -> Add(h2d_sfw_BDT_good_TETAGAM, 1.);
    h2d_sfw_BDT_good_MCSUM -> Add(h2d_sfw_BDT_good_TISR3PI_SIG, 1.);
    h2d_sfw_BDT_good_MCSUM -> Add(h2d_sfw_BDT_good_MCREST, 1.);
    h2d_sfw_BDT_good_MCSUM -> SetName("h2d_sfw_BDT_good_MCSUM");

    // mc sum no etagam
    TH2D *h2d_sfw_BDT_good_MCSUM_NOETA = (TH2D *) h2d_sfw_BDT_good_TEEG -> Clone();
    h2d_sfw_BDT_good_MCSUM_NOETA -> Add(h2d_sfw_BDT_good_TKSL, 1.);
    h2d_sfw_BDT_good_MCSUM_NOETA -> Add(h2d_sfw_BDT_good_TOMEGAPI, 1.);
    h2d_sfw_BDT_good_MCSUM_NOETA -> Add(h2d_sfw_BDT_good_MCREST, 1.);
    h2d_sfw_BDT_good_MCSUM_NOETA -> SetName("h2d_sfw_BDT_good_MCSUM_NOETA");


    // TH1D
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
    format_h(hM3pi_BDT_good_TDATA, 4, 2);
    format_h(hM3pi_BDT_bad_TDATA, 8, 2);
    

    // Plot KLOE results, hM3pi MC and data
    TCanvas *cv_3pi_data = cv_plot(hM3pi_TDATA, hM3pi_BDT_TDATA, hM3pi_BDT_good_TDATA, hM3pi_BDT_bad_TDATA, "Events", "M_{3#pi} [MeV/c^{2}]", "Data", "Invariant mass of 3pi (Data)");
    TCanvas *cv_3pi_isr3pi = cv_plot(hM3pi_TISR3PI_SIG, hM3pi_BDT_TISR3PI_SIG, hM3pi_BDT_good_TISR3PI_SIG, hM3pi_BDT_bad_TISR3PI_SIG, "Events", "M_{3#pi} [MeV/c^{2}]", "MC", "Invariant mass of 3pi (Signal)");
    
    //int nentries = outtree -> GetEntries();
    //cout << "Tree has " << nentries << " entries" << endl;

    /*
    // Plot
    TCanvas *cv_signal = plot_cv("TISR3PI_SIG", "Singal", h2d_sfw_BDT_good_TISR3PI_SIG, "Singal");
    TCanvas *cv_etagam = plot_cv("TETAGAM", "etagam", h2d_sfw_BDT_good_TETAGAM, "#eta#gamma");
    TCanvas *cv_data = plot_cv("TDATA", "data", h2d_sfw_BDT_good_TDATA, "Data");
    TCanvas *cv_mcsum_noeta = plot_cv("MCSUM_NOETA", "MC sum (no etagam)", h2d_sfw_BDT_good_MCSUM_NOETA, "Others");

    
    cv_signal -> SaveAs("cv_sfw2d_TISR3PI_SIG.pdf");
    cv_etagam -> SaveAs("cv_sfw2d_TETAGAM.pdf");
    cv_mcsum_noeta -> SaveAs("cv_sfw2d_TMCSUM_NOETA.pdf");
    cv_data -> SaveAs("cv_sfw2d_TDATA.pdf");
    */
    
  }// end check input file existence.

  
    
}

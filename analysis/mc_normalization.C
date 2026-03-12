#include "helper.h"
#include "sfw2d.h"

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
  
  gPad->SetLogy(1); 
  
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
	//cout << "classnm = " << classnm_tree << ", objnm = " << objnm_tree << endl;
      }
      
      TTree *tree_tmp = (TTree*)file -> Get(objnm_tree);
      
    }
    
    // Get histos
    file -> cd(); // Make sure we're in the output file

    // TH2D
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
    // TISR3PI_SIG
    TH1D* hM3pi_TISR3PI_SIG = (TH1D*)file -> Get("hM3pi_TISR3PI_SIG");
    TH1D* hM3pi_good_TISR3PI_SIG = (TH1D*)file -> Get("hM3pi_good_TISR3PI_SIG");
    TH1D* hM3pi_bad_TISR3PI_SIG = (TH1D*)file -> Get("hM3pi_bad_TISR3PI_SIG");

    TH1D* hM3pi_BDT_TISR3PI_SIG = (TH1D*)file -> Get("hM3pi_BDT_TISR3PI_SIG");
    TH1D* hM3pi_BDT_good_TISR3PI_SIG = (TH1D*)file -> Get("hM3pi_BDT_good_TISR3PI_SIG");
    TH1D* hM3pi_BDT_bad_TISR3PI_SIG = (TH1D*)file -> Get("hM3pi_BDT_bad_TISR3PI_SIG");

    format_h(hM3pi_TISR3PI_SIG, 1, 2);
    format_h(hM3pi_good_TISR3PI_SIG, 4, 2);
    formatfill_h(hM3pi_bad_TISR3PI_SIG, 4, 3001);

    format_h(hM3pi_BDT_TISR3PI_SIG, 2, 2);
    format_h(hM3pi_BDT_good_TISR3PI_SIG, 3, 2);
    formatfill_h(hM3pi_BDT_bad_TISR3PI_SIG, 2, 3001);

    //
    TH1D* hM3pi_TDATA = (TH1D*)file -> Get("hM3pi_TDATA");
    TH1D* hM3pi_good_TDATA = (TH1D*)file -> Get("hM3pi_good_TDATA");
    TH1D* hM3pi_bad_TDATA = (TH1D*)file -> Get("hM3pi_bad_TDATA");
    
    TH1D* hM3pi_BDT_TDATA = (TH1D*)file -> Get("hM3pi_BDT_TDATA");
    TH1D* hM3pi_BDT_good_TDATA = (TH1D*)file -> Get("hM3pi_BDT_good_TDATA");
    TH1D* hM3pi_BDT_bad_TDATA = (TH1D*)file -> Get("hM3pi_BDT_bad_TDATA");

    format_h(hM3pi_TDATA, 1, 2);
    format_h(hM3pi_good_TDATA, 4, 2);
    formatfill_h(hM3pi_bad_TDATA, 4, 3001);
    
    
    format_h(hM3pi_BDT_TDATA, 2, 2);
    format_h(hM3pi_BDT_good_TDATA, 4, 2);
    format_h(hM3pi_BDT_bad_TDATA, 2, 2);

    // Preparing MC normalization
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
    TTree* TSFW2D = new TTree("TSFW2D", "recreate");
    TSFW2D -> SetAutoSave(0);
    
    TSFW2D -> Branch("Br_nb_data", &nb_data, "Br_nb_data/D");
    TSFW2D -> Branch("Br_nb_eeg", &nb_eeg, "Br_nb_eeg/D");
    TSFW2D -> Branch("Br_nb_ksl", &nb_ksl, "Br_nb_ksl/D");
    TSFW2D -> Branch("Br_nb_omegapi", &nb_omegapi, "Br_nb_omegapi/D");
    TSFW2D -> Branch("Br_nb_etagam", &nb_etagam, "Br_nb_etagam/D");
    TSFW2D -> Branch("Br_nb_isr3pi", &nb_isr3pi, "Br_nb_isr3pi/D");
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
    h2d_sfw_BDT_good_MCSUM -> Draw();
    
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
	 
    /*
    // Plot KLOE results, hM3pi MC and data
    hM3pi_TDATA -> Draw("E");
    hM3pi_bad_TDATA -> Draw("Same");
    
    TCanvas *cv_m3pi_data = cv_plot("TDATA", hM3pi_TDATA, hM3pi_good_TDATA, hM3pi_bad_TDATA, hM3pi_BDT_TDATA, hM3pi_BDT_good_TDATA, hM3pi_BDT_bad_TDATA, "Events", "M_{3#pi} [MeV/c^{2}]", "Data", "Invariant mass of 3pi (Data)");
    
    TCanvas *cv_m3pi_isr3pi = cv_plot("TISR3PI_SIG", hM3pi_TISR3PI_SIG, hM3pi_good_TISR3PI_SIG, hM3pi_bad_TISR3PI_SIG, hM3pi_BDT_TISR3PI_SIG, hM3pi_BDT_good_TISR3PI_SIG, hM3pi_BDT_bad_TISR3PI_SIG, "Events", "M_{3#pi} [MeV/c^{2}]", "MC", "Invariant mass of 3pi (Signal)");
    
    //int nentries = outtree -> GetEntries();
    //cout << "Tree has " << nentries << " entries" << endl;

    // Plot
    TCanvas *cv_signal = plot_sfw("TISR3PI_SIG", "Singal", h2d_sfw_BDT_good_TISR3PI_SIG, "Singal");
    TCanvas *cv_etagam = plot_sfw("TETAGAM", "etagam", h2d_sfw_BDT_good_TETAGAM, "#eta#gamma");
    TCanvas *cv_data = plot_sfw("TDATA", "data", h2d_sfw_BDT_good_TDATA, "Data");
    TCanvas *cv_mcsum_noeta = plot_sfw("MCSUM_NOETA", "MC sum (no etagam)", h2d_sfw_BDT_good_MCSUM_NOETA, "Others");
    */
    
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
    
  }// end check input file existence.

  
    
}

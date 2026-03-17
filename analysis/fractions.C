#include "helper.h"

void fractions(const char* input_filename = "./sfw2d.root") {

  TFile* file = new TFile(input_filename);

  TH1D* hdata = (TH1D*)file -> Get("hM3pi_BDT_TDATA");
  TH1D* hsig = (TH1D*)file -> Get("hM3pi_BDT_good_TISR3PI_SIG");
  TH1D* hsig_comb = (TH1D*)file -> Get("hM3pi_BDT_bad_TISR3PI_SIG");
  TH1D* h3pigam = (TH1D*)file -> Get("hM3pi_BDT_good_T3PIGAM");
  TH1D* h3pigam_comb = (TH1D*)file -> Get("hM3pi_BDT_bad_T3PIGAM");
  TH1D* heeg = (TH1D*)file -> Get("hM3pi_BDT_TEEG");
  TH1D* homegapi = (TH1D*)file -> Get("hM3pi_BDT_TOMEGAPI");
  TH1D* hkpm = (TH1D*)file -> Get("hM3pi_BDT_TKPM");
  TH1D* hksl = (TH1D*)file -> Get("hM3pi_BDT_TKSL");
  TH1D* hrhopi = (TH1D*)file -> Get("hM3pi_BDT_TRHOPI");
  TH1D* hetagam = (TH1D*)file -> Get("hM3pi_BDT_TETAGAM");
  TH1D* hmcrest = (TH1D*)file -> Get("hM3pi_BDT_TBKGREST");

  TH1D* hM3pi_BDT_T3PIGAM = (TH1D*)file -> Get("hM3pi_BDT_T3PIGAM");
  TH1D* hM3pi_BDT_TISR3PI_SIG = (TH1D*)file -> Get("hM3pi_BDT_TISR3PI_SIG");
  
  const double sf_tmp =  hM3pi_BDT_T3PIGAM -> GetEntries() / hM3pi_BDT_TISR3PI_SIG -> GetEntries();
  
  cout << "sf_tmp = " << sf_tmp << endl;
    
  // Verify all histograms exist
  TH1D* histos[] = {hdata, hsig, hsig_comb, heeg, homegapi, hkpm, hksl, hrhopi, hetagam, hmcrest};
  const char* names[] = {"hM3pi_BDT_TDATA",
			 "hM3pi_BDT_good_TISR3PI_SIG_TMP",
			 "hM3pi_BDT_bad_TISR3PI_SIG_TMP",
			 "hM3pi_BDT_TEEG",
			 "hM3pi_BDT_TOMEGAPI",
			 "hM3pi_BDT_TKPM",
			 "hM3pi_BDT_TKSL",
			 "hM3pi_BDT_TRHOPI",
			 "hM3pi_BDT_TETAGAM",
			 "hM3pi_BDT_TBKGREST"};

  
  for (int i = 0; i < 10; i++) {
    if (!histos[i]) {
      std::cout << "Error: Cannot find histogram " << names[i] << std::endl;
      file->Close();
      return;
    }
  }

  
  //==================================== Combine similar components =================================

  // Clone histograms to avoid modifying originals
  TH1D* hsig_total = (TH1D*)hsig -> Clone("hsig_tmp");
  hsig_total -> Add(hsig_comb);
  
  // Combine all backgrounds
  TH1D* hbkg_total = (TH1D*)heeg->Clone("hbkg_total");
  hbkg_total->Add(homegapi);
  hbkg_total->Add(hkpm);
  hbkg_total->Add(hksl);
  hbkg_total->Add(hrhopi);
  hbkg_total->Add(hetagam);
  hbkg_total->Add(hmcrest);

  std::cout << "\nSTEP 1: Fit signal vs background" << std::endl;
  std::cout << "  Total signal integral: " << hsig_total->Integral() << std::endl;
  std::cout << "  Total background integral: " << hbkg_total->Integral() << std::endl;
  
  TObjArray* MCList = new TObjArray(2);
  MCList -> Add(hsig_total);
  MCList -> Add(hbkg_total);
  
  
    
  //TH1D* heeg_tmp = (TH1D*)heeg -> Clone("heeg_tmp");
  //TH1D* homegapi_tmp = (TH1D*)homegapi -> Clone("homegapi_tmp");
  //TH1D* hksl_tmp = (TH1D*)hksl -> Clone("hksl_tmp");
  //TH1D* hetagam_tmp = (TH1D*)hetagam -> Clone("hetagam_tmp");
  
  // Combine all small backgrounds into one "other" component
  //TH1D* hother = (TH1D*)hkpm->Clone("hother");
  //hother->Add(hrhopi);
  //hother->Add(hmcrest);

  
  //==================================== TFractionFitter =================================
  //TObjArray* MCList = new TObjArray(7);
  //MCList -> Add(hsig_tmp);
  //MCList -> Add(hsig_comb_tmp);
  //MCList -> Add(heeg_tmp);
  //MCList -> Add(homegapi_tmp);
  //MCList -> Add(hksl_tmp);
  //MCList -> Add(hetagam_tmp);
  //MCList -> Add(hother);

  std::cout << "Added " << MCList->GetEntries() << " templates for fitting." << std::endl;

  // Normalize MC templates to unity
  for (int i = 0; i < MCList -> GetEntries(); i ++) {
    TH1D* h = (TH1D*)MCList -> At(i);
    double integral = h -> Integral();
    if (integral > 0) {
      h -> Scale(1.0 / integral);
      cout << "Template " << i << ", " << h -> GetName() << " normalized (integral = " << integral << ")" << endl;
    } else {
      cout << "Warning: Template: " << i << " has zero integral!" << endl;
    }
  }
  
  
  TVirtualFitter::SetDefaultFitter("Minuit2");
  
  TFractionFitter* fit = new TFractionFitter(hdata, MCList);

  for (int i = 0; i < MCList -> GetEntries(); i ++) {
    fit->Constrain(i, 0.0, 1.0);
  }
  
  int status = fit -> Fit();
  
  /*
  //
  hdata -> Draw("E");
  hsig -> Draw("HistSame");
  hsig_comb -> Draw("HistSame");
  homegapi -> Draw("HistSame");
  hksl -> Draw("HistSame");
  
  gPad->SetLogy(1); 
  */


}

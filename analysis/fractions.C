#include "helper.h"

void fractions(const char* input_filename = "./sfw2d.root") {

  TFile* file = new TFile(input_filename);

  TH1D* hdata = (TH1D*)file -> Get("hM3pi_BDT_TDATA");
  TH1D* hsig = (TH1D*)file -> Get("hM3pi_BDT_good_TISR3PI_SIG_TMP");
  TH1D* hsig_comb = (TH1D*)file -> Get("hM3pi_BDT_bad_TISR3PI_SIG_TMP");
  TH1D* heeg = (TH1D*)file -> Get("hM3pi_BDT_TEEG");
  TH1D* homegapi = (TH1D*)file -> Get("hM3pi_BDT_TOMEGAPI");
  TH1D* hkpm = (TH1D*)file -> Get("hM3pi_BDT_TKPM");
  TH1D* hksl = (TH1D*)file -> Get("hM3pi_BDT_TKSL");
  TH1D* hrhopi = (TH1D*)file -> Get("hM3pi_BDT_TRHOPI");
  TH1D* hetagam = (TH1D*)file -> Get("hM3pi_BDT_TETAGAM");
  TH1D* hmcrest = (TH1D*)file -> Get("hM3pi_BDT_TBKGREST");

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

  /*
  // Clone histograms to avoid modifying originals
  TH1D* hsig_good = (TH1D*)hsig->Clone("hsig_good");
  TH1D* hsig_bad = (TH1D*)hsig_comb->Clone("hsig_bad");
  
  // Combine all small backgrounds into one "other" component
  TH1D* hother = (TH1D*)heeg->Clone("hother");
  hother->Add(homegapi);
  hother->Add(hkpm);
  hother->Add(hksl);
  hother->Add(hrhopi);
  hother->Add(hetagam);
  hother->Add(hmcrest);
  
  std::cout << "\nCreated combined 'other' background with integral: " 
            << hother->Integral() << std::endl;
  */

  //==================================== TFractionFitter =================================
  TObjArray* MCList = new TObjArray(9);
  MCList -> Add(hsig);
  MCList -> Add(hsig_comb);
  MCList -> Add(heeg);
  MCList -> Add(homegapi);
  MCList -> Add(hkpm);
  MCList -> Add(hksl);
  MCList -> Add(hrhopi);
  MCList -> Add(hetagam);
  MCList -> Add(hmcrest);

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

  /*
  
  TVirtualFitter::SetDefaultFitter("Minuit");
  
  TFractionFitter* fit = new TFractionFitter(hdata, MCList);

  for (int i = 0; i < MCList -> GetEntries(); i ++) {
    fit->Constrain(i, 0.0, 1.0);
  }
  
  int status = fit -> Fit();
  */
  
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

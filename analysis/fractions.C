#include "helper.h"

void fractions(const char* input_filename = "./sfw2d.root") {

  TFile* file = new TFile(input_filename);

  TH1D* hdata = (TH1D*)file -> Get("hM3pi_BDT_TDATA");
  TH1D* hsig = (TH1D*)file -> Get("hM3pi_BDT_good_TISR3PI_SIG_TMP");
  TH1D* hsig_comb = (TH1D*)file -> Get("hM3pi_BDT_bad_TISR3PI_SIG_TMP");

  TH1D* homegapi = (TH1D*)file -> Get("hM3pi_BDT_TOMEGAPI");
  TH1D* hksl = (TH1D*)file -> Get("hM3pi_BDT_TKSL");

  //==================================== TFractionFitter =================================
  TObjArray* MCList = new TObjArray(4);
  MCList -> Add(hsig);
  MCList -> Add(hsig_comb);
  MCList -> Add(homegapi);
  MCList -> Add(hksl);

  TFractionFitter* fit = new TFractionFitter(hdata, MCList);
  fit->Constrain(0, 0.0, 1.0);
  fit->Constrain(1, 0.0, 1.0);
  fit->Constrain(2, 0.0, 1.0);  // Still constrain range even if fixing
  fit->Constrain(3, 0.0, 1.0);

  fit -> Fit();

  
  //
  hdata -> Draw("E");
  hsig -> Draw("HistSame");
  hsig_comb -> Draw("HistSame");
  homegapi -> Draw("HistSame");
  hksl -> Draw("HistSame");
  
  gPad->SetLogy(1); 
  
}

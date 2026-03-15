#include "helper.h"

void plot_hist(const char* input_filename = "./output_main_bdt.root") {

  TFile* file = new TFile(input_filename);

  TH1D* hM3pi_BDT_TBKGREST = (TH1D*)file -> Get("hM3pi_BDT_TBKGREST");
  format_h(hM3pi_BDT_TBKGREST, 42, 2);

  hM3pi_BDT_TBKGREST -> Draw();
  
}

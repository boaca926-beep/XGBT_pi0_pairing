#include <TMVA/RBDT.hxx>
#include <TMVA/RTensor.hxx>
#include <ROOT/RDataFrame.hxx>
#include <TFile.h>
#include <TTree.h>
#include <TSystem.h>
#include <iostream>
#include <TCanvas.h>
#include <TLegend.h>
#include <TPaveText.h>
#include <TMath.h>

using namespace TMVA::Experimental;

double pi = TMath::Pi();

void legtextsize(TLegend* l, Double_t size) {
  for(int i = 0 ; i < l -> GetListOfPrimitives() -> GetSize() ; i++) {
    TLegendEntry *header = (TLegendEntry*)l->GetListOfPrimitives()->At(i);
    header->SetTextSize(size);
  }
}

//
void PteAttr(TPaveText *pt) {

  pt -> SetTextSize(0.04);
  pt -> SetFillColor(0);
  pt -> SetTextAlign(12);
  pt -> SetBorderSize(0);
}

void format_h(TH1D* h, Int_t linecolor, Int_t width) {
  h->SetLineColor(linecolor);
  //cout << "histo format" << endl;
  h->SetLineWidth(width);
}

void formatfill_h(TH1D* h, Int_t fillcolor, Int_t fillstyle) {
  h -> SetFillStyle(fillstyle);
  h -> SetFillColor(fillcolor);
  h -> SetLineColor(0);
}

double get_cos_theta(int i_idx, int j_idx, double photons[3][4]){

    double p1_mag = TMath::Sqrt(TMath::Max(0.0, 
        photons[i_idx][1] * photons[i_idx][1] + 
        photons[i_idx][2] * photons[i_idx][2] +
        photons[i_idx][3] * photons[i_idx][3]) 
    );

    double p2_mag = TMath::Sqrt(TMath::Max(0.0,
        photons[j_idx][1] * photons[j_idx][1] + 
        photons[j_idx][2] * photons[j_idx][2] +
        photons[j_idx][3] * photons[j_idx][3])
    );

    double dot = photons[i_idx][1] * photons[j_idx][1] + 
    photons[i_idx][2] * photons[j_idx][2] +
    photons[i_idx][3] * photons[j_idx][3];

    double cos_theta = dot / (p1_mag * p2_mag + 1e-10);
    cos_theta = TMath::Max(-1.0, TMath::Min(1.0, cos_theta));

    return cos_theta;
}

//
double inv_mass_4vector(int i_idx, int j_idx, double photons[3][4]){
    double inv_mass = 0;

    // Calculate total energy and momentum
    double e = photons[i_idx][0] + photons[j_idx][0];
    
    double px = photons[i_idx][1] + photons[j_idx][1];
    double py = photons[i_idx][2] + photons[j_idx][2];
    double pz = photons[i_idx][3] + photons[j_idx][3];

    double mass_square = e*e - (px*px + py*py + pz*pz);
    
    if (mass_square < 0 && mass_square > -1e-10) {
        return 0;
    }
    
    return (mass_square > 0) ? TMath::Sqrt(mass_square) : 0;

}

//
double inv_3pimass_4vector(int i_idx, int j_idx, double photons[3][4], double trk[2][4]){
    double inv_mass = 0;

    // Calculate total energy and momentum
    double e = photons[i_idx][0] + photons[j_idx][0] + trk[0][0] + trk[1][0];
    
    double px = photons[i_idx][1] + photons[j_idx][1] + trk[0][1] + trk[1][1];
    double py = photons[i_idx][2] + photons[j_idx][2] + trk[0][2] + trk[1][2];
    double pz = photons[i_idx][3] + photons[j_idx][3] + trk[0][3] + trk[1][3];

    double mass_square = e*e - (px*px + py*py + pz*pz);
    
    if (mass_square < 0 && mass_square > -1e-8) {
        return 0;
    }
    
    return (mass_square > 0) ? TMath::Sqrt(mass_square) : 0;

}

void main_analysis(const char* model_filename = "../training/models/bdt_pi0_TCOMB.root",
                const char* data_filename = "../data/kloe_small_sample.root"){

  gErrorIgnoreLevel = kError;
  TGaxis::SetMaxDigits(4);
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  gStyle->SetFitFormat("6.4g");
  gSystem->Load("libTMVA");
  gSystem->Load("libTMVAUtils");
    
  cout << "BDT full anlysis on KLOE dataset ..." << endl;

  // 1. Check if model file exists
  if (gSystem->AccessPathName(model_filename)) {
    std::cout << "" << model_filename << " does not exists!" << std::endl;  // Added std::
    return;
  }

  // If data file exists, process it with RDataFame
    if(!gSystem -> AccessPathName(data_filename)){
    
        cout << "\nProcessing data file: " << data_filename << endl;

        // Open the root file
        TFile* file = TFile::Open(data_filename);
        if (!file || file -> IsZombie())
        {
            cout << "Error: Cannot open file " << data_filename << endl;
            return;
        }

	// Loop over keys
	TIter next_tree(file -> GetListOfKeys());

	TString objnm_tree, classnm_tree;

	TKey *key;

	while ((key = (TKey *) next_tree())) { // key loop

	  objnm_tree = key -> GetName();
	  classnm_tree = key -> GetClassName();
	  key -> GetSeekKey();

	  cout << "classnm = " << classnm_tree << ", objnm = " << objnm_tree << endl;

	  cout << "===============Staring Pi0 Selection=============" << endl;
	  // Define histos
	  TH1D* he1 = new TH1D("he1_" + objnm_tree, "", 200, 0, 500);
	  TH1D* he2 = new TH1D("he2_" + objnm_tree, "", 200, 0, 500);
	  TH1D* he3 = new TH1D("he3_" + objnm_tree, "", 200, 0, 500);
	  
	  TH1D* hm_gg = new TH1D("hm_gg_" + objnm_tree, "", 200, 0, 1000);
	  TH1D* hcos_theta = new TH1D("hcos_theta_" + objnm_tree, "", 200, -1, 1);
	  TH1D* hopen_angle = new TH1D("hopen_angle_" + objnm_tree, "", 200, 0, pi);
	  TH1D* hE_asym = new TH1D("hE_asym_" + objnm_tree, "", 200, 0, 1);
	  TH1D* he_min_x_angle = new TH1D("he_min_x_angle_" + objnm_tree, "", 200, 0, 1000);
	  TH1D* hE_diff = new TH1D("hE_diff_" + objnm_tree, "", 200, 0, 500);
	  TH1D* hasym_x_angle = new TH1D("hasym_x_angle_" + objnm_tree, "", 200, 0, pi);
	  
	  TH1D* hM3pi_BDT = new TH1D("hM3pi_BDT_" + objnm_tree, "", 200, 400, 1000); // BDT selection
	  TH1D* hM3pi_BDT_good = new TH1D("hM3pi_BDT_good_" + objnm_tree, "", 200, 400, 1000);
	  TH1D* hM3pi_BDT_bad = new TH1D("hM3pi_BDT_bad_" + objnm_tree, "", 200, 400, 1000);
	  
	  TH1D* hM_gg_BDT = new TH1D("hM_gg_BDT_" + objnm_tree, "", 200, 50, 200); 
	  TH1D* hM_gg_BDT_good = new TH1D("hM_gg_BDT_good_" + objnm_tree, "", 200, 50, 200);
	  TH1D* hM_gg_BDT_bad = new TH1D("hM_gg_BDT_bad_" + objnm_tree, "", 200, 50, 200);
	  
	  TH1D* hE1_BDT_good = new TH1D("hE1_BDT_good_" + objnm_tree, "", 200, 0, 500); 
	  TH1D* hE1_BDT_bad = new TH1D("hE1_BDT_bad_" + objnm_tree, "", 200, 0, 500); 
	  
	  TH1D* hE2_BDT_good = new TH1D("hE2_BDT_good_" + objnm_tree, "", 200, 0, 500); 
	  TH1D* hE2_BDT_bad = new TH1D("hE2_BDT_bad_" + objnm_tree, "", 200, 0, 500); 
	  
	  TH1D* hE1 = new TH1D("hE1_" + objnm_tree, "", 200, 0, 500); // KLOE selection
	  TH1D* hE1_good = new TH1D("hE1_good_" + objnm_tree, "", 200, 0, 500); 
	  TH1D* hE1_bad = new TH1D("hE1_bad_" + objnm_tree, "", 200, 0, 500); 
	  
	  TH1D* hE2 = new TH1D("hE2_" + objnm_tree, "", 200, 0, 500); 
	  TH1D* hE2_good = new TH1D("hE2_good_" + objnm_tree, "", 200, 0, 500); 
	  TH1D* hE2_bad = new TH1D("hE2_bad_" + objnm_tree, "", 200, 0, 500); 
	  
	  TH1D* hM_gg = new TH1D("hM_gg_" + objnm_tree, "", 200, 50, 200); 
	  TH1D* hM_gg_good = new TH1D("hM_gg_good_" + objnm_tree, "", 200, 50, 200);
	  TH1D* hM_gg_bad = new TH1D("hM_gg_bad_" + objnm_tree, "", 200, 50, 200);
	  
	  TH1D* hM3pi = new TH1D("hM3pi_" + objnm_tree, "", 200, 400, 1000);
	  TH1D* hM3pi_good = new TH1D("hM3pi_good_" + objnm_tree, "", 200, 400, 1000);
	  TH1D* hM3pi_bad = new TH1D("hM3pi_bad_" + objnm_tree, "", 200, 400, 1000);
	  
	  

	  cout << "===============End Pi0 Selection=============" << endl;
	  
	} // end key loop
	
    } // check input root file end
} // main program end


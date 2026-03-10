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

  // 2. Load the BDT model
  std::cout << "Loading model from " << model_filename << std::endl;  // Added std::
  TMVA::Experimental::RBDT bdt("BDT_pi0", model_filename);  // ← Declaration is HERE
  
  std::cout << "✓ Model loaded successfully!" << std::endl;  // Added this line
  
  //std::cout << "Number of input features: " << bdt.GetNInputDim() << std::endl;
  
  // 3. Create output file
  TFile* outfile = TFile::Open("output_with_bdt.root", "RECREATE");

  
  
  // 4.If data file exists, process it with RDataFame
  if(!gSystem -> AccessPathName(data_filename)){ // check input root file
    
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
    
    int ch_nb = 0;
    
    while ((key = (TKey *) next_tree())) { // key loop
      
      objnm_tree = key -> GetName();
      classnm_tree = key -> GetClassName();
      key -> GetSeekKey();
      
      cout << "classnm = " << classnm_tree << ", objnm = " << objnm_tree << endl;

      outfile -> cd(); // Make sure we're in the output file
      TTree* outtree = new TTree(Form("%s", objnm_tree.Data()),  "Tree with BDT response");

      // Declare output variables
      int out_event;
      int pi0_pho1_idx, pi0_pho2_idx; // indices of bdt pi0 photons
      int prompt_pho_idx; // prompt photon index
      double bdt_score;
      
      int evnt_KLOE = 0;
      int evnt_good = 0;
      int evnt_bad = 0;
      
      int bdt_indx = -999;
      int kloe_indx = -999;
      
      const double energy_threshold = 5.0;
      
      int n_found = 0;
      
      double m_gg_bdt = 0, m_gg = 0;
      double m3pi = 0, m3pi_bdt = 0;
      
      double e1_bdt = 0, e2_bdt = 0, e3_bdt = 0;
      double opening_angle = 0, cos_theta = 0;
      double E_asym = 0, e_min_x_angle = 0;
      double asym_x_angle = 0, E_diff = 0;
      
      // Get the tree
      TTree* tree = (TTree*)file -> Get(objnm_tree);
      if (!tree) {
	cout << "Error: Cannot fine '' in file" << endl;
	file -> Close();
	return;
      }
      
      int nentries = tree -> GetEntries();
      cout << "Tree has " << nentries << " entries" << endl;
      
      // Set branch addres for input features
      double lagvalue_min_7C;
      double E1, px1, py1, pz1;
      double E2, px2, py2, pz2;
      double E3, px3, py3, pz3;
      double ppl_E, ppl_px, ppl_py, ppl_pz;
      double pmi_E, pmi_px, pmi_py, pmi_pz;
      
      int bkg_indx, recon_indx;
      
      tree -> SetBranchAddress("Br_lagvalue_min_7C", &lagvalue_min_7C);
      tree -> SetBranchAddress("Br_bkg_indx", &bkg_indx);
      tree -> SetBranchAddress("Br_recon_indx", &recon_indx);
      
      tree -> SetBranchAddress("Br_ppl_E", &ppl_E);
      tree -> SetBranchAddress("Br_ppl_px", &ppl_px);
      tree -> SetBranchAddress("Br_ppl_py", &ppl_py);
      tree -> SetBranchAddress("Br_ppl_pz", &ppl_pz);
      
      tree -> SetBranchAddress("Br_pmi_E", &pmi_E);
      tree -> SetBranchAddress("Br_pmi_px", &pmi_px);
      tree -> SetBranchAddress("Br_pmi_py", &pmi_py);
      tree -> SetBranchAddress("Br_pmi_pz", &pmi_pz);
      
      tree -> SetBranchAddress("Br_E1", &E1);
      tree -> SetBranchAddress("Br_px1", &px1);
      tree -> SetBranchAddress("Br_py1", &py1);
      tree -> SetBranchAddress("Br_pz1", &pz1);
      
      tree -> SetBranchAddress("Br_E2", &E2);
      tree -> SetBranchAddress("Br_px2", &px2);
      tree -> SetBranchAddress("Br_py2", &py2);
      tree -> SetBranchAddress("Br_pz2", &pz2);
      
      tree -> SetBranchAddress("Br_E3", &E3);
      tree -> SetBranchAddress("Br_px3", &px3);
      tree -> SetBranchAddress("Br_py3", &py3);
      tree -> SetBranchAddress("Br_pz3", &pz3);
      
      
      // Define histos
      TH1D* he1 = new TH1D(Form("he1_%s", objnm_tree.Data()), "", 200, 0, 500);
      TH1D* he2 = new TH1D(Form("he2_%s", objnm_tree.Data()), "", 200, 0, 500);
      TH1D* he3 = new TH1D(Form("he3_%s", objnm_tree.Data()), "", 200, 0, 500);
      
      TH1D* hm_gg = new TH1D(Form("hm_gg_%s", objnm_tree.Data()), "", 200, 0, 1000);
      TH1D* hcos_theta = new TH1D(Form("hcos_theta_%s", objnm_tree.Data()), "", 200, -1, 1);
      TH1D* hopen_angle = new TH1D(Form("hopen_angle_%s", objnm_tree.Data()), "", 200, 0, pi);
      TH1D* hE_asym = new TH1D(Form("hE_asym_%s", objnm_tree.Data()), "", 200, 0, 1);
      TH1D* he_min_x_angle = new TH1D(Form("he_min_x_angle_%s", objnm_tree.Data()), "", 200, 0, 1000);
      TH1D* hE_diff = new TH1D(Form("hE_diff_%s", objnm_tree.Data()), "", 200, 0, 500);
      TH1D* hasym_x_angle = new TH1D(Form("hasym_x_angle_%s", objnm_tree.Data()), "", 200, 0, pi);
      
      TH1D* hM3pi_BDT = new TH1D(Form("hM3pi_BDT_%s", objnm_tree.Data()), "", 200, 400, 1000);
      TH1D* hM3pi_BDT_good = new TH1D(Form("hM3pi_BDT_good_%s", objnm_tree.Data()), "", 200, 400, 1000);
      TH1D* hM3pi_BDT_bad = new TH1D(Form("hM3pi_BDT_bad_%s", objnm_tree.Data()), "", 200, 400, 1000);

      
      TH1D* hM_gg_BDT = new TH1D(Form("hM_gg_BDT_%s", objnm_tree.Data()), "", 200, 50, 200); 
      TH1D* hM_gg_BDT_good = new TH1D(Form("hM_gg_BDT_good_%s", objnm_tree.Data()), "", 200, 50, 200);
      TH1D* hM_gg_BDT_bad = new TH1D(Form("hM_gg_BDT_bad_%s", objnm_tree.Data()), "", 200, 50, 200);
      
      TH1D* hE1_BDT_good = new TH1D(Form("hE1_BDT_good_%s", objnm_tree.Data()), "", 200, 0, 500); 
      TH1D* hE1_BDT_bad = new TH1D(Form("hE1_BDT_bad_%s", objnm_tree.Data()), "", 200, 0, 500); 
      
      TH1D* hE2_BDT_good = new TH1D(Form("hE2_BDT_good_%s", objnm_tree.Data()), "", 200, 0, 500);
      TH1D* hE2_BDT_bad = new TH1D(Form("hE2_BDT_bad_%s", objnm_tree.Data()), "", 200, 0, 500); 
      
      TH1D* hE1 = new TH1D(Form("hE1_%s", objnm_tree.Data()), "", 200, 0, 500); // KLOE selection
      TH1D* hE1_good = new TH1D(Form("hE1_good_%s", objnm_tree.Data()), "", 200, 0, 500); 
      TH1D* hE1_bad = new TH1D(Form("hE1_bad_%s", objnm_tree.Data()), "", 200, 0, 500); 
      
      TH1D* hE2 = new TH1D(Form("hE2_%s", objnm_tree.Data()), "", 200, 0, 500); 
      TH1D* hE2_good = new TH1D(Form("hE2_good_%s", objnm_tree.Data()), "", 200, 0, 500); 
      TH1D* hE2_bad = new TH1D(Form("hE2_bad_%s", objnm_tree.Data()), "", 200, 0, 500);

      
      TH1D* hM_gg = new TH1D(Form("hM_gg_%s", objnm_tree.Data()), "", 200, 50, 200); 
      TH1D* hM_gg_good = new TH1D(Form("hM_gg_good_%s", objnm_tree.Data()), "", 200, 50, 200);
      TH1D* hM_gg_bad = new TH1D(Form("hM_gg_bad_%s", objnm_tree.Data()), "", 200, 50, 200);

      TH1D* hM3pi = new TH1D(Form("hM3pi_%s", objnm_tree.Data()), "", 200, 400, 1000);
      TH1D* hM3pi_good = new TH1D(Form("hM3pi_good_%s", objnm_tree.Data()), "", 200, 400, 1000);
      TH1D* hM3pi_bad = new TH1D(Form("hM3pi_bad_%s", objnm_tree.Data()), "", 200, 400, 1000);
      
      // Add branches
      outtree -> Branch("event", &out_event);
      outtree->Branch("pi0_pho1_idx", &pi0_pho1_idx);
      outtree->Branch("pi0_pho2_idx", &pi0_pho2_idx);
      outtree->Branch("prompt_pho_idx", &prompt_pho_idx);
      outtree->Branch("bdt_score", &bdt_score);
      outtree->Branch("bdt_indx", &bdt_indx);
      outtree->Branch("kloe_indx", &kloe_indx);
      outtree->Branch("m_gg_bdt", &m_gg_bdt);
      outtree->Branch("m3pi_bdt", &m3pi_bdt);
      outtree->Branch("e1_bdt", &e1_bdt);
      outtree->Branch("e2_bdt", &e2_bdt);

      // Copy original branches if needed
      //outtree -> Branch("E1", &E1);

      cout << "===============Staring Pi0 Selection=============" << endl;
      // Loop over entries
      for (int i = 0; i < nentries; i++) { // loop entries
	tree -> GetEntry(i);

	// Cuts
	//cout << lagvalue_min_7C << endl;
	if (lagvalue_min_7C > 43) continue;
	
	// Clean data
	if (TMath::IsNaN(E1) || TMath::IsNaN(E2) || TMath::IsNaN(E3)) continue;
	//if (!TMath::IsNaN(px1)) continue;
	
	// Store tracks
	double trk[2][4] = {
	  {ppl_E, ppl_px, ppl_py, ppl_pz},
	  {pmi_E, pmi_px, pmi_py, pmi_pz}
	};
	
	//cout << "(ppl_E, ppl_px, ppl_py, ppl_z) = (" << ppl_E << ", " << ppl_px << ", " << ppl_py << ", " << ppl_pz <<  ")" << endl;

	// Store photons
	double photons[3][4] = {
	  {E1, px1, py1, pz1},
	  {E2, px2, py2, pz2},
	  {E3, px3, py3, pz3}
	};
	
	//cout << "(E1, px1, py1, pz1) = (" << E1 << ", " << px1 << ", " << py1 << ", " << pz1 <<  ")" << endl;
	//cout << "(E2, px2, py2, pz2) = (" << E2 << ", " << px2 << ", " << py2 << ", " << pz2 <<  ")" << endl;
	//cout << "(E3, px3, py3, pz3) = (" << E3 << ", " << px3 << ", " << py3 << ", " << pz3 <<  ")" << endl;
        
	// All 3 possible pairs
	int pair_indicies[3][2] = {{0, 1}, {2, 0}, {1, 2}};
	
	/*
	  cout << pair_indicies[0][0] << ", " << pair_indicies[0][1] << "\n"
	  << pair_indicies[1][0] << ", " << pair_indicies[1][1] << "\n"
	  << pair_indicies[2][0] << ", " << pair_indicies[2][1] << "\n\n";  
	*/

	// Stor scores and pi0 masses for each pair
	double scores[3] = {0., 0., 0.};
	double masses[3] = {0., 0., 0.};

	// Calculate BDT score for each pair and store them
	for (int p = 0; p < 3; p++){// perumation
	  int i_idx = pair_indicies[p][0];
	  int j_idx = pair_indicies[p][1];
	  //cout << "(i, j) = (" << i_idx << ", " << j_idx << ")\n"; 

	  // Paired photon energies
	  double e1 = photons[i_idx][0];
	  double e2 = photons[j_idx][0];
	  //cout << "e1 = " << e1 << ", e2 = " << e2 << endl;
	  
	  // Found unpaired photon energy
	  int unpaired_idx = -1;
	  for (int k = 0; k < 3; k++) {
	    if (k != i_idx && k != j_idx) {
	      unpaired_idx = k;
	      break;
	    }
	  }
	  double e3 = photons[unpaired_idx][0];

	  // Fill histos
	  he1 -> Fill(e1);
	  he2 -> Fill(e2);
	  he3 -> Fill(e3);
	  //cout << "i_idx = " << i_idx << ", j_idx = " << j_idx << ", unpaired_idx = " << unpaired_idx << endl;
	  
	  // Calculate features
	  //double opening_angle = 0,  cos_theta = 0;
	  //double E_asym = 0, e_min_x_angle = 0;
	  //double asym_x_angle = 0, E_diff = 0;
	  //cout << opening_angle << endl;

	  if (e1 >= energy_threshold && e2 >= energy_threshold) {// check valid e1 and e2
	    // Invariant mass
	    m_gg = inv_mass_4vector(i_idx, j_idx, photons);
            
	    masses[p] = m_gg;
	    
	    //cout << m3pi << endl;
	    
	    hm_gg -> Fill(m_gg);
	    
	    // cos_theta
	    cos_theta = get_cos_theta(i_idx, j_idx, photons);
	    hcos_theta -> Fill(cos_theta);
	    
	    // opening angle
	    opening_angle = TMath::ACos(cos_theta);
	    hopen_angle -> Fill(opening_angle);
	    
	    //cout << opening_angle << endl;
	    
	    // energy features
	    E_asym = TMath::Abs(e1 - e2) / (e1 + e2 + 1e-10);
	    E_asym = TMath::Max(0.0, TMath::Min(1.0, E_asym));
	    E_diff = TMath::Abs(e1 - e2);
	    e_min_x_angle = TMath::Min(e1, e2) * opening_angle;
	    asym_x_angle = E_asym * opening_angle;
	    
	    //cout << E_asym << endl;
	    //cout << e_min_x_angle << endl;
	    hE_asym -> Fill(E_asym);
	    he_min_x_angle -> Fill(e_min_x_angle);
	    hE_diff -> Fill(E_diff);
	    hasym_x_angle -> Fill(asym_x_angle);
	    
	  }// end e1 & e2 check

	  // Prepare features for BDT
	  std::vector<float> features = {
	    (float)m_gg, (float)opening_angle, (float)cos_theta, 
	    (float)E_asym, (float)e_min_x_angle, 
	    (float)e1, (float)e2, (float)e3, 
	    (float)asym_x_angle, (float)E_diff
	  }; // end preparation of BDT features

	  // Get BDT score
	  // Convert vector to tensor (1 event, n_features)
	  //TMVA::Experimental::RTensor<float> input(dummy.data(), {1, (size_t)n_features});
	  
	  TMVA::Experimental::RTensor<float> input_tensor(features.data(), {1, features.size()});
	  auto result = bdt.Compute(input_tensor);
	  scores[p] = result(0, 0);
	  //cout << "p: " << p << ", score: " << scores[p] << endl;
	  
	} // end permutation

	// Find the best pair (highest BDT score)
	int best_pair = 0;
	if(scores[1] > scores[best_pair]) best_pair = 1;
	if(scores[2] > scores[best_pair]) best_pair = 2;
	
	// Get the indices for the best pair
	int best_i = pair_indicies[best_pair][0];
	int best_j = pair_indicies[best_pair][1];

	// Find prompt photon (the one not in the best pair)
	int prompt_idx = -1;
	for (size_t k = 0; k < 3; k++){
	  if (k != best_i && k != best_j){
	    prompt_idx = k;
	    break;
	  }
	}
	
	//cout << "best pair indices: (" << best_i << ", " << best_j << "), prompt index: " << prompt_idx << endl;

	// Calculate pi0 4-vector
	pi0_pho1_idx = best_i;
	pi0_pho2_idx = best_j;
	prompt_pho_idx = prompt_idx;
	
	e1_bdt = photons[pi0_pho1_idx][0];
	e2_bdt = photons[pi0_pho2_idx][0];
	
	m_gg_bdt = inv_mass_4vector(pi0_pho1_idx, pi0_pho2_idx, photons);
	//m_gg = inv_mass_4vector(0, 1, photons);
	m_gg = masses[0];
	//cout << m_gg << ", " << masses[0] << endl;
	m3pi = inv_3pimass_4vector(0, 1, photons, trk);
	m3pi_bdt = inv_3pimass_4vector(pi0_pho1_idx, pi0_pho2_idx, photons, trk);
        
	//m_gg_bdt = inv_mass_4vector(0, 1, photons);
	hM_gg -> Fill(m_gg);
	hM3pi -> Fill(m3pi);
	hE1 -> Fill(photons[0][0]);
	hE2 -> Fill(photons[1][0]);
        
	bdt_score = scores[best_pair];
        
	//cout << "m_gg_bdt = " << m_gg_bdt << endl;
	
	/*
	// Get scores for other pairs
	int other1 = (best_pair + 1) % 3;
	int other2 = (best_pair + 2) % 3;
	bdt_score_other1 = scores[other1];
	bdt_score_other2 = scores[other2];
        
	// Check if correct (if truth available)
	is_correct = 0;
	if (is_signal == 1) {
	    // You would need truth information about which pair is correct
	    // This depends on your data
	}
	*/

	//out_event = event_id;
	//cout << out_event << endl;
	outtree -> Fill();
        
	// BDT selection
	hM_gg_BDT -> Fill(m_gg_bdt);
	if (scores[best_pair] > 0.5) {
	  n_found ++;
	  hE1_BDT_good -> Fill(e1_bdt);
	  hE2_BDT_good -> Fill(e2_bdt);
	  hM_gg_BDT_good -> Fill(m_gg_bdt);
	  hM3pi_BDT_good -> Fill(m3pi_bdt);

	  //cout << m_gg_bdt << endl;
	  
	  bdt_indx = 1;
	}
	else {
	  hE1_BDT_bad -> Fill(e1_bdt);
	  hE2_BDT_bad -> Fill(e2_bdt);
	  hM_gg_BDT_bad -> Fill(m_gg_bdt);
	  hM3pi_BDT_bad -> Fill(m3pi_bdt);

	  bdt_indx = 0;
	}

	// KLOE selection
	if (recon_indx == 2 && bkg_indx == 1){//  true pi0 gg
	  hE1_good -> Fill(photons[0][0]);
	  hE2_good -> Fill(photons[1][0]);
	  hM_gg_good -> Fill(m_gg);
	  hM3pi_good -> Fill(m3pi);
          
	  evnt_good += 1;
	  kloe_indx = 1;
	}
	else{// false pi0 gg
	  hE1_bad -> Fill(photons[0][0]);
	  hE2_bad -> Fill(photons[1][0]);
	  hM_gg_bad -> Fill(m_gg);
	  hM3pi_bad -> Fill(m3pi);
          
	  evnt_bad += 1;
	  kloe_indx = 0;
	} 
	
	evnt_KLOE += 1;

      } // end loop entries
      
      cout << "===============End Pi0 Selection=============" << endl;

      // Write histograms for this tree
      he1 -> Write(); // permutations
      he2 -> Write();
      he3 -> Write();
      hm_gg -> Write();
      
      hcos_theta -> Write();
      hopen_angle -> Write();
      hE_asym -> Write();
      he_min_x_angle -> Write();
      hE_diff -> Write();
      hasym_x_angle -> Write();
      
      hM3pi_BDT -> Write(); // BDT selection
      hM3pi_BDT_good -> Write();
      hM3pi_BDT_bad -> Write();
      
      hM_gg_BDT -> Write(); 
      hM_gg_BDT_good -> Write();
      hM_gg_BDT_bad -> Write();
      
      hE1_BDT_good -> Write(); 
      hE1_BDT_bad -> Write(); 
      
      hE2_BDT_good -> Write(); 
      hE2_BDT_bad-> Write(); 

      hE1 -> Write(); // KLOE selection
      hE1_good -> Write(); 
      hE1_bad -> Write(); 
      
      hE2 -> Write(); 
      hE2_good -> Write(); 
      hE2_bad -> Write(); 
      
      hM_gg -> Write(); 
      hM_gg_good -> Write();
      hM_gg_bad -> Write();
      
      hM3pi -> Write();
      hM3pi_good -> Write();
      hM3pi_bad -> Write();
      
      
      // Delete hist, canvases to avoid memory leak
      delete he1;
      delete he2;
      delete he3;
      
      delete hm_gg;
      delete hcos_theta;
      delete hopen_angle;
      delete hE_asym;
      delete he_min_x_angle;
      delete hE_diff;
      delete hasym_x_angle;
      
      delete hM3pi_BDT; // BDT selection
      delete hM3pi_BDT_good;
      delete hM3pi_BDT_bad;
      
      delete hM_gg_BDT; 
      delete hM_gg_BDT_good;
      delete hM_gg_BDT_bad;
      
      delete hE1_BDT_good; 
      delete hE1_BDT_bad; 
      
      delete hE2_BDT_good; 
      delete hE2_BDT_bad; 
      
      delete hE1; // KLOE selection
      delete hE1_good; 
      delete hE1_bad; 
      
      delete hE2; 
      delete hE2_good; 
      delete hE2_bad; 
      
      delete hM_gg; 
      delete hM_gg_good;
      delete hM_gg_bad;
      
      delete hM3pi;
      delete hM3pi_good;
      delete hM3pi_bad;
      
      ch_nb ++;
      //cout << ch_nb << endl;
      
      if (ch_nb > 1) {
	cout << ch_nb << ": " << objnm_tree << endl;
	break;
      }
      
    } // end key loop

    outfile -> Write();
    outfile -> Close();
    delete outfile;
    file -> Close();
    
  } // check input root file end
  
} // main program end
  
  

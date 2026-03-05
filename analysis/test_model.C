#include <TMVA/RBDT.hxx>
#include <TMVA/RTensor.hxx>
#include <ROOT/RDataFrame.hxx>
#include <TFile.h>
#include <TTree.h>
#include <TSystem.h>
#include <iostream>

using namespace TMVA::Experimental;

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

void test_model(const char* model_filename = "../training/models/bdt_pi0_TCOMB.root",
                const char* data_filename = "../data/kloe_small_sample.root"){

    // Manually load libraries
    gSystem->Load("libTMVA");
    gSystem->Load("libTMVAUtils");
    
    std::cout << "Testing model_KLOE ..." << std::endl;  // Added std::

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

    /*
    // ===== DEBUG: Find available methods =====
    std::cout << "\n--- Available RBDT methods ---" << std::endl;
    
    TClass* cl = bdt.IsA();
    TList* methods = cl->GetListOfMethods();
    TIter next(methods);
    TMethod* method;
    
    while ((method = (TMethod*)next())) {
        TString name = method->GetName();
        // Look for methods that might give feature count
        if (name.Contains("Feature") || name.Contains("Input") || 
            name.Contains("Dim") || name.Contains("N") ||
            name.Contains("Get")) {
            std::cout << "  " << method->GetName() << "()" << std::endl;
        }
    }
    */
    
    /*
    // ===== DEBUG: Find RBDT-specific methods =====
    std::cout << "\n--- RBDT-specific methods ---" << std::endl;
    
    TClass* cl = bdt.IsA();
    TList* methods = cl->GetListOfMethods();
    TIter next(methods);
    TMethod* method;
    
    // List of ROOT generic methods to ignore
    std::vector<TString> generic_methods = {
        "Streamer", "StreamerNVirtual", "DeclFileName", "ImplFileName", 
        "Class", "Class_Name", "Class_Version", "Dictionary", "IsA",
        "Browse", "DrawClass", "DrawClone", "Dump", "Inspect", "Move",
        "SetDrawOption", "GetDrawOption", "GetObjectInfo", "GetOption",
        "GetUniqueID", "SetUniqueID", "GetObject", "Notify", "Paint",
        "Pop", "Print", "SaveAs", "Write", "Read", "Clone", "Delete"
    };
    
    while ((method = (TMethod*)next())) {
        TString name = method->GetName();
        
        // Skip generic ROOT methods
        bool is_generic = false;
        for (const auto& g : generic_methods) {
            if (name == g) {
                is_generic = true;
                break;
            }
        }
        
        if (!is_generic) {
            std::cout << "  " << name << "()" << std::endl;
        }
    }
    */

    // Test with dummy data (using 10 features from your training)
    const int n_features = 10; // Should known from the training
    std::vector<float> dummy(n_features, 0.5); // Fill vector with 0.5 to each entry

    // Convert vector to tensor (1 event, n_features)
    TMVA::Experimental::RTensor<float> input(dummy.data(), {1, (size_t)n_features});

    // Compute BDT response
    auto output = bdt.Compute(input);
    cout << "Test response: " << output(0, 0) << std::endl;

    // Define histos
    TH1D* he1 = new TH1D("he1", "", 200, 0, 500);
    TH1D* he2 = new TH1D("he2", "", 200, 0, 500);
    TH1D* he3 = new TH1D("he3", "", 200, 0, 500);

    TH1D* hm_gg = new TH1D("hm_gg", "", 200, 0, 1000);

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

        // Get the tree
        TTree* tree = (TTree*)file -> Get("TISR3PI_SIG");
        if (!tree) {
            cout << "Error: Cannot find 'tree' in file" << endl;
            file -> Close();
            return;
        }

        int nentries = tree -> GetEntries();
        cout << "Tree has " << nentries << " entries" << endl;

        // Set branch addres for input features
        double E1, px1, py1, pz1;
        double E2, px2, py2, pz2;
        double E3, px3, py3, pz3;

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

        // Create output file and tree
        TFile* outfile = TFile::Open("output_with_bdt.root", "RECREATE");
        TTree* outtree = new TTree("new_tree", "Tree with BDT response");

        double bdt_response;
        outtree -> Branch("bdt_response", &bdt_response);

        // Copy original branches if needed
        //outtree -> Branch("E1", &E1);

        const double energy_threshold = 5.0;
        int n_found = 0;

        // Loop over entries
        for (int i = 0; i < nentries; i++) {
            tree -> GetEntry(i);

            // Clean data
            if (TMath::IsNaN(E1) || TMath::IsNaN(E2) || TMath::IsNaN(E3)) continue;
            //if (!TMath::IsNaN(px1)) continue;

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
            for (int p = 0; p < 3; p++){
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
                double m_gg = 0, opening_angle = 0,  cos_theta = 0;
                double E_asym = 0, e_min_x_angle = 0;
                double asym_x_angle = 0, E_diff = 0;

                if (e1 >= energy_threshold && e2 >= energy_threshold) {
                    // Invariant mass
                    m_gg = inv_mass_4vector(i_idx, j_idx, photons);
                    masses[p] = m_gg;
                    hm_gg -> Fill(m_gg);
                }


            }

            // Prepare features
            vector<double> features = {
                E1
            };

            //if (i > 10) break; // Fisrt 10 events



        }

    }    

    /*
    // Plot e1, e2 and e3
    TCanvas *c1 = new TCanvas("c1", "Photon Energies Comparison", 800, 600);

    he1->SetLineColor(kRed);
    he1->SetLineWidth(2);
    he1->Draw();

    he2->SetLineColor(kBlue);
    he2->SetLineWidth(2);
    he2->Draw("SAME");

    he3->SetLineColor(kGreen+2);
    he3->SetLineWidth(2);
    he3->Draw("SAME");

    TLegend *leg = new TLegend(0.7, 0.7, 0.9, 0.9);
    leg->AddEntry(he1, "Photon 1", "l");
    leg->AddEntry(he2, "Photon 2", "l");
    leg->AddEntry(he3, "Photon 3", "l");
    leg->Draw();
    */

    // Plot m_gg
    TCanvas *c2 = new TCanvas("c1", "Gamma-gamma Invariant Mass", 800, 600);
    hm_gg -> Draw();

}

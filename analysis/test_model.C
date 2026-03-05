#include <TMVA/RBDT.hxx>
#include <TMVA/RTensor.hxx>
#include <ROOT/RDataFrame.hxx>
#include <TFile.h>
#include <TTree.h>
#include <TSystem.h>
#include <iostream>

using namespace TMVA::Experimental;

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
        double E1, px1;

        tree -> SetBranchAddress("Br_E1", &E1);
        tree -> SetBranchAddress("Br_px1", &px1);
       

        // Create output file and tree
        TFile* outfile = TFile::Open("output_with_bdt.root", "RECREATE");
        TTree* outtree = new TTree("new_tree", "Tree with BDT response");

        double bdt_response;
        outtree -> Branch("bdt_response", &bdt_response);

        // Copy original branches if needed
        //outtree -> Branch("E1", &E1);

        // Loop over entries
        for (int i = 0; i < nentries; i++) {
            tree -> GetEntry(i);

            cout << "(E1) = (" << E1 << endl;
            // Prepare features
            vector<double> features = {
                E1
            };



        }



    }    

}

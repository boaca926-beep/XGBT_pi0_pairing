void plot_bdt(const char* input_filename = "./output_with_bdt.root") {

  gErrorIgnoreLevel = kError;
  TGaxis::SetMaxDigits(4);
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  gStyle->SetFitFormat("6.4g");
  
  cout << "Plotting BDT results ... " << endl;

  // Load input root file: output_with_bdt.root
  std::cout << "Loading input " << input_filename << std::endl;  // Added std::

  // If data file exists, process it with RDataFame
    if(!gSystem -> AccessPathName(input_filename)){
    
        cout << "\nProcessing data file: " << input_filename << endl;

        // Open the root file
        TFile* file = TFile::Open(input_filename);
        if (!file || file -> IsZombie())
        {
            cout << "Error: Cannot open file " << input_filename << endl;
            return;
        }// end open root file

	// Get the tree
        TTree* tree = (TTree*)file -> Get("new_tree");
        if (!tree) {
            cout << "Error: Cannot find 'tree' in file" << endl;
            file -> Close();
            return;
        }

        int nentries = tree -> GetEntries();
        cout << "Tree has " << nentries << " entries" << endl;

	 // Set branch addres for input features
       
    }// end check file existence
  
    
}

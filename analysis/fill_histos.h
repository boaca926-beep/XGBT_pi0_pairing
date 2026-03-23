const char* mc_names[] = {"TDATA",
			  "TEEG",
			  "TOMEGAPI",
			  "TKPM",
			  "TKSL",
			  "TRHOPI",
			  "TETAGAM",
			  "TBKGREST",
			  "T3PIGAM",
			  "TISR3PI_SIG"
};

const int list_size = 3;
const TString ch_type[list_size] = {"TDATA",
				    "TETAGAM",
				    "TISR3PI_SIG"
};

TLine *line1 = new TLine(400., 900., 1000., 900.); // horizontal upper
line1 -> SetLineColor(kRed);
line1 -> SetLineWidth(2);

TLine *line2 = new TLine(400., 650., 1000., 650.); // horizontal lower
line2 -> SetLineColor(kRed);
line2 -> SetLineWidth(2);

TLine *line11 = new TLine(650., 0., 650., 1500.); // vertical left
line11 -> SetLineColor(42);
line11 -> SetLineWidth(4);

TLine *line22 = new TLine(900., 0., 900., 1500.); // vertical right
line22 -> SetLineColor(42);
line22 -> SetLineWidth(4);

  
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

void legtextsize(TLegend* l, Double_t size) {
  for(int i = 0 ; i < l -> GetListOfPrimitives() -> GetSize() ; i++) {
    TLegendEntry *header = (TLegendEntry*)l->GetListOfPrimitives()->At(i);
    header->SetTextSize(size);
  }
}

TLegend* set_legend(const double x1, const double x2, const double y1, const double y2){

  TLegend *legd_cv = new TLegend(x1, x2, y1, y2);

  legd_cv -> SetTextFont(132);
  legd_cv -> SetFillStyle(0);
  legd_cv -> SetBorderSize(0);
  legd_cv -> SetNColumns(1);

  return legd_cv;
}

void format_h(TH1D* h, Int_t linecolor, Int_t width) {
  h->SetLineColor(linecolor);
  h->SetMarkerColor(linecolor);
  //cout << "histo format" << endl;
  h->SetLineWidth(width);
}

void formatfill_h(TH1D* h, Int_t fillcolor, Int_t fillstyle) {
  h -> SetFillStyle(fillstyle);
  h -> SetFillColor(fillcolor);
  h -> SetLineColor(0);
}

TPaveText* set_pt(const double x1, const double x2, const double y1, const double y2) {

  // Create TPaveText
  TPaveText* pt = new TPaveText(x1, x2, y1, y2, "NDC");  // Coordinates in NDC (0-1)
  pt->SetFillColor(0);           // White background
  pt->SetTextColor(kBlack);      // Black text
  pt->SetTextSize(0.04);         // Text size
  //pt->SetBorderSize(1);        // Border size (0 = no border)
  pt->SetTextAlign(12);          // Left aligned, vertically centered
  
  return pt;
}

//
void check_trees(TFile* file) {

  if (!file || file -> IsZombie()){
    cout << "Error: Cannot open file " << file -> GetName() << endl;
      return;
    }// end open root file

    TIter next_tree(file -> GetListOfKeys());

    TString objnm_tree, classnm_tree;
    
    TKey *key;
    
    while ( (key = (TKey *) next_tree() ) ) {// start tree while loop
      
      objnm_tree   =  key -> GetName();
      classnm_tree = key -> GetClassName();
      key -> GetSeekKey();

      if (classnm_tree == "TH2D") { // list all TH2D
      //if (classnm_tree == "TH1D") { // list all TH1D
	cout << "classnm = " << classnm_tree << ", objnm = " << objnm_tree << endl;
      }
      
      TTree *tree_tmp = (TTree*)file -> Get(objnm_tree);
      
    }
    
    // Get histos
    file -> cd(); // Make sure we're in the output file

}






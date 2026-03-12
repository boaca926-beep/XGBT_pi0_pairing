double nb_data_sum = 0.;
double nb_isr3pi_sum = 0.;
double nb_eeg_sum = 0.;
double nb_omegapi_sum = 0.;
double nb_kpm_sum = 0.;
double nb_ksl_sum = 0.;
double nb_rhopi_sum = 0.;
double nb_etagam_sum = 0.;
double nb_mcrest_sum = 0.;
double nb_mcsum = 0.;

double chi2_sfw2d_sum = 0., residul_size_sfw2d = 0.;

// Add a penalty strength parameter (you can adjust this)
double g_penalty_strength = 1000.0;  // Start with this value

TTree* TSFW2D = new TTree("TSFW2D", "recreate");
    

//
double GetScalError(double N_d, double N, double f, double f_error) {
  double error = 0.;
  double scale = N_d * f / N;
  error = scale * TMath::Sqrt(1. / N_d + 1. / N + TMath::Power(f_error / f, 2));

  //cout << "D = " << N_d << ", N = "<< N << ", f = " << f << "+/-" << f_error << ", scale = " << scale << " +/- " << error << endl;

  return error;

}

//
double getloglh(double n_d, double mu) {
  double value = 0.;

  //value = n_d * TMath::Log(mu) - mu + n_d - n_d * TMath::Log(n_d);
  //value = n_d * TMath::Log(mu) - mu - (n_d * TMath::Log(n_d) - n_d + 0.5 * TMath::Log(n_d) + 0.5 * TMath::Log(2 * TMath::Pi()));
  value = n_d * TMath::Log(mu) - mu;
  //cout << "n_d = " << n_d << endl;

  return value;
}

//
double getscale(double Nd, double fra, double N){
  double value = 0.;

  value =  Nd * fra / N;

  return value;
}

//
void fcn_sfw2d(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag){


  int counter = 0;

  double nb_data = 0.;
  double nb_isr3pi = 0.;
  double nb_eeg = 0.;
  double nb_omegapi = 0.;
  double nb_kpm = 0.;
  double nb_ksl = 0.;
  double nb_rhopi = 0.;
  double nb_etagam = 0.;
  double nb_mcrest = 0.;

  double f_isr3pi = par[0];
  double f_eeg = par[1];
  double f_omegapi = par[2];
  double f_kpm = par[3];
  double f_ksl = par[4];
  //double f_rhopi = par[5];
  double f_etagam = par[5];
  double f_mcrest = par[6];

  //cout << "f_isr3pi = " << f_isr3pi << "\n";

  double isr3pi_sfw = 0., isr3pi_mu = 0.;
  double eeg_sfw = 0., eeg_mu = 0.;
  double omegapi_sfw = 0., omegapi_mu = 0.;
  double kpm_sfw = 0., kpm_mu = 0.;
  double ksl_sfw = 0., ksl_mu = 0.;
  double rhopi_sfw = 0., rhopi_mu = 0.;
  double etagam_sfw = 0., etagam_mu = 0.;
  double mcrest_sfw = 0., mcrest_mu = 0.;

  double mu_tmp = 0.;

  double chi2_sum_tmp = 0.;
  double llh_sum = 0.;

  for (Int_t irow = 0; irow < TSFW2D -> GetEntries(); irow++) {// fcn loop
    
    TSFW2D -> GetEntry(irow);

    // data
    nb_data = TSFW2D -> GetLeaf("Br_nb_data") -> GetValue(0);

    // isr3pi
    nb_isr3pi = TSFW2D -> GetLeaf("Br_nb_isr3pi") -> GetValue(0);
    isr3pi_sfw = getscale(nb_data_sum, f_isr3pi, nb_isr3pi_sum);
    isr3pi_mu = nb_isr3pi * isr3pi_sfw;

    // eeg
    nb_eeg = TSFW2D -> GetLeaf("Br_nb_eeg") -> GetValue(0);
    eeg_sfw = getscale(nb_data_sum, f_eeg, nb_eeg_sum);
    eeg_mu = nb_eeg * eeg_sfw;

    // omegapi
    nb_omegapi = TSFW2D -> GetLeaf("Br_nb_omegapi") -> GetValue(0);
    omegapi_sfw = getscale(nb_data_sum, f_omegapi, nb_omegapi_sum);
    omegapi_mu = nb_omegapi * omegapi_sfw;

    // kpm
    nb_kpm = TSFW2D -> GetLeaf("Br_nb_kpm") -> GetValue(0);
    kpm_sfw = getscale(nb_data_sum, f_kpm, nb_kpm_sum);
    kpm_mu = nb_kpm * kpm_sfw;

    // ksl
    nb_ksl = TSFW2D -> GetLeaf("Br_nb_ksl") -> GetValue(0);
    ksl_sfw = getscale(nb_data_sum, f_ksl, nb_ksl_sum);
    ksl_mu = nb_ksl * ksl_sfw;

    // rhopi
    //nb_rhopi = TSFW2D -> GetLeaf("Br_nb_rhopi") -> GetValue(0);
    //rhopi_sfw = getscale(nb_data_sum, f_rhopi, nb_rhopi_sum);
    //rhopi_mu = nb_rhopi * rhopi_sfw;

    // etagam
    nb_etagam = TSFW2D -> GetLeaf("Br_nb_etagam") -> GetValue(0);
    etagam_sfw = getscale(nb_data_sum, f_etagam, nb_etagam_sum);
    etagam_mu = nb_etagam * etagam_sfw;

    // mcrest
    nb_mcrest = TSFW2D -> GetLeaf("Br_nb_mcrest") -> GetValue(0);
    mcrest_sfw = getscale(nb_data_sum, f_mcrest, nb_mcrest_sum);
    mcrest_mu = nb_mcrest * mcrest_sfw;

    //cout << nb_mcrest_sum << endl;

    /*
    cout << "nb_data = " << nb_data << ", sum = " << nb_data_sum << "\n"
	 << "1. nb_isr3pi = " << nb_isr3pi << ", isr3pi_sfw = " << isr3pi_sfw << ", isr3pi_mu = " << isr3pi_mu << ", sum = " << nb_isr3pi_sum  << "\n"
	 << "2. nb_eeg = " << nb_eeg << ", eeg_sfw = " << eeg_sfw << ", eeg_mu = " << eeg_mu << ", sum = " << nb_eeg_sum << "\n"
	 << "3. nb_omegapi = " << nb_omegapi << ", omegapi_sfw = " << omegapi_sfw << ", omegapi_mu = " << omegapi_mu << ", sum = " << nb_omegapi_sum << "\n"
	 << "4. nb_kpm = " << nb_kpm << ", kpm_sfw = " << kpm_sfw << ", kpm_mu = " << kpm_mu << ", sum = " << nb_kpm_sum << "\n"
	 << "5. nb_ksl = " << nb_ksl << ", ksl_sfw = " << ksl_sfw << ", ksl_mu = " << ksl_mu << ", sum = " << nb_ksl_sum << "\n"
	 << "6. nb_etagam = " << nb_etagam << ", etagam_sfw = " << etagam_sfw << ", etagam_mu = " << etagam_mu << ", sum = " << nb_etagam_sum << "\n"
	 << "7. nb_rhopi = " << nb_rhopi << ", rhopi_sfw = " << rhopi_sfw << ", rhopi_mu = " << rhopi_mu << ", sum = " << nb_rhopi_sum << "\n"
      	 << "8. nb_mcrest = " << nb_mcrest << ", mcrest_sfw = " << mcrest_sfw << ", mcrest_mu = " << mcrest_mu << ", sum = " << nb_mcrest_sum << "\n";
    */


    // mu
    mu_tmp = isr3pi_mu + eeg_mu + omegapi_mu + kpm_mu + ksl_mu + etagam_mu + mcrest_mu;

    if (mu_tmp > 0. && nb_data > 0.) {


      counter ++;

      chi2_sum_tmp += (nb_data - mu_tmp) * (nb_data - mu_tmp) / (nb_data + mu_tmp);
      
      llh_sum -= 2. * getloglh(nb_data, mu_tmp);
    
    }
    
  } // end fcn loop

  chi2_sfw2d_sum = chi2_sum_tmp;
  residul_size_sfw2d = counter;

  // Add the sum constraint penalty here

  // Calculate sum of all fractions
  double sum_par = f_isr3pi + f_eeg + f_omegapi + f_kpm + f_ksl + f_etagam + f_mcrest;

  // Add quadratic penalty for deviation from 1.0
  double sum_penalty = g_penalty_strength * (sum_par - 1.0) * (sum_par - 1.0);
  
  // Add penalty for negative parameters (unphysical)
  double neg_penalty = 0.0;
  for (int i = 0; i < 7; i++) {
    if (par[i] < 0.0) {
      neg_penalty += 10000.0 * par[i] * par[i];  // Strong penalty for negatives
    }
  }
  
  // Total FCN = likelihood + constraints
  f = llh_sum + sum_penalty + neg_penalty;
  //f = llh_sum;
  

}


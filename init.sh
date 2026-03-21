#!/bin/bash
# install.sh - Readme and requirements.txt

cd ~/Desktop/XGBT_pi0_pairing

cat > requirements.txt << 'EOF'
awkward>=2.9.0
joblib>=1.5.3
matplotlib>=3.10.8
numpy>=2.4.2
pandas>=3.0.0
psutil>=7.2.2
scikit-learn>=1.8.0
scikit-optimize>=0.10.2
seaborn>=0.13.2
uproot>=5.7.1
xgboost>=3.2.0
EOF


cat > README_NEW.md << 'EOF'
# π⁰ Photon Pairing with XGBoost

ML project to identify π⁰ → γγ decays using XGBoost.

## Quick Start

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train
python training/main_training.py

# Validate
cd validation && python main_validation.py
EOF

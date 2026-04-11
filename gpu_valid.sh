# Run in virtual enviroment
cd /home/bo/Desktop/XGBT_pi0_pairing/training
python -c "
import xgboost as xgb
import numpy as np

# Small test
X = np.random.rand(1000, 10)
y = np.random.randint(0, 2, 1000)

model = xgb.XGBClassifier(
    tree_method='hist',
    device='cuda',
    n_estimators=10
)
model.fit(X, y)
print('✓ GPU training successful!')
"
